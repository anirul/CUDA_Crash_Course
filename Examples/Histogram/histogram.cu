#include "histogram.hpp"

#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>  // for cudaMalloc, cudaMemcpy, etc.

// ================== //
//   ERROR CHECK MACRO
// ================== //
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            std::cerr << "CUDA Error at " << __FILE__ << ":"          \
                      << __LINE__ << " -> " << cudaGetErrorString(err)\
                      << std::endl;                                   \
            throw std::runtime_error("CUDA error.");                  \
        }                                                             \
    } while(0)


// -----------------------------------------------------------
// KERNELS
// -----------------------------------------------------------

__global__ void kernelLuminosity(const unsigned char* bgra, 
                                 unsigned char* lum,
                                 int width,
                                 int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = width * height;
    if (idx < size) {
        // BGRA: 4 bytes/pixel: (B=0, G=1, R=2, A=3)
        unsigned char b = bgra[4 * idx + 0];
        unsigned char g = bgra[4 * idx + 1];
        unsigned char r = bgra[4 * idx + 2];
        float val = 0.299f * r + 0.587f * g + 0.114f * b;
        lum[idx] = static_cast<unsigned char>(val);
    }
}

__global__ void kernelInit(unsigned int* partialHist, int length)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        partialHist[idx] = 0u;
    }
}

__global__ void kernelPartial(const unsigned char* lum,
                              unsigned int* partial,
                              int size,
                              int numGroups)
{
    int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int groupId  = blockIdx.x;
    if (globalId < size) {
        unsigned char val = lum[globalId];
        int offset = groupId * 256;
        atomicAdd(&partial[offset + val], 1);
    }
}

__global__ void kernelReduce(const unsigned int* partial,
                             int numGroups,
                             unsigned int* finalHist)
{
    int bin = threadIdx.x; // assume blockDim.x == 256, blockIdx.x == 0
    if (bin < 256) {
        unsigned int sum = 0;
        for (int g = 0; g < numGroups; g++) {
            sum += partial[g * 256 + bin];
        }
        finalHist[bin] = sum;
    }
}


// -----------------------------------------------------------
// CLASS IMPLEMENTATION
// -----------------------------------------------------------

cuda_histogram::cuda_histogram(int deviceIndex)
    : device_(deviceIndex)
{
    // Select CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA devices found.");
    }
    if (deviceIndex >= deviceCount) {
        throw std::runtime_error("Invalid device index.");
    }

    CUDA_CHECK(cudaSetDevice(deviceIndex));

    // (Optional) Print device name
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceIndex));
    std::cout << "Using device   : " << prop.name << std::endl;
}

cuda_histogram::~cuda_histogram()
{
    if (d_bgra_)  cudaFree(d_bgra_);
    if (d_lum_)   cudaFree(d_lum_);
    if (d_part_)  cudaFree(d_part_);
    if (d_final_) cudaFree(d_final_);
}

void cuda_histogram::setup(unsigned int width, unsigned int height)
{
    width_  = width;
    height_ = height;
    totalSize_ = width_ * height_;

    // 1D block/grid for the main kernels
    gridSize_ = (totalSize_ + blockSize_ - 1) / blockSize_;
    numGroups_ = gridSize_;  // each block is a group

    // Allocate device buffers
    CUDA_CHECK(cudaMalloc(&d_bgra_,  4 * totalSize_ * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_lum_,   totalSize_     * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_part_,  256 * numGroups_ * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_final_, 256 * sizeof(unsigned int)));
}

void cuda_histogram::prepare(const std::vector<unsigned char>& inputBGRA)
{
    if (inputBGRA.size() != 4u * totalSize_) {
        throw std::runtime_error("Input BGRA size != width*height*4");
    }

    // Copy image data to device
    CUDA_CHECK(cudaMemcpy(d_bgra_,
                          inputBGRA.data(),
                          4 * totalSize_ * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));
}

std::chrono::nanoseconds cuda_histogram::run(std::vector<unsigned int>& output)
{
    // 1) Convert to luminosity
    {
        dim3 block(blockSize_);
        dim3 grid(gridSize_);
        kernelLuminosity<<<grid, block>>>(d_bgra_, d_lum_, width_, height_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto start = std::chrono::high_resolution_clock::now();

    // 2) Init partial hist
    {
        int length = 256 * numGroups_;
        int initGrid = (length + blockSize_ - 1) / blockSize_;
        kernelInit<<<initGrid, blockSize_>>>(d_part_, length);
        CUDA_CHECK(cudaGetLastError());
    }

    // 3) Build partial hist
    {
        dim3 block(blockSize_);
        dim3 grid(gridSize_);
        kernelPartial<<<grid, block>>>(d_lum_, d_part_, totalSize_, numGroups_);
        CUDA_CHECK(cudaGetLastError());
    }

    // 4) Reduce partials into final hist
    {
        // 256 threads in one block
        kernelReduce<<<1, 256>>>(d_part_, numGroups_, d_final_);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    // Copy final histogram to host
    output.resize(256);
    CUDA_CHECK(cudaMemcpy(output.data(), d_final_, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    return (end - start);
}
