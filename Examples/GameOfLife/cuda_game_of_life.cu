#include "cuda_game_of_life.hpp"

#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cuda_runtime.h>

// ===========================
//   Error-check macro
// ===========================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err_ = (call); \
        if (err_ != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ \
                      << " -> " << cudaGetErrorString(err_) << std::endl; \
            throw std::runtime_error("CUDA error."); \
        } \
    } while(0)

// --------------------------------------------------------------------------
//  Kernel: video_image
//
//  - in, out: pointers to byte arrays (uchar4 or grayscale).
//  - width, height: frame size
//  - nb_col: number of channels (1=gray, 4=BGRA).
//    We'll store / interpret data as float for accumulation, but
//    the memory is laid out as unsigned char.
//
//  This replicates your "video_image" kernel logic in CUDA style.
// --------------------------------------------------------------------------
__global__ void video_image_kernel(const unsigned char* in,
                                   unsigned char* out,
                                   int width,
                                   int height,
                                   int nb_col)
{
    // 2D coordinate
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    // index in a row-major 1D array
    int idx = (y * width + x) * nb_col;
    int alive = 0;
    for (int dx = -1; dx < 2; ++dx)
    {
        for (int dy = -1; dy < 2; ++dy)
        {
            if (dx == 0 && dy == 0) continue;
            int iddx = ((y + dy) * width + (x + dx)) * nb_col;
            if (in[iddx] > 127) alive++;
        }
    }
    if (in[idx] > 127) {
        if ((alive == 2) || (alive == 3))
        {
            out[idx] = 255;
        }
        else
        {
            out[idx] = 127;
        }
    }
    else 
    {
        if (alive == 3) 
        {
            out[idx] = 255;
        }
        else
        {
            out[idx] = (in[idx] > 1) ? in[idx] - 1 : 0;
        }
    }
}

// --------------------------------------------------------
//   cuda_game_of_life class Implementation
// --------------------------------------------------------
cuda_game_of_life::cuda_game_of_life(bool gpu, unsigned int deviceIndex)
    : gpu_(gpu)
    , deviceIndex_(deviceIndex)
{
    // Optionally, pick GPU/CPU. In normal CUDA, you only have GPU devices,
    // but you can do device 0 by default. We ignore 'gpu_' here because CUDA
    // doesn't do CPU fallback. 
    selectDevice(deviceIndex_);
}

cuda_game_of_life::~cuda_game_of_life()
{
    // Free device memory
    if (d_in_)  cudaFree(d_in_);
    if (d_out_) cudaFree(d_out_);
}

void cuda_game_of_life::selectDevice(unsigned int deviceIndex)
{
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        throw std::runtime_error("No CUDA devices found.");
    }
    if ((int)deviceIndex >= deviceCount) {
        throw std::runtime_error("Invalid device index.");
    }
    CUDA_CHECK(cudaSetDevice(deviceIndex));
    // (Optional) print device name
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceIndex));
    std::cout << "Using CUDA device: " << prop.name << std::endl;
}

void cuda_game_of_life::setup(const std::pair<unsigned int, unsigned int>& size,
                       unsigned int nb_col)
{
    width_  = size.first;
    height_ = size.second;
    nb_col_ = nb_col;
    totalSize_ = static_cast<size_t>(width_ * height_ * nb_col_);

    // Allocate device memory for in/out frames
    if (d_in_)  cudaFree(d_in_);
    if (d_out_) cudaFree(d_out_);

    CUDA_CHECK(cudaMalloc(&d_in_,  totalSize_));
    CUDA_CHECK(cudaMalloc(&d_out_, totalSize_));
}

void cuda_game_of_life::prepare(const std::vector<char>& input)
{
    // Copy from host to device
    if (input.size() != totalSize_) {
        throw std::runtime_error("Input size doesn't match frame size!");
    }
    CUDA_CHECK(cudaMemcpy(d_in_, input.data(), totalSize_, cudaMemcpyHostToDevice));
}

std::chrono::nanoseconds cuda_game_of_life::run(std::vector<char>& output)
{
    if (output.size() != totalSize_) {
        output.resize(totalSize_);
    }

    // Time the kernel only
    auto t0 = std::chrono::high_resolution_clock::now();

    // launch a 2D kernel
    dim3 block(16, 16);
    dim3 grid((width_ + block.x - 1) / block.x,
              (height_ + block.y - 1) / block.y);

    video_image_kernel<<<grid, block>>>(d_in_, d_out_, width_, height_, nb_col_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t1 = std::chrono::high_resolution_clock::now();

    // Copy result back
    CUDA_CHECK(cudaMemcpy(output.data(), d_out_, totalSize_, cudaMemcpyDeviceToHost));

    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0);
}
