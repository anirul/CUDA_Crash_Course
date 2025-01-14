#include "floyd_warshall.hpp"

#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err_ = (call);                                    \
        if (err_ != cudaSuccess) {                                    \
            std::cerr << "CUDA Error at " << __FILE__ << ":"          \
                      << __LINE__ << " -> " << cudaGetErrorString(err_)\
                      << std::endl;                                   \
            throw std::runtime_error("CUDA error.");                  \
        }                                                             \
    } while(0)

// ----------------------------------------------------------------------
// Kernel: One iteration of Floyd–Warshall. 
// For a pivot "k", each thread updates d_mat[i*n + j] = min(...).
// ----------------------------------------------------------------------
__global__ void fw_iteration_kernel(float* d_mat, int k, int n)
{
    // 2D index (i,j)
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < n) {
        float oldVal = d_mat[i*n + j];
        float altVal = d_mat[i*n + k] + d_mat[k*n + j];
        if (altVal < oldVal) {
            d_mat[i*n + j] = altVal;
        }
    }
}

// ----------------------------------------------------------------------
// cuda_floyd_warshall class
// ----------------------------------------------------------------------

cuda_floyd_warshall::cuda_floyd_warshall(int deviceIndex)
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

    // Optional: Print device name
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceIndex));
    std::cout << "Using device: " << prop.name << std::endl;
}

cuda_floyd_warshall::~cuda_floyd_warshall()
{
    if (d_mat_) {
        cudaFree(d_mat_);
    }
}

void cuda_floyd_warshall::setup(unsigned int n)
{
    n_ = n;
    // Allocate device memory for n*n floats
    CUDA_CHECK(cudaMalloc(&d_mat_, n_ * n_ * sizeof(float)));
}

void cuda_floyd_warshall::prepare(const std::vector<float>& in)
{
    if (in.size() != n_ * n_) {
        throw std::runtime_error("Input size != n*n in prepare()");
    }
    // Copy from host to device
    CUDA_CHECK(cudaMemcpy(d_mat_, in.data(),
                          n_ * n_ * sizeof(float),
                          cudaMemcpyHostToDevice));
}

std::chrono::nanoseconds cuda_floyd_warshall::run(std::vector<float>& out)
{
    if (out.size() != n_ * n_) {
        out.resize(n_ * n_);
    }

    // Start timing the main O(n) loop
    auto start = std::chrono::high_resolution_clock::now();

    // For each pivot k
    for (unsigned int k = 0; k < n_; ++k)
    {
        dim3 block(blockSizeX_, blockSizeY_);
        dim3 grid((n_ + blockSizeX_ - 1) / blockSizeX_,
                  (n_ + blockSizeY_ - 1) / blockSizeY_);

        fw_iteration_kernel<<<grid, block>>>(d_mat_, k, n_);
        CUDA_CHECK(cudaDeviceSynchronize()); // Sync each iteration
    }

    auto end = std::chrono::high_resolution_clock::now();

    // Copy final result to host
    CUDA_CHECK(cudaMemcpy(out.data(),
                          d_mat_,
                          n_ * n_ * sizeof(float),
                          cudaMemcpyDeviceToHost));

    return (end - start);
}
