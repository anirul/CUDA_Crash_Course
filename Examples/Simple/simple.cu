#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cstdlib>

// A simple CUDA kernel that adds two arrays element-wise
__global__ void simpleKernel(const float* in1, const float* in2, float* out, int size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < size) {
        // Do whatever your OpenCL kernel was doing;
        // for example, just add them:
        out[idx] = in1[idx] + in2[idx];
    }
}

int main(int ac, char** av)
{
    // 1. Query how many CUDA-capable devices are in the system
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    for (int d = 0; d < device_count; ++d)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, d);
        std::cout << "using device   : " << prop.name << std::endl;
        // Set the current device
        cudaSetDevice(d);
        // 3. Prepare host data
        const int vector_size = 1 << 20; // e.g. 1 million elements
        std::vector<float> in1(vector_size), in2(vector_size), out(vector_size);

        for (int i = 0; i < vector_size; i++) {
            in1[i] = static_cast<float>(std::rand());
            in2[i] = static_cast<float>(std::rand());
        }

        // 4. Allocate device memory
        float* d_in1, * d_in2, * d_out;
        cudaMalloc(&d_in1, vector_size * sizeof(float));
        cudaMalloc(&d_in2, vector_size * sizeof(float));
        cudaMalloc(&d_out, vector_size * sizeof(float));

        // 5. Copy data from host to device
        cudaMemcpy(d_in1, in1.data(), vector_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_in2, in2.data(), vector_size * sizeof(float), cudaMemcpyHostToDevice);

        // 6. Configure the kernel launch parameters (grid and block sizes)
        int blockSize = 256; // typical block size
        int gridSize = (vector_size + blockSize - 1) / blockSize;

        // 7. Time the kernel execution
        auto start = std::chrono::system_clock::now();

        // 8. Launch the kernel
        simpleKernel << <gridSize, blockSize >> > (d_in1, d_in2, d_out, vector_size);

        // 9. Wait for the kernel to finish
        cudaDeviceSynchronize();

        auto end = std::chrono::system_clock::now();

        // 10. Copy results back to host
        cudaMemcpy(out.data(), d_out, vector_size * sizeof(float), cudaMemcpyDeviceToHost);

        // 11. Print timing
        std::cout << "Computing time : "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
            << " us" << std::endl;

        // 12. Free device memory
        cudaFree(d_in1);
        cudaFree(d_in2);
        cudaFree(d_out);
    }
    return 0;
}