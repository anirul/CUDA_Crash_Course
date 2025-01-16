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
//  Constants similar to your OpenCL kernel
// --------------------------------------------------------------------------
static const int  RADIUS          = 5;
static const int  POWER_RADIUS    = RADIUS * RADIUS;
static const int  INTENSITY_LEVEL = 6;

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

    // We'll accumulate color in float4 if nb_col=4, or single float if nb_col=1.
    // But let's unify by reading as float4 in memory—if nb_col=1, we store grayscale
    // in .x and set the rest to 0.
    // For simplicity, let's convert to float at load, do the same logic, then
    // store back as unsigned char.

    // We'll define a helper to load a pixel from in[] as float4
    auto loadPixel = [&](int xx, int yy) -> float4 {
        if (xx < 0 || xx >= width || yy < 0 || yy >= height) {
            return make_float4(0, 0, 0, 1);
        }
        int i2 = (yy * width + xx) * nb_col;
        if (nb_col == 4) {
            // BGRA
            float b = in[i2 + 0] / 255.0f;
            float g = in[i2 + 1] / 255.0f;
            float r = in[i2 + 2] / 255.0f;
            float a = in[i2 + 3] / 255.0f;
            return make_float4(r, g, b, a);
        }
        else {
            // Gray
            float g = in[i2] / 255.0f;
            return make_float4(g, g, g, 1.0f);
        }
    };

    // We'll define a helper to store a float4 result back to out[] as BGRA or grayscale
    auto storePixel = [&](int iOut, float4 c){
        // clamp to [0,1]
        float r = fminf(fmaxf(c.x, 0.0f), 1.0f);
        float g = fminf(fmaxf(c.y, 0.0f), 1.0f);
        float b = fminf(fmaxf(c.z, 0.0f), 1.0f);
        // for grayscale, we just store intensity in out[iOut]
        if (nb_col == 4) {
            out[iOut + 0] = static_cast<unsigned char>(b * 255.0f);
            out[iOut + 1] = static_cast<unsigned char>(g * 255.0f);
            out[iOut + 2] = static_cast<unsigned char>(r * 255.0f);
            out[iOut + 3] = 255; // alpha
        } else {
            // simple average for grayscale
            float gray = 0.3333f*(r+g+b);
            out[iOut]   = static_cast<unsigned char>(gray * 255.0f);
        }
    };

    // The logic from your kernel:
    //   1) Gather color in a radius around (x, y).
    //   2) Bucket by intensity into INTENSITY_LEVEL bins.
    //   3) Pick the bin with the maximum count, average its color.

    int intensity_count[INTENSITY_LEVEL];
    float4 average_color[INTENSITY_LEVEL];

    for (int i = 0; i < INTENSITY_LEVEL; ++i) {
        intensity_count[i] = 0;
        average_color[i]   = make_float4(0,0,0,0);
    }

    // gather
    for (int xx = x - RADIUS; xx <= x + RADIUS; ++xx) {
        for (int yy = y - RADIUS; yy <= y + RADIUS; ++yy) {
            int dx = x - xx;
            int dy = y - yy;
            if ((dx*dx + dy*dy) > POWER_RADIUS) {
                continue;
            }
            float4 c = loadPixel(xx, yy);
            // intensity = average of (r,g,b)
            float intensity = (c.x + c.y + c.z) / 3.0f;
            int bin = (int)(intensity * INTENSITY_LEVEL);
            if (bin >= INTENSITY_LEVEL) bin = INTENSITY_LEVEL - 1;

            intensity_count[bin] += 1;
            average_color[bin].x += c.x;
            average_color[bin].y += c.y;
            average_color[bin].z += c.z;
            average_color[bin].w += c.w; // though alpha not used for bin picking
        }
    }

    // pick the bin with the highest count
    int max_count = 0;
    int max_bin   = 0;
    for (int i = 0; i < INTENSITY_LEVEL; i++) {
        if (intensity_count[i] > max_count) {
            max_count = intensity_count[i];
            max_bin   = i;
        }
    }

    // final color = average_color[max_bin] / max_count
    float4 finalCol;
    if (max_count > 0) {
        finalCol.x = average_color[max_bin].x / max_count;
        finalCol.y = average_color[max_bin].y / max_count;
        finalCol.z = average_color[max_bin].z / max_count;
        finalCol.w = 1.0f;
    } else {
        finalCol = make_float4(0,0,0,1);
    }

    storePixel(idx, finalCol);
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
