#ifndef CUDA_FLOYD_WARSHALL_HPP
#define CUDA_FLOYD_WARSHALL_HPP

#include <vector>
#include <chrono>

class cuda_floyd_warshall
{
public:
    // Constructor picks a CUDA device (default = 0)
    explicit cuda_floyd_warshall(int deviceIndex = 0);
    ~cuda_floyd_warshall();

    // Setup the matrix size (n x n), allocate device memory
    void setup(unsigned int n);

    // Copy a distance matrix from host to device
    // 'in' must have n*n elements
    void prepare(const std::vector<float>& in);

    // Run the Floyd–Warshall algorithm on the GPU
    // Returns how long the main O(n) pivot loop took.
    // Copies the final matrix to 'out' (size n*n).
    std::chrono::nanoseconds run(std::vector<float>& out);

private:
    int device_ = 0;       // Which CUDA device
    unsigned int n_ = 0;   // Matrix is n x n

    float* d_mat_ = nullptr;  // Device pointer to distance matrix (size n*n floats)

    // Example block size for 2D kernel launch
    int blockSizeX_ = 16;
    int blockSizeY_ = 16;
};

#endif // CUDA_FLOYD_WARSHALL_HPP
