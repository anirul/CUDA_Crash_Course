#ifndef MYAPP_HISTOGRAM_HPP
#define MYAPP_HISTOGRAM_HPP

#include <vector>
#include <chrono>

class cuda_histogram
{
public:
    // Constructor that picks a CUDA device (default = 0).
    explicit cuda_histogram(int deviceIndex = 0);
    ~cuda_histogram();

    // Prepare internal buffers for an image of size (width x height).
    void setup(unsigned int width, unsigned int height);

    // Copy the BGRA image data into device memory (input must have 4*width*height bytes).
    void prepare(const std::vector<unsigned char>& inputBGRA);

    // Run the histogram computation, returning how long the main part took.
    std::chrono::nanoseconds run(std::vector<unsigned int>& output);

private:
    // (You can keep these private since they’re implementation details.)
    // Device pointers
    unsigned char *d_bgra_  = nullptr;
    unsigned char *d_lum_   = nullptr;
    unsigned int  *d_part_  = nullptr;
    unsigned int  *d_final_ = nullptr;

    // Image info
    unsigned int width_     = 0;
    unsigned int height_    = 0;
    unsigned int totalSize_ = 0;

    // Kernel launch info
    int blockSize_ = 256;   
    int gridSize_  = 0;     
    int numGroups_ = 0;     

    // Which CUDA device we’re using
    int device_ = 0;
};

#endif // MYAPP_HISTOGRAM_HPP
