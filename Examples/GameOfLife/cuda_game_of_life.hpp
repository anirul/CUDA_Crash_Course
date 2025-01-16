#ifndef CUDA_VIDEO_HEADER_DEFINED
#define CUDA_VIDEO_HEADER_DEFINED

#include <vector>
#include <utility>      // for std::pair
#include <chrono>       // for timing

class cuda_game_of_life
{
public:
    // Constructor picks a GPU device index (default = 0)
    cuda_game_of_life(bool gpu, unsigned int deviceIndex = 0);
    ~cuda_game_of_life();

    // Sets up the frame size (width, height) and number of color channels (1 or 4)
    void setup(const std::pair<unsigned int, unsigned int>& size,
               unsigned int nb_col);

    // Copies the input frame from host memory to device
    void prepare(const std::vector<char>& input);

    // Runs the filter kernel, copies the result back to 'output'.
    // Returns the elapsed time for the kernel call (CPU-side measurement).
    std::chrono::nanoseconds run(std::vector<char>& output);

private:
    // Internal fields to mirror your old CL code
    bool gpu_ = true;
    unsigned int deviceIndex_ = 0;

    unsigned int width_  = 0;
    unsigned int height_ = 0;
    unsigned int nb_col_ = 0;     // 1 for gray, 4 for BGRA

    // Device pointers
    unsigned char* d_in_  = nullptr;  // device input frame
    unsigned char* d_out_ = nullptr;  // device output frame
    size_t totalSize_ = 0;            // width * height * nb_col_

    // Helper to pick a CUDA device, etc.
    void selectDevice(unsigned int deviceIndex);
};

#endif // CUDA_VIDEO_HEADER_DEFINED
