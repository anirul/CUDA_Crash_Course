#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <cstdlib>

// Abseil flags
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include <opencv2/opencv.hpp>

// The CUDA video filter class (from our example)
#include "cuda_game_of_life.hpp"

// ---------- Abseil Flags ----------
// For example, replicate the flags from your old code:
ABSL_FLAG(unsigned int, device, 0, "CUDA device index.");
ABSL_FLAG(unsigned int, seed, 0, "random seed for the initial grid");

int main(int argc, char** argv)
{
    // Parse all abseil flags from the command line
    absl::ParseCommandLine(argc, argv);

    // Gather flag values in local variables
    unsigned int device = absl::GetFlag(FLAGS_device);
    unsigned int seed = absl::GetFlag(FLAGS_seed);

    srand(seed);

    std::cout << "CUDA device    : " << device << "\n\n";

    try {
        
        // Frame size
        unsigned int width  = 1280;
        unsigned int height = 720;
        unsigned int nb_col = 1;

        std::cout << "Frame size     : " << width << " x " << height << "\n";
        std::cout << "Channels       : " << nb_col << "\n";

        // 3) Allocate a host buffer for the frame data
        std::vector<char> hostFrame(width * height * nb_col);

        // 4) Create our CUDA video object
        cuda_game_of_life cgol(device);
        // In CUDA, we don’t “compile from .cl,” but we keep the call for similarity
        cgol.setup({width, height}, nb_col);

        // 5) Transfer the first frame
        // Randomize the start value
        for (auto& c : hostFrame)
        {
            c = std::clamp(rand() % 256, 0, 255);
        }
        // Prepare on GPU
        cgol.prepare(hostFrame);

        // 6) Main loop: read frames, process, display
        cv::namedWindow("CUDA Filter", cv::WINDOW_AUTOSIZE);

        bool keepRunning = true;
        while (keepRunning) {

            // d) CUDA: prepare + run kernel
            cgol.prepare(hostFrame);
            auto durationNs = cgol.run(hostFrame);

            // 1 channel
            cv::Mat displayMat(height, width, CV_8UC1, hostFrame.data());
            cv::imshow("CUDA Filter", displayMat);
            
            // f) Print timing or handle keys
            auto ms = std::chrono::duration_cast<std::chrono::microseconds>(durationNs).count();
            std::cout << "Frame processed in " << ms << " ms\n";

            int key = cv::waitKey(1);
            if (key == 27) { // ESC
                keepRunning = false;
            }
        }
        cv::destroyAllWindows();
    } 
    catch (const std::exception& e) {
        std::cerr << "Exception : " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
