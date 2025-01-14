#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

// Abseil Flags
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

#include "histogram.hpp"

// ----- ABSEIL FLAGS -----

ABSL_FLAG(std::string, input_image, "", "Path to the input image (BGRA or BGR).");
ABSL_FLAG(bool, gpu, true, "Select GPU (true) or CPU (false). [Currently symbolic if you want multiple devices]");
ABSL_FLAG(unsigned int, device, 0, "CUDA device index to use.");
ABSL_FLAG(unsigned int, width,  0, "Image width override (optional).");
ABSL_FLAG(unsigned int, height, 0, "Image height override (optional).");

int main(int argc, char** argv)
{
    // Parse flags
    absl::ParseCommandLine(argc, argv);

    // Read flags into local variables
    std::string input_image = absl::GetFlag(FLAGS_input_image);
    bool gpu               = absl::GetFlag(FLAGS_gpu);
    unsigned int device    = absl::GetFlag(FLAGS_device);
    unsigned int forcedW   = absl::GetFlag(FLAGS_width);
    unsigned int forcedH   = absl::GetFlag(FLAGS_height);

    if (input_image.empty()) {
        std::cerr << "ERROR: --input_image is required.\n";
        return 1;
    }

    // Load image with OpenCV
    cv::Mat frame = cv::imread(input_image, cv::IMREAD_COLOR);
    if (frame.empty()) {
        std::cerr << "Failed to load image: " << input_image << std::endl;
        return 1;
    }

    // Convert to BGRA
    cv::Mat img;
    cv::cvtColor(frame, img, cv::COLOR_BGR2BGRA);

    unsigned int width  = (forcedW > 0) ? forcedW : img.cols;
    unsigned int height = (forcedH > 0) ? forcedH : img.rows;

    std::cout << "Using image size : " << width << " x " << height << std::endl;

    // Copy the BGRA data into a std::vector
    std::vector<unsigned char> vec_img(width * height * 4);
    std::memcpy(vec_img.data(), img.ptr(), vec_img.size());

    try {
        // We’re ignoring the `gpu` flag in this sample—CUDA is GPU-only.
        // If you wanted CPU fallback, you’d need a different path or library.
        cuda_histogram hist(device);
        hist.setup(width, height);
        hist.prepare(vec_img);

        std::vector<unsigned int> out;
        auto duration = hist.run(out);

        std::cout << "Histogram computed in "
                  << std::chrono::duration_cast<std::chrono::microseconds>(duration).count()
                  << " us.\n";

        // (Optional) Print out the histogram
        // for (int i = 0; i < 256; i++) {
        //     std::cout << i << " -> " << out[i] << "\n";
        // }

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
