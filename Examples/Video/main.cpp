#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <stdexcept>

// Abseil flags
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

// OpenCV
#include <opencv2/opencv.hpp>

// The CUDA video filter class (from our example)
#include "cuda_video.hpp"

// ---------- Abseil Flags ----------
// For example, replicate the flags from your old code:
ABSL_FLAG(std::string, input_video, "",
          "Path to input video file (if empty, capture from webcam).");
ABSL_FLAG(bool, gpu, true, "Enable GPU (in this example, always CUDA).");
ABSL_FLAG(unsigned int, device, 0, "CUDA device index.");
ABSL_FLAG(bool, black_white, false, "Enable black & white mode (1 channel).");

int main(int argc, char** argv)
{
    // Parse all abseil flags from the command line
    absl::ParseCommandLine(argc, argv);

    // Gather flag values in local variables
    std::string inputVideo = absl::GetFlag(FLAGS_input_video);
    bool gpu                = absl::GetFlag(FLAGS_gpu);
    unsigned int device     = absl::GetFlag(FLAGS_device);
    bool colorMode          = !absl::GetFlag(FLAGS_black_white);

    std::cout << "---------------------------------------\n";
    std::cout << " CUDA Video Example with Abseil Flags\n";
    std::cout << "---------------------------------------\n";
    if (!inputVideo.empty()) {
        std::cout << "Input video    : " << inputVideo << "\n";
    } else {
        std::cout << "Input video    : Webcam (device 0)\n";
    }
    std::cout << "Mode           : " << (colorMode ? "Color (BGRA)" : "Black & White") << "\n";
    std::cout << "GPU?           : " << (gpu ? "Yes" : "No (but this example is CUDA-only)") << "\n";
    std::cout << "CUDA device    : " << device << "\n\n";

    try {
        // 1) Open the video (file or webcam)
        cv::VideoCapture video;
        if (!inputVideo.empty()) {
            video.open(inputVideo);
        } else {
            video.open(0); // webcam
        }
        if (!video.isOpened()) {
            throw std::runtime_error("Could not open video: " + inputVideo);
        }

        // 2) Grab one frame to determine size
        cv::Mat frame;
        if (!video.read(frame)) {
            throw std::runtime_error("Could not read the first frame from video.");
        }

        // Convert to desired color format
        cv::Mat converted;
        if (!colorMode) {
            // single channel (gray)
            cv::cvtColor(frame, converted, cv::COLOR_BGR2GRAY);
        } else {
            // 4 channels (BGRA) — OpenCV is typically BGR, so we add alpha
            cv::cvtColor(frame, converted, cv::COLOR_BGR2BGRA);
        }

        // Frame size
        unsigned int width  = converted.cols;
        unsigned int height = converted.rows;
        unsigned int nb_col = (colorMode ? 4 : 1);

        std::cout << "Frame size     : " << width << " x " << height << "\n";
        std::cout << "Channels       : " << nb_col << "\n";

        // 3) Allocate a host buffer for the frame data
        std::vector<char> hostFrame(width * height * nb_col);

        // 4) Create our CUDA video object
        cuda_video cvid(gpu, device);
        // In CUDA, we don’t “compile from .cl,” but we keep the call for similarity
        cvid.init(""); // empty string
        cvid.setup({width, height}, nb_col);

        // 5) Transfer the first frame
        // Copy the Mat to our hostFrame buffer
        std::memcpy(hostFrame.data(), converted.ptr(), hostFrame.size());
        // Prepare on GPU
        cvid.prepare(hostFrame);

        // 6) Main loop: read frames, process, display
        cv::namedWindow("CUDA Filter", cv::WINDOW_AUTOSIZE);

        bool keepRunning = true;
        while (keepRunning) {
            // a) We already have the last frame processed, so display that result
            //    But let's do it after the kernel runs (below)...

            // b) Grab the next frame
            if (!video.read(frame)) {
                std::cout << "End of video or cannot read frame.\n";
                break;
            }

            // Flip or do any transformations you want (like your original code):
            // cv::flip(frame, frame, 0); // example if you want to flip vertically

            // Convert to grayscale or BGRA as needed
            if (!colorMode) {
                cv::cvtColor(frame, converted, cv::COLOR_BGR2GRAY);
            } else {
                cv::cvtColor(frame, converted, cv::COLOR_BGR2BGRA);
            }

            // c) Copy it into hostFrame
            std::memcpy(hostFrame.data(), converted.ptr(), hostFrame.size());

            // d) CUDA: prepare + run kernel
            cvid.prepare(hostFrame);
            auto durationNs = cvid.run(hostFrame);

            // e) Convert hostFrame back to cv::Mat for display
            if (!colorMode) {
                // 1 channel
                cv::Mat displayMat(height, width, CV_8UC1, hostFrame.data());
                cv::imshow("CUDA Filter", displayMat);
            } else {
                // 4 channels BGRA
                cv::Mat displayMat(height, width, CV_8UC4, hostFrame.data());
                cv::imshow("CUDA Filter", displayMat);
            }

            // f) Print timing or handle keys
            auto ms = std::chrono::duration_cast<std::chrono::microseconds>(durationNs).count();
            std::cout << "Frame processed in " << ms << " ms\n";

            int key = cv::waitKey(1);
            if (key == 27) { // ESC
                keepRunning = false;
            }
        }

        video.release();
        cv::destroyAllWindows();
    } 
    catch (const std::exception& e) {
        std::cerr << "Exception : " << e.what() << std::endl;
        return -1;
    }
    return 0;
}
