#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <stdexcept>
#include <cstdlib>

// Abseil flags
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"

// Our CUDA Floyd–Warshall
#include "floyd_warshall.hpp"

// The EWD parser
#include "ewd_file.hpp"

// ---------- Abseil Flags ----------
ABSL_FLAG(bool, gpu, true, "Compute using GPU (CUDA).");
ABSL_FLAG(unsigned int, device, 0, "CUDA device index.");
ABSL_FLAG(std::string, file_in,  "rome99.txt", "Input graph file (EWD format).");
ABSL_FLAG(std::string, file_out, "",           "Output graph file (EWD format).");
ABSL_FLAG(unsigned int, loops,   1,  "Number of loops to run (take best time).");

int main(int argc, char** argv)
{
    absl::ParseCommandLine(argc, argv);

    bool enable_gpu   = absl::GetFlag(FLAGS_gpu);
    unsigned int dev  = absl::GetFlag(FLAGS_device);
    std::string fin   = absl::GetFlag(FLAGS_file_in);
    std::string fout  = absl::GetFlag(FLAGS_file_out);
    unsigned int nloop = absl::GetFlag(FLAGS_loops);

    std::cout << "Floyd-Warshall (CUDA) Example\n";
    std::cout << "GPU: " << (enable_gpu ? "enabled" : "disabled") 
              << " (In this example, we only do CUDA if GPU=true)\n";
    std::cout << "Device index: " << dev << "\n";
    std::cout << "Input file  : " << fin << "\n";
    if (!fout.empty()) {
        std::cout << "Output file : " << fout << "\n";
    }
    std::cout << "Loops       : " << nloop << "\n\n";

    try {
        // 1) Read the EWD file
        ewd_file ef;
        ef.import_file(fin);
        size_t n = ef.size();

        std::cout << "Graph size   : " << n << "\n";

        // 2) Export graph -> distance matrix
        std::vector<float> dist(n*n, 0.0f);
        ef.export_matrix(dist.data(), dist.size());

        // (Optional) If it's small, print it:
        if (dist.size() <= 64) { // e.g. 8x8
            ef.print_matrix(std::cout);
        }

        // 3) Create the cuda_floyd_warshall object
        //    (We ignore the 'enable_gpu' flag in this example because we do CUDA-only.)
        cuda_floyd_warshall cfw(dev);
        cfw.setup(static_cast<unsigned int>(n));

        // We'll keep track of the best time
        std::chrono::nanoseconds bestTime(std::chrono::minutes(60));

        // 4) Loop multiple times if requested
        std::vector<float> result; // for final output
        for (unsigned int i = 0; i < nloop; i++) {

            // a) Reload the input matrix
            cfw.prepare(dist);  
            //  (We reload each loop so each iteration starts from original distances.)

            // b) Run
            auto t0 = std::chrono::steady_clock::now();
            auto loopTime = cfw.run(result);
            auto t1 = std::chrono::steady_clock::now();

            // c) Compare times
            if (loopTime < bestTime) {
                bestTime = loopTime;
            }

            auto totalTime = t1 - t0;
            std::cout << "Iteration [" << (i+1) << "/" << nloop << "]\n"
                      << "  - Loop time : "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(loopTime).count()
                      << " ms (for the k-loop)\n"
                      << "  - Total time: "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(totalTime).count()
                      << " ms (including host/device copies)\n";
        }

        // 5) Print best time
        std::cout << "\nBest loop time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(bestTime).count()
                  << " ms\n";

        // 6) If requested, write final results
        //    'result' contains the last run's final matrix.  If you want the best
        //    run's result, you'd store that separately. For simplicity, we assume
        //    they all produce the same final matrix (Floyd–Warshall is deterministic).
        ef.import_matrix(result.data(), result.size());

        if (!fout.empty()) {
            ef.export_file(fout);
        }
        if (result.size() <= 64) {
            std::cout << "Final matrix:\n";
            ef.print_matrix(std::cout);
        }

    } catch (std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return 0;
}
