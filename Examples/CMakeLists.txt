cmake_minimum_required(VERSION 3.30)

project(cuda-crash-course DESCRIPTION "CUDA Crash Course." LANGUAGES CXX CUDA)

# C++ standard version.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(absl CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_subdirectory(Simple)
add_subdirectory(Histogram)
