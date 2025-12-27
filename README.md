# CUDA Crash Course

Simple crash course for CUDA I wrote for the SAW. This contain samples and slides.

## Build

### Prerequired

You will need some prerequire before using this project.

1. a CUDA compatible card (NVidia)
2. an up to date driver (466 here)
3. a version of visual studio that works (VS 2022 here)
4. an up to date versin of [CMake](https://cmake.org/download/)
5. an up to date version of [CUDA](https://developer.nvidia.com/cuda-downloads)

### Install the software

Clone and use the new cmake to create the repo. In case you are using linux just replace `windows` with `linux-release` or `linux-debug`, and then the build with the same.

```pwrsh
git clone https://github.com/anirul/CUDA_Crash_Course.git
cd CUDA_Crash_Course
git submodule update --init --recursive
cmake --preset windows
```

You can now either use VS2022 or just build!

```pwrsh
cmake --build --preset windows-release
```

## Slides

The slides are on google slides but there will be a copy of the PDF here.

## Examples

### Dependencies

I use some dependencies:

- Abseil
- CMake
- SDL
- OpenGL
- OpenCV
- CUDA

### Simple (C++)

Simple example of how to use CUDA with C/C++.

### Floyd Warshall (C++ - abseil)

Implementing a fast version of [Floyd Warshall](http://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm). This demonstrate how to recurse and iterate on arrays. Also multiple call to the same kernel.

### Histogram (C++ - abseil / OpenCV)

Local memory working group and advance structures in CUDA. Basic notion of dispatching computation and reducing it in another kernel (2 kernel operation).

### Game of life (C++ - abseil / OpenGL)

An implementation in CUDA of [Conway's game of life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) in CUDA using OpenGL for displaying the result (with some decaying colors).

### Video (C++ - abseil / OpenCV / OpenGL)

Get video from OpenCV make modification with CUDA and send it back to OpenGL to draw it on screen. It come from my own example. It shows some interaction between OpenCV, CUDA and OpenGL.
