# CUDA Crash Course

Simple crash course for CUDA I wrote for the SAW. This contain samples and slides.

## Build

### Prerequired

You will need some prerequire before using this project.
1. a CUDA compatible card (NVidia).
2. an up to date driver (466 here).
3. a version of visual studio that works (VS 2022 here).
4. an up to date versin of [CMake](https://cmake.org/download/)
5. an up to date version of [CUDA](https://developer.nvidia.com/cuda-downloads)
6. a version of [VCPKG](https://vcpkg.io/)

### Install the software

Check your installation of VCPKG and create this in the same directory than your VCPKG repo (I mean parallel).

```
\
.vcpkg
.CUDA_Crash_Course
```

Now clone and use the new cmake to create the repo.

```pwrsh
https://github.com/anirul/CUDA_Crash_Course.git
cd CUDA_Crash_Course
cd Examples
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE="../../../vcpkg/scripts/buildsystems/vcpkg.cmake"
```

You can now either use VS2022 or just build!

```pwrsh
cmake --build . --config Release
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
- CUDA 11

### Simple (C++)

Simple example of how to use CUDA with C/C++.

### Floyd Warshall (C++ - abseil)

Implementing a fast version of [Floyd Warshall](http://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm). This demonstrate how to recurse and iterate on arrays. Also multiple call to the same kernel.

### Histogram (C++ - abseil / OpenCV)

Local memory working group and advance structures in CUDA. Basic notion of dispatching computation and reducing it in another kernel (2 kernel operation).

### Video (C++ - abseil / OpenCV / OpenGL)

Get video from OpenCV make modification with CUDA and send it back to OpenGL to draw it on screen. It come from my own example. It shows some interaction between OpenCV, CUDA and OpenGL.
