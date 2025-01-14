# CUDA Crash Course

Simple crash course for CUDA I wrote for the SAW. This contain samples and slides.

## Slides

Thre slides are on goolge slides but there will be a copy of the PDF here.

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
