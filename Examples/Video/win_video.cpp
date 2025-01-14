/*
 * Copyright (c) 2014, Frederic Dubouchet
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Calodox nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Frederic Dubouchet ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Frederic DUBOUCHET BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <chrono> // <-- Use C++ chrono instead of Boost date_time

#ifdef __linux__
#include <GL/glut.h>
#include <GL/gl.h>
#include <GL/glu.h>
#endif
#ifdef __APPLE__
#include <glut/glut.h>
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#endif
#ifdef WIN32
#define NOMINMAX
#include <windows.h>
#include <GL/GL.h>
#include <GL/glut.h>
#endif

#include "cuda_video.hpp"    // <-- Our CUDA-based class
#include "glut_win.hpp"
#include "win_video.hpp"

#ifndef GL_BGRA
#define GL_BGRA     0x80E1
#endif

win_video::win_video(
    const std::pair<unsigned int, unsigned int>& size,
    const std::vector<char> initial_image,
    std::function<bool(std::vector<char>&)> callback,
    const std::string& cl_file,
    bool color,
    bool gpu,
    unsigned int device)
    : range_(size),
    video_(nullptr),
    color_(color),
    gpu_(gpu),
    device_(device),
    texture_id_(0),
    cl_file_(cl_file),
    callback_(callback)
{
    // Convert 60 minutes to nanoseconds as a large initial "best" threshold
    best_time_ = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::minutes(60));
    current_image_ = initial_image;
}

win_video::~win_video()
{
}

// inherited from the i_win interface
void win_video::init()
{
    glClearColor(0, 0, 0, 0);
    gluOrtho2D(-1, 1, -1, 1);
    glGenTextures(1, &texture_id_);

    // Create our CUDA video object instead of an OpenCL object.
    video_ = new cuda_video(gpu_, device_);

    // In CUDA, init() is typically a no-op, but we call it for consistency.
    // 'cl_file_' is not actually used in CUDA, but we pass it anyway.
    video_->init(cl_file_);

    // Setup with dimensions and channels (4 = BGRA, 1 = grayscale)
    video_->setup(range_, (color_ ? 4 : 1));
}

void win_video::display()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture_id_);

    glPushMatrix();
    glBegin(GL_QUADS);
    {
        glTexCoord2f(0, 1); glVertex2f(-1, 1);
        glTexCoord2f(1, 1); glVertex2f(1, 1);
        glTexCoord2f(1, 0); glVertex2f(1, -1);
        glTexCoord2f(0, 0); glVertex2f(-1, -1);
    }
    glEnd();
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);
    glFlush();
    glutPostRedisplay();
}

void win_video::idle()
{
    glFinish();
    if (video_) {
        // 1) Get the next frame from the user-supplied callback
        callback_(current_image_);

        // 2) Copy it to GPU and run the CUDA kernel
        video_->prepare(current_image_);
        // 'run' returns a std::chrono::nanoseconds
        auto actual_time = video_->run(current_image_);

        // Update our best time if this frame is faster
        if (actual_time < best_time_) {
            best_time_ = actual_time;
        }

        // Print timing in milliseconds (or any other unit you prefer)
        auto ms_current = std::chrono::duration_cast<std::chrono::milliseconds>(actual_time).count();
        auto ms_best = std::chrono::duration_cast<std::chrono::milliseconds>(best_time_).count();

        std::cout << "\rCompute time    : " << ms_current << " ms"
            << " | best : " << ms_best << " ms";
        std::cout.flush();
    }

    // 3) Update the texture with the latest GPU -> CPU result
    glBindTexture(GL_TEXTURE_2D, texture_id_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    if (color_) {
        // BGRA => 4 channels
        glTexImage2D(GL_TEXTURE_2D,
            0,
            GL_RGBA,
            range_.first,
            range_.second,
            0,
            GL_BGRA,
            GL_UNSIGNED_BYTE,
            current_image_.data());
    }
    else {
        // Grayscale => 1 channel
        glTexImage2D(GL_TEXTURE_2D,
            0,
            GL_LUMINANCE,
            range_.first,
            range_.second,
            0,
            GL_LUMINANCE,
            GL_UNSIGNED_BYTE,
            current_image_.data());
    }

    glFinish();
}

void win_video::reshape(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glViewport(0, 0, w, h);
    glMatrixMode(GL_MODELVIEW);
    glFinish();
}

void win_video::mouse_event(int button, int state, int x, int y)
{
    // Not used
}

void win_video::mouse_move(int x, int y)
{
    // Not used
}

void win_video::keyboard(unsigned char key, int x, int y)
{
    // Not used
}

void win_video::finish()
{
    if (video_) {
        delete video_;
        video_ = nullptr;
    }
    std::cout << std::endl;
}
