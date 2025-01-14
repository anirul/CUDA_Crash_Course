enable_language(CUDA)

add_executable(Video
    main.cpp
    cuda_video.cu
    cuda_video.hpp
    glut_win.cpp
    glut_win.hpp
    win_video.cpp
    win_video.hpp
)

target_link_libraries(Video
    PRIVATE 
        GLUT::GLUT
        ${OpenCV_LIBS}
        absl::base
        absl::flags
        absl::flags_parse
        absl::strings
        CUDA::cudart
)
