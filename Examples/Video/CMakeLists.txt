enable_language(CUDA)

add_executable(Video
    main.cpp
    cuda_video.cu
    cuda_video.hpp
)

target_link_libraries(Video
    PRIVATE
        ${OpenCV_LIBS}
        absl::base
        absl::flags
        absl::flags_parse
        absl::strings
        CUDA::cudart
)
