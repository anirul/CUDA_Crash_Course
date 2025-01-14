enable_language(CUDA)

add_executable(Histogram 
    main.cpp
    histogram.hpp
    histogram.cu
)

target_link_libraries(Histogram 
    PRIVATE 
        ${OpenCV_LIBS}
        absl::base
        absl::flags
        absl::flags_parse
        absl::strings
        CUDA::cudart
)
