enable_language(CUDA)

add_executable(FloydWarshall 
    main.cpp
    floyd_warshall.cu
    floyd_warshall.hpp
    ewd_file.cpp
    ewd_file.hpp
)

target_link_libraries(FloydWarshall 
    PRIVATE 
        ${OpenCV_LIBS}
        absl::base
        absl::flags
        absl::flags_parse
        absl::strings
        CUDA::cudart
)
