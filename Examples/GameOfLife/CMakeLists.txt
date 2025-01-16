enable_language(CUDA)

add_executable(GameOfLife
    main.cpp
    cuda_game_of_life.cu
    cuda_game_of_life.hpp
)

target_link_libraries(GameOfLife
    PRIVATE 
        ${OpenCV_LIBS}
        absl::base
        absl::flags
        absl::flags_parse
        absl::strings
        CUDA::cudart
)
