enable_language(CUDA)

add_executable(Simple simple.cu)

target_link_libraries(Simple 
    PRIVATE 
        absl::base
        absl::flags
        absl::flags_parse
        absl::strings
        CUDA::cudart
)