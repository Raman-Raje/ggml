
# opencl-relu.cpp
set(TEST_TARGET opencl-relu)
add_executable(${TEST_TARGET} opencl-relu.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml)

if (GGML_OPENCL)
    add_compile_definitions(GGML_USE_OPENCL)
endif()