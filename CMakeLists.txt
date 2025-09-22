cmake_minimum_required(VERSION 3.16)
project(buflib LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -diag-suppress 177")

find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs imgproc cudaimgproc cudaarithm)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/buf)
set(MAIN_SOURCES buf/bufcore.cu buf/bufapi.cu main.cpp)

# test
add_executable(buftst ${MAIN_SOURCES})
target_link_libraries(buftst PRIVATE ${OpenCV_LIBS})
set_target_properties(buftst PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)
target_compile_options(buftst PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress 177>)

# timing
set(TIMING_SOURCES buf/bufcore.cu buf/bufapi.cu timing.cpp)
add_executable(timing ${TIMING_SOURCES})
target_link_libraries(timing PRIVATE ${OpenCV_LIBS})
set_target_properties(timing PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)
target_compile_options(timing PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress 177>)

