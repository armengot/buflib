cmake_minimum_required(VERSION 3.18)
project(buflib LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core highgui imgcodecs imgproc cudaimgproc cudaarithm)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/buf)

set(SOURCES
    buf/bufcore.cu
    buf/bufapi.cu
    main.cpp
)

add_executable(buf_test ${SOURCES})
target_link_libraries(buf_test PRIVATE ${OpenCV_LIBS})
set_target_properties(buf_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 17)
