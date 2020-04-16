cmake_minimum_required(VERSION 3.14)
project(Test)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

## ncnn
add_library(ncnn STATIC IMPORTED )
set_target_properties(ncnn
        PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/lib/libncnn.a)
include_directories(ncnn/include)

aux_source_directory(src/ DIR_SRCS)

add_executable(Test ${DIR_SRCS})
target_link_libraries(Test ${OpenCV_LIBS} ncnn)
