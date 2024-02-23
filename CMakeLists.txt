cmake_minimum_required(VERSION 3.28)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(mocap)


# Dependencies
find_package(OpenCV 4.8 REQUIRED COMPONENTS core highgui videoio dnn)
include_directories(${OpenCV_INCLUDE_DIRS})


# Executable
add_executable(${PROJECT_NAME} main.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_23)
target_compile_options(${PROJECT_NAME} PRIVATE -Og -g1 -Werror -Wall -Wextra -pedantic)
target_link_libraries(${PROJECT_NAME} opencv_core opencv_highgui opencv_videoio opencv_dnn)
