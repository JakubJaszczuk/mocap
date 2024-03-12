cmake_minimum_required(VERSION 3.28)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_TOOLCHAIN_FILE ./vcpkg/scripts/buildsystems/vcpkg.cmake)

project(mocap)

# Find dependencies
find_package(OpenCV 4.8 REQUIRED COMPONENTS core highgui videoio dnn)

#find_package(onnxruntime 1.14 REQUIRED)
#include_directories(onnxruntime)


# Executable
add_executable(${PROJECT_NAME} main.cpp)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_compile_options(${PROJECT_NAME} PRIVATE -Og -g1 -Werror -Wall -Wextra -pedantic)

# Libs
## OpenCV
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} opencv_core opencv_highgui opencv_videoio opencv_dnn)
