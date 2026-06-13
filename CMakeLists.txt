cmake_minimum_required(VERSION 3.10)
project(track_to_centerline_tool LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies the reconstructor needs.
find_package(Eigen3 REQUIRED)
find_package(GSL REQUIRED)

# Build the reconstructor library from the submodule
add_subdirectory(lib/track-reconstructor/cpp track_reconstructor_build)

add_executable(track_to_centerline tools/track_to_centerline.cpp)
target_link_libraries(track_to_centerline PRIVATE track_reconstructor)
target_include_directories(track_to_centerline PRIVATE
    lib/track-reconstructor/cpp/include)