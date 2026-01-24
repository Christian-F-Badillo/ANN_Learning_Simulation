cmake_minimum_required(VERSION 3.10)
project(ArtNeuralNetworkSim)

set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SOURCE "src/*.cpp")

include_directories(src)

find_package(raylib REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


add_executable(ArtNeuralNetworkSim ${SOURCES})
target_link_libraries(ArtNeuralNetworkSim  PRIVATE raylib)
