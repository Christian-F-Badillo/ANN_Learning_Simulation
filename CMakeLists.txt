cmake_minimum_required(VERSION 3.10)
project(ArtNeuralNetworkSim)

set(CMAKE_CXX_STANDARD 17)

find_package(raylib REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(ArtNeuralNetworkSim src/main.cpp)
target_link_libraries(ArtNeuralNetworkSim  PRIVATE raylib)
