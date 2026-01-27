cmake_minimum_required(VERSION 3.10)
project(ArtNeuralNetworkSim)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(raylib REQUIRED)
find_package(OpenMP)

file(GLOB_RECURSE LIB_SOURCES "src/*.cpp")

# Filter the main
list(FILTER LIB_SOURCES EXCLUDE REGEX "src/main.cpp")

# CORE lib
add_library(SimCore STATIC ${LIB_SOURCES})

target_include_directories(SimCore PUBLIC src)

# Linker
target_link_libraries(SimCore PUBLIC raylib)
if(OpenMP_CXX_FOUND)
  target_link_libraries(SimCore PUBLIC OpenMP::OpenMP_CXX)
endif()

# Main App
add_executable(ArtNeuralNetworkSim src/main.cpp)
# Linker
target_link_libraries(ArtNeuralNetworkSim PRIVATE SimCore)


# TESTS
enable_testing()

# Tensor Test
add_executable(test_tensor tests/test_tensor.cpp)
target_link_libraries(test_tensor PRIVATE SimCore)

# Test Math
add_executable(test_ops tests/test_math_tensor.cpp)
target_link_libraries(test_ops PRIVATE SimCore)

# Test Matrix Ops
add_executable(test_mat_ops tests/test_matmul.cpp)
target_link_libraries(test_mat_ops PRIVATE SimCore)
