cmake_minimum_required(VERSION 3.14)

project(BrainSim VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------------------
# 1. GESTIÓN DE DEPENDENCIAS (RAYLIB) - OPTIMIZADO
# ------------------------------------------------------------------------------
include(FetchContent)

message(STATUS "Configurando Raylib (esto puede tardar un poco la primera vez)...")

FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG        master
    GIT_SHALLOW    TRUE   # <--- CLAVE: Descarga rápida (sin historial)
    GIT_PROGRESS   TRUE   # <--- Muestra progreso en la consola
)
FetchContent_MakeAvailable(raylib)

# ------------------------------------------------------------------------------
# 2. DEFINICIÓN DEL EJECUTABLE PRINCIPAL
# ------------------------------------------------------------------------------
# Busca todos los .cpp en src/ y el main.cpp
file(GLOB_RECURSE SOURCES "src/*.cpp" "main.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

# ------------------------------------------------------------------------------
# 3. INCLUDES Y LINKEO (PRINCIPAL)
# ------------------------------------------------------------------------------
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Enlazar con Raylib
target_link_libraries(${PROJECT_NAME} PRIVATE raylib)

# ------------------------------------------------------------------------------
# 4. CONFIGURACIÓN ESPECÍFICA POR SO
# ------------------------------------------------------------------------------
if (WIN32)
  set_target_properties(${PROJECT_NAME} PROPERTIES WIN32_EXECUTABLE ON)
  target_compile_definitions(${PROJECT_NAME} PRIVATE NOMINMAX)
endif()

# ------------------------------------------------------------------------------
# 5. ASSETS
# ------------------------------------------------------------------------------
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# ------------------------------------------------------------------------------
# 6. TESTS UNITARIOS
# ------------------------------------------------------------------------------
enable_testing()

function(add_brain_test test_name source_file)
  add_executable(${test_name} ${source_file})

  target_include_directories(${test_name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

  # Enlazar tests con lo necesario (a veces necesitan raylib si usan Math de raylib)
  target_link_libraries(${test_name} PRIVATE raylib)

  set_target_properties(${test_name} PROPERTIES CXX_STANDARD 20)
  add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Tus tests
add_brain_test(test_mat_ops           tests/test_matmul.cpp)
add_brain_test(test_dense_layer       tests/test_dense_layer.cpp)
add_brain_test(test_dense_math        tests/test_dense_math.cpp)
add_brain_test(test_sequential        tests/test_sequential.cpp)
add_brain_test(test_cost_func         tests/test_cost_func.cpp)
add_brain_test(test_model_lregression tests/test_model_lregression.cpp)
