cmake_minimum_required(VERSION 3.16)
Project(PA_MH)

# Instalación de armadillo
find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# C++ estándar
set(CMAKE_CXX_STANDARD 11)

# Cabeceras
include_directories(${PROJECT_SOURCE_DIR}/include)

# Fuentes
file(GLOB_RECURSE SOURCE_FILES ${PROJECT_SOURCE_DIR}/src/*)

# Directorio de ejecutables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../BIN)

# Flags del compilador
set(CMAKE_CXX_FLAGS "-std=c++11 -larmadillo -O3 -fopenmp")

# Ejecutable
add_executable(PA_MH ${SOURCE_FILES})

target_link_libraries(PA_MH ${ARMADILLO_LIBRARIES})
