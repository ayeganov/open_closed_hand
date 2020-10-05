cmake_minimum_required(VERSION 3.15)

project(HandDetect VERSION 0.0.1)
execute_process(COMMAND conan install ${CMAKE_SOURCE_DIR} --build=missing -pr=hand_detect)

# Specify the output directories for libraries and binaries
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add conan fetched packges
include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

#Find_Package(OpenCV REQUIRED)

# Use the package PkgConfig to detect GTK+ headers/library files
FIND_PACKAGE(PkgConfig REQUIRED)
PKG_CHECK_MODULES(GTK3 REQUIRED gtk+-3.0)

# Setup CMake to use GTK+, tell the compiler where to look for headers
# and to the linker where to look for libraries
INCLUDE_DIRECTORIES(${GTK3_INCLUDE_DIRS})
LINK_DIRECTORIES(${GTK3_LIBRARY_DIRS})

# Add other flags to the compiler
ADD_DEFINITIONS(${GTK3_CFLAGS_OTHER})


set(CMAKE_CXX_FLAGS_RELEASE "--static")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(histogram histogram.cpp)
target_link_libraries(histogram ${CONAN_LIBS} ${GTK3_LIBRARIES})
