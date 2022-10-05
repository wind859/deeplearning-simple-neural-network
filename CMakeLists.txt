cmake_minimum_required(VERSION 3.10)
project(demo_neural_network VERSION 3.0.0)

add_executable(demo_neural_network main.cpp neural_network.h neural_network.cpp)

set(EXECUTABLE_OUTPUT_PATH ../install/bin)
