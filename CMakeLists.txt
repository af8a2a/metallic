cmake_minimum_required(VERSION 3.20)
project(Metallic)

set(CMAKE_CXX_STANDARD 20)

include(cmake/SetupSlang.cmake)

add_subdirectory(External)
add_subdirectory(Source)
add_subdirectory(Tools)
