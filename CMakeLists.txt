cmake_minimum_required(VERSION 3.20)
project(rendergraph)

set(CMAKE_CXX_STANDARD 20)

include(cmake/DownloadSlang.cmake)

list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/External/slang")
find_package(slang REQUIRED)

add_subdirectory(External)
add_subdirectory(Source)
