cmake_minimum_required(VERSION 3.20)
project(Metallic LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(METALLIC_CLUSTER_LOD_TOPOLOGY_NYX
    "Build ClusterLOD topology with the experimental Nyx-style top-level BVH instead of the selector chain"
    OFF)

foreach(METALLIC_SETUP_MODULE IN ITEMS
    SetupSlang
    SetupStreamline
    SetupNsight
    SetupAftermath
)
    set(METALLIC_SETUP_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/${METALLIC_SETUP_MODULE}.cmake")
    if(EXISTS "${METALLIC_SETUP_MODULE_PATH}")
        include("${METALLIC_SETUP_MODULE_PATH}")
    endif()
endforeach()

find_package(Git QUIET)

function(ensure_git_submodule path marker display_name)
    if(EXISTS "${marker}")
        return()
    endif()

    if(NOT GIT_FOUND OR NOT EXISTS "${CMAKE_SOURCE_DIR}/.git")
        message(FATAL_ERROR
            "${display_name} dependency is missing at ${path}.\n"
            "Run `git submodule update --init --recursive -- ${path}` and reconfigure.")
    endif()

    message(STATUS "Initializing ${display_name} submodule in ${path}")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" submodule update --init --recursive -- "${path}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        RESULT_VARIABLE submodule_result
        OUTPUT_VARIABLE submodule_output
        ERROR_VARIABLE submodule_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )

    if(NOT submodule_result EQUAL 0 OR NOT EXISTS "${marker}")
        message(FATAL_ERROR
            "Failed to initialize ${display_name} in ${path}.\n"
            "${submodule_output}\n${submodule_error}")
    endif()
endfunction()

ensure_git_submodule(
    "External/microprofile"
    "${CMAKE_SOURCE_DIR}/External/microprofile/stb/stb_sprintf.h"
    "microprofile"
)

add_subdirectory(External)
add_subdirectory(Source)

if(EXISTS "${CMAKE_SOURCE_DIR}/Tools/CMakeLists.txt")
    add_subdirectory(Tools)
endif()
