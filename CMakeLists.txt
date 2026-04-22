cmake_minimum_required(VERSION 3.20)
project(Metallic)

set(CMAKE_CXX_STANDARD 20)

include(cmake/SetupSlang.cmake)
include(cmake/SetupStreamline.cmake)
include(cmake/SetupNsight.cmake)
include(cmake/SetupAftermath.cmake)

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
add_subdirectory(Tools)
