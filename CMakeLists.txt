cmake_minimum_required(VERSION 3.20)
project(Metallic LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Ninja parses MSVC /showIncludes output to track header dependencies. Force
# English diagnostics so the dependency prefix remains byte-stable even when
# the configure shell and build shell use different Windows code pages.
if(MSVC AND CMAKE_GENERATOR MATCHES "Ninja")
    set(CMAKE_C_COMPILER_LAUNCHER
        "${CMAKE_COMMAND};-P;${CMAKE_SOURCE_DIR}/cmake/RunMsvcCompiler.cmake;--")
    set(CMAKE_CXX_COMPILER_LAUNCHER
        "${CMAKE_COMMAND};-P;${CMAKE_SOURCE_DIR}/cmake/RunMsvcCompiler.cmake;--")
    set(CMAKE_C_CL_SHOWINCLUDES_PREFIX "Note: including file:  ")
    set(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX "Note: including file:  ")
    set(CMAKE_CL_SHOWINCLUDES_PREFIX "Note: including file:  ")
endif()

option(METALLIC_CLUSTER_LOD_TOPOLOGY_NYX
    "Build ClusterLOD topology with the experimental Nyx-style top-level BVH instead of the selector chain"
    OFF)
option(METALLIC_BUILD_TESTS "Build Metallic test targets" ON)

set(RTXCR_ROOT "" CACHE PATH
    "RTXCR checkout or RTXCR Material Library root used by the RTXCR sample")
set(_METALLIC_RTXCR_CANDIDATES)
if(RTXCR_ROOT)
    list(APPEND _METALLIC_RTXCR_CANDIDATES
        "${RTXCR_ROOT}"
        "${RTXCR_ROOT}/libraries/rtxcr/material")
endif()
list(APPEND _METALLIC_RTXCR_CANDIDATES
    "${CMAKE_SOURCE_DIR}/External/RTXCR-Material"
    "${CMAKE_SOURCE_DIR}/../RTXCR/libraries/rtxcr/material")

set(METALLIC_RTXCR_MATERIAL_ROOT "")
foreach(_METALLIC_RTXCR_CANDIDATE IN LISTS _METALLIC_RTXCR_CANDIDATES)
    if(EXISTS "${_METALLIC_RTXCR_CANDIDATE}/shaders/include/rtxcr/HairChiangBSDF.hlsli" AND
       EXISTS "${_METALLIC_RTXCR_CANDIDATE}/shaders/include/rtxcr/SubsurfaceScattering.hlsli")
        get_filename_component(METALLIC_RTXCR_MATERIAL_ROOT
            "${_METALLIC_RTXCR_CANDIDATE}" ABSOLUTE)
        break()
    endif()
endforeach()
unset(_METALLIC_RTXCR_CANDIDATE)
unset(_METALLIC_RTXCR_CANDIDATES)

if(METALLIC_RTXCR_MATERIAL_ROOT)
    set(METALLIC_RTXCR_SHADER_INCLUDE_DIR
        "${METALLIC_RTXCR_MATERIAL_ROOT}/shaders/include")
    message(STATUS
        "RTXCR Material Library: ${METALLIC_RTXCR_MATERIAL_ROOT}")
else()
    set(METALLIC_RTXCR_SHADER_INCLUDE_DIR "")
    message(WARNING
        "RTXCR Material Library was not found. MetallicRtxcrSample will build, "
        "but its shader cannot run until RTXCR_ROOT points to an RTXCR checkout "
        "or RTXCR-Material-Library is placed in External/RTXCR-Material.")
endif()

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

ensure_git_submodule(
    "External/googletest"
    "${CMAKE_SOURCE_DIR}/External/googletest/googletest/include/gtest/gtest.h"
    "GoogleTest"
)

ensure_git_submodule(
    "External/Rtxtf"
    "${CMAKE_SOURCE_DIR}/External/Rtxtf/STFSamplerState.hlsli"
    "RTXTF-Library"
)
add_subdirectory(External)
add_subdirectory(Source)

if(EXISTS "${CMAKE_SOURCE_DIR}/Tools/CMakeLists.txt")
    add_subdirectory(Tools)
endif()

if(METALLIC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
