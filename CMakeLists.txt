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
    "RTXCR checkout used by the RTXCR sample; individual submodules are preferred")
set(RTXCR_ASSETS_ROOT "" CACHE PATH
    "RTXCR-Assets checkout used by the licensed RTXCR reference sample")
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

set(_METALLIC_RTXCR_GEOMETRY_CANDIDATES)
if(RTXCR_ROOT)
    list(APPEND _METALLIC_RTXCR_GEOMETRY_CANDIDATES
        "${RTXCR_ROOT}/libraries/rtxcr/geometry")
endif()
list(APPEND _METALLIC_RTXCR_GEOMETRY_CANDIDATES
    "${CMAKE_SOURCE_DIR}/External/RTXCR-Geometry"
    "${CMAKE_SOURCE_DIR}/../RTXCR/libraries/rtxcr/geometry")

set(METALLIC_RTXCR_GEOMETRY_ROOT "")
foreach(_METALLIC_RTXCR_GEOMETRY_CANDIDATE IN LISTS _METALLIC_RTXCR_GEOMETRY_CANDIDATES)
    if(EXISTS "${_METALLIC_RTXCR_GEOMETRY_CANDIDATE}/include/CurveTessellation.h" AND
       EXISTS "${_METALLIC_RTXCR_GEOMETRY_CANDIDATE}/shaders/include/rtxcr/geometry.hlsli")
        get_filename_component(METALLIC_RTXCR_GEOMETRY_ROOT
            "${_METALLIC_RTXCR_GEOMETRY_CANDIDATE}" ABSOLUTE)
        break()
    endif()
endforeach()
unset(_METALLIC_RTXCR_GEOMETRY_CANDIDATE)
unset(_METALLIC_RTXCR_GEOMETRY_CANDIDATES)

if(METALLIC_RTXCR_GEOMETRY_ROOT)
    set(METALLIC_RTXCR_GEOMETRY_SHADER_INCLUDE_DIR
        "${METALLIC_RTXCR_GEOMETRY_ROOT}/shaders/include")
    message(STATUS
        "RTXCR Geometry Library: ${METALLIC_RTXCR_GEOMETRY_ROOT}")
else()
    set(METALLIC_RTXCR_GEOMETRY_SHADER_INCLUDE_DIR "")
    message(WARNING
        "RTXCR Geometry Library was not found. RTXCR curve glTF assets cannot "
        "be converted to DOTS geometry until External/RTXCR-Geometry is initialized.")
endif()

set(_METALLIC_RTXCR_ASSET_CANDIDATES)
if(RTXCR_ASSETS_ROOT)
    list(APPEND _METALLIC_RTXCR_ASSET_CANDIDATES "${RTXCR_ASSETS_ROOT}")
endif()
if(RTXCR_ROOT)
    list(APPEND _METALLIC_RTXCR_ASSET_CANDIDATES "${RTXCR_ROOT}/assets")
endif()
list(APPEND _METALLIC_RTXCR_ASSET_CANDIDATES
    "${CMAKE_SOURCE_DIR}/External/RTXCR-Assets"
    "${CMAKE_SOURCE_DIR}/../RTXCR/assets")

set(METALLIC_RTXCR_ASSETS_ROOT "")
foreach(_METALLIC_RTXCR_ASSET_CANDIDATE IN LISTS _METALLIC_RTXCR_ASSET_CANDIDATES)
    set(_METALLIC_RTXCR_REFERENCE_GROOM
        "${_METALLIC_RTXCR_ASSET_CANDIDATE}/Claire/ponyTail_15vtx.gltf")
    set(_METALLIC_RTXCR_REFERENCE_GROOM_DATA
        "${_METALLIC_RTXCR_ASSET_CANDIDATE}/Claire/ponyTail_15vtx.bin")
    if(EXISTS "${_METALLIC_RTXCR_REFERENCE_GROOM}" AND
       EXISTS "${_METALLIC_RTXCR_REFERENCE_GROOM_DATA}" AND
       EXISTS "${_METALLIC_RTXCR_ASSET_CANDIDATE}/Claire/NVIDIA License for Claire Assets (2024.11.18) [FINAL].pdf")
        file(SIZE "${_METALLIC_RTXCR_REFERENCE_GROOM_DATA}" _METALLIC_RTXCR_GROOM_DATA_SIZE)
        if(_METALLIC_RTXCR_GROOM_DATA_SIZE GREATER 1000)
            get_filename_component(METALLIC_RTXCR_ASSETS_ROOT
                "${_METALLIC_RTXCR_ASSET_CANDIDATE}" ABSOLUTE)
            break()
        endif()
    endif()
endforeach()
unset(_METALLIC_RTXCR_REFERENCE_GROOM)
unset(_METALLIC_RTXCR_REFERENCE_GROOM_DATA)
unset(_METALLIC_RTXCR_GROOM_DATA_SIZE)
unset(_METALLIC_RTXCR_ASSET_CANDIDATE)
unset(_METALLIC_RTXCR_ASSET_CANDIDATES)

if(METALLIC_RTXCR_ASSETS_ROOT)
    message(STATUS "RTXCR licensed reference assets: ${METALLIC_RTXCR_ASSETS_ROOT}")
else()
    message(WARNING
        "RTXCR reference assets were not found, or their Git LFS payloads are "
        "missing. Run `git submodule update --init --recursive -- External/RTXCR-Assets` "
        "and `git -C External/RTXCR-Assets lfs pull` after accepting the included licenses.")
endif()

foreach(METALLIC_SETUP_MODULE IN ITEMS
    SetupSlang
    SetupStreamline
    SetupNsight
    SetupAftermath
    SetupNrc
    SetupNtc
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
