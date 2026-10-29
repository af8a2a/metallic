cmake_minimum_required(VERSION 3.20)
if(POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
endif()
project(Metallic LANGUAGES C CXX)

option(METALLIC_BUILD_DEPENDENCIES_ONLY "Build the reusable dependency package only" OFF)
if(METALLIC_BUILD_DEPENDENCIES_ONLY)
    add_subdirectory(cmake/dependencies)
    return()
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC AND NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
include(cmake/CompilerLauncher.cmake)

option(METALLIC_ENABLE_OPENUSD "Enable USD scene import" ON)
option(METALLIC_ENABLE_NRD "Enable NVIDIA NRD denoising" ON)
option(METALLIC_ENABLE_NVIDIA_SDKS "Enable optional NVIDIA SDK integrations" ON)
option(METALLIC_ENABLE_RTXCR "Enable locally available RTXCR material, geometry and reference assets" ON)
# Override effective values without erasing the user's individual cached choices.
# Setup modules still create their interface targets and unsupported-build stubs.
if(NOT METALLIC_ENABLE_NVIDIA_SDKS)
    foreach(feature IN ITEMS STREAMLINE NV_LOW_LATENCY DLSS_NR NRD
            NSIGHT_EVENTS NSIGHT_GRAPHICS_CAPTURE NSIGHT_AFTERMATH NVPERF NTC RTXCR)
        set(METALLIC_ENABLE_${feature} OFF)
    endforeach()
endif()
set(METALLIC_DEPENDENCY_MODE SOURCE CACHE STRING "Dependency provider: SOURCE or PREBUILT")
set_property(CACHE METALLIC_DEPENDENCY_MODE PROPERTY STRINGS SOURCE PREBUILT)
if(NOT METALLIC_DEPENDENCY_MODE MATCHES "^(SOURCE|PREBUILT)$")
    message(FATAL_ERROR "METALLIC_DEPENDENCY_MODE must be SOURCE or PREBUILT")
endif()
if(METALLIC_DEPENDENCY_MODE STREQUAL "PREBUILT")
    include(cmake/DependencyCache.cmake)
    metallic_dependency_identity()
    file(CONFIGURE OUTPUT "${CMAKE_BINARY_DIR}/metallic-dependencies-expected.txt"
        CONTENT "@METALLIC_DEPENDENCY_IDENTITY@" @ONLY)
    metallic_check_dependency_package()
    list(PREPEND CMAKE_PREFIX_PATH "${METALLIC_DEPENDENCY_ROOT}")
    message(STATUS "Using dependency package: ${METALLIC_DEPENDENCY_ROOT}")
endif()

option(METALLIC_CLUSTER_LOD_TOPOLOGY_NYX
    "Build ClusterLOD topology with the experimental Nyx-style top-level BVH instead of the selector chain"
    OFF)
option(METALLIC_RHI_DIAGNOSTICS "Compile optional Vulkan barrier/submit observation hooks" OFF)
option(METALLIC_BUILD_TESTS "Build Metallic test targets" ON)
option(METALLIC_ENABLE_TRACY "Enable Tracy CPU/GPU profiling" ON)

foreach(METALLIC_SETUP_MODULE IN ITEMS
    SetupSlang
    SetupRTXCR
    SetupStreamline
    SetupDLSSNR
    SetupNsight
    SetupNsightGraphicsCapture
    SetupNvPerf
    SetupAftermath
    SetupNTC
)
    set(METALLIC_SETUP_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/${METALLIC_SETUP_MODULE}.cmake")
    if(EXISTS "${METALLIC_SETUP_MODULE_PATH}")
        include("${METALLIC_SETUP_MODULE_PATH}")
    endif()
endforeach()

find_package(Git QUIET)
option(METALLIC_SUBMODULE_AUTO_DOWNLOAD "Initialize missing required Git submodules" ON)

function(ensure_git_submodule path marker display_name)
    if(EXISTS "${marker}")
        return()
    endif()

    if(NOT METALLIC_SUBMODULE_AUTO_DOWNLOAD OR NOT GIT_FOUND OR NOT EXISTS "${CMAKE_SOURCE_DIR}/.git")
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

foreach(dependency IN ITEMS "imgui|imgui.h" "imnodes|imnodes.h"
        "ImGuizmo|src/ImGuizmo.cpp" "VulkanMemoryAllocator|include/vk_mem_alloc.h"
        "meshoptimizer|src/meshoptimizer.h" "stdexec|include/stdexec/execution.hpp"
        "openpbr-bsdf|impl/data/openpbr_ideal_dielectric_energy_complement_data.h"
        "SDL3|src/video/khronos/vulkan/vulkan.h")
    string(REPLACE "|" ";" dependency_parts "${dependency}")
    list(GET dependency_parts 0 dependency_path)
    list(GET dependency_parts 1 dependency_marker)
    ensure_git_submodule(
        "External/${dependency_path}"
        "${CMAKE_SOURCE_DIR}/External/${dependency_path}/${dependency_marker}"
        "${dependency_path}")
endforeach()
if(METALLIC_DEPENDENCY_MODE STREQUAL "SOURCE")
    ensure_git_submodule("External/spdlog"
        "${CMAKE_SOURCE_DIR}/External/spdlog/CMakeLists.txt" "spdlog")
endif()

ensure_git_submodule(
    "External/microprofile"
    "${CMAKE_SOURCE_DIR}/External/microprofile/stb/stb_sprintf.h"
    "microprofile"
)

if(METALLIC_ENABLE_TRACY)
    ensure_git_submodule(
        "External/tracy"
        "${CMAKE_SOURCE_DIR}/External/tracy/public/tracy/Tracy.hpp"
        "Tracy"
    )
endif()

if(METALLIC_BUILD_TESTS)
    ensure_git_submodule(
        "External/googletest"
        "${CMAKE_SOURCE_DIR}/External/googletest/googletest/include/gtest/gtest.h"
        "GoogleTest"
    )
endif()

ensure_git_submodule(
    "External/Rtxtf"
    "${CMAKE_SOURCE_DIR}/External/Rtxtf/STFSamplerState.hlsli"
    "RTXTF-Library"
)

if(METALLIC_ENABLE_OPENUSD AND METALLIC_DEPENDENCY_MODE STREQUAL "SOURCE")
    ensure_git_submodule(
        "External/oneTBB"
        "${CMAKE_SOURCE_DIR}/External/oneTBB/include/oneapi/tbb.h"
        "oneTBB"
    )
    ensure_git_submodule(
        "External/OpenUSD"
        "${CMAKE_SOURCE_DIR}/External/OpenUSD/pxr/usd/usd/stage.h"
        "OpenUSD"
    )
endif()
include(cmake/DependencyOptions.cmake)
add_subdirectory(External EXCLUDE_FROM_ALL)

# spdlog follows BUILD_SHARED_LIBS even when SPDLOG_BUILD_SHARED is disabled.
# When a shared build is requested, keep every executable self-contained by
# deploying the generated DLL beside it. The utility target is intentionally
# always out of date so rebuilding a focused executable also restores a DLL
# that was removed from its output directory.
function(metallic_copy_spdlog_runtime target_name)
    if(NOT WIN32 OR NOT TARGET ${target_name} OR NOT TARGET spdlog::spdlog)
        return()
    endif()

    get_target_property(spdlog_runtime_target spdlog::spdlog ALIASED_TARGET)
    if(NOT spdlog_runtime_target)
        set(spdlog_runtime_target spdlog::spdlog)
    endif()
    get_target_property(spdlog_target_type ${spdlog_runtime_target} TYPE)
    if(NOT spdlog_target_type STREQUAL "SHARED_LIBRARY")
        return()
    endif()

    set(runtime_target "${target_name}SpdlogRuntime")
    add_custom_target(${runtime_target}
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:${target_name}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:${spdlog_runtime_target}>"
            "$<TARGET_FILE_DIR:${target_name}>"
        DEPENDS ${spdlog_runtime_target}
        COMMENT "Deploying spdlog runtime for ${target_name}"
        VERBATIM
    )
    add_dependencies(${target_name} ${runtime_target})
endfunction()

add_subdirectory(Source)

if(EXISTS "${CMAKE_SOURCE_DIR}/Tools/CMakeLists.txt")
    add_subdirectory(Tools)
endif()

if(METALLIC_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

option(METALLIC_BUILD_SHADER_MATH_HARNESS
    "Build the standalone shader math numeric validation harness (shader-math-harness/)" OFF)
if(METALLIC_BUILD_SHADER_MATH_HARNESS)
    add_subdirectory(shader-math-harness)
endif()
