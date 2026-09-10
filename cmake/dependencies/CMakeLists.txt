cmake_minimum_required(VERSION 3.27)
project(MetallicDependencies LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(MSVC AND NOT DEFINED CMAKE_MSVC_RUNTIME_LIBRARY)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()
include("${CMAKE_CURRENT_LIST_DIR}/../CompilerLauncher.cmake")
option(METALLIC_ENABLE_OPENUSD "Include OpenUSD and oneTBB in the dependency package" ON)
set(METALLIC_DEPENDENCY_JOBS 8 CACHE STRING "Maximum parallel compiler jobs per dependency")
include("${CMAKE_CURRENT_LIST_DIR}/../DependencyCache.cmake")
metallic_dependency_identity()
message(STATUS "Metallic dependency package: ${METALLIC_DEPENDENCY_ROOT}")

if(EXISTS "${METALLIC_DEPENDENCY_ROOT}/metallic-dependencies.txt")
    metallic_check_dependency_package()
    add_custom_target(MetallicDependencies ALL
        COMMAND "${CMAKE_COMMAND}" -E echo "Reusing dependency package ${METALLIC_DEPENDENCY_HASH}")
    return()
endif()

include(ExternalProject)
# Use a cache file so list-valued flags and launcher paths survive spaces and
# semicolons. Both the ABI key and the child builds use these exact settings.
set(initial_cache "")
string(TOUPPER "${CMAKE_BUILD_TYPE}" configuration)
foreach(variable IN ITEMS CMAKE_C_COMPILER CMAKE_CXX_COMPILER CMAKE_MAKE_PROGRAM
        CMAKE_BUILD_TYPE CMAKE_CXX_STANDARD CMAKE_CXX_STANDARD_REQUIRED CMAKE_CXX_EXTENSIONS
        CMAKE_MSVC_RUNTIME_LIBRARY CMAKE_MSVC_DEBUG_INFORMATION_FORMAT
        CMAKE_C_COMPILER_LAUNCHER CMAKE_CXX_COMPILER_LAUNCHER
        CMAKE_C_COMPILER_TARGET CMAKE_CXX_COMPILER_TARGET CMAKE_TOOLCHAIN_FILE
        CMAKE_C_FLAGS CMAKE_CXX_FLAGS CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS CMAKE_STATIC_LINKER_FLAGS
        CMAKE_C_FLAGS_${configuration} CMAKE_CXX_FLAGS_${configuration}
        CMAKE_EXE_LINKER_FLAGS_${configuration} CMAKE_SHARED_LINKER_FLAGS_${configuration}
        CMAKE_STATIC_LINKER_FLAGS_${configuration}
        CMAKE_OSX_ARCHITECTURES CMAKE_OSX_DEPLOYMENT_TARGET CMAKE_OSX_SYSROOT CMAKE_SYSROOT)
    if(DEFINED ${variable})
        string(APPEND initial_cache "set(${variable} [==[${${variable}}]==] CACHE STRING \"\" FORCE)\n")
    endif()
endforeach()
set(recipe_root "${CMAKE_CURRENT_LIST_DIR}")
set(package_build "${CMAKE_CURRENT_BINARY_DIR}/${METALLIC_DEPENDENCY_HASH}")
file(MAKE_DIRECTORY "${package_build}")
file(CONFIGURE OUTPUT "${package_build}/InitialCache.cmake" CONTENT "@initial_cache@" @ONLY)

set(previous_target "")
function(metallic_add_dependency name)
    set(source "${METALLIC_REPOSITORY_ROOT}/External/${name}")
    if(NOT EXISTS "${source}/CMakeLists.txt")
        message(FATAL_ERROR "Initialize External/${name} with git submodule update --init --recursive -- External/${name}")
    endif()
    ExternalProject_Add(${name}
        SOURCE_DIR "${source}"
        BINARY_DIR "${package_build}/${name}"
        PREFIX "${package_build}/${name}-steps"
        DOWNLOAD_COMMAND "" UPDATE_COMMAND ""
        CMAKE_GENERATOR "${CMAKE_GENERATOR}"
        CMAKE_ARGS
            -C "${package_build}/InitialCache.cmake"
            "-DCMAKE_INSTALL_PREFIX:PATH=${METALLIC_DEPENDENCY_ROOT}"
            "-DCMAKE_PREFIX_PATH:PATH=${METALLIC_DEPENDENCY_ROOT}"
            "-DCMAKE_INSTALL_LIBDIR:STRING=lib"
            "-DCMAKE_POLICY_VERSION_MINIMUM:STRING=3.5"
            "-DCMAKE_POLICY_DEFAULT_CMP0141:STRING=NEW"
            "-DCMAKE_PROJECT_INCLUDE:FILEPATH=${recipe_root}/ConfigureDependency.cmake"
            ${ARGN}
        BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --parallel "${METALLIC_DEPENDENCY_JOBS}"
        INSTALL_COMMAND "${CMAKE_COMMAND}" --install <BINARY_DIR>
        DEPENDS ${previous_target}
        USES_TERMINAL_BUILD TRUE)
    set(previous_target "${name}" PARENT_SCOPE)
endfunction()

# Serialize projects to bound memory use; each project still compiles in parallel.
metallic_add_dependency(SDL3 -DSDL_SHARED=ON -DSDL_STATIC=OFF -DSDL_INSTALL=ON)
metallic_add_dependency(spdlog -DBUILD_SHARED_LIBS=OFF -DSPDLOG_BUILD_SHARED=OFF
    -DSPDLOG_BUILD_EXAMPLE=OFF -DSPDLOG_BUILD_TESTS=OFF -DSPDLOG_INSTALL=ON)
if(METALLIC_ENABLE_OPENUSD)
    metallic_add_dependency(oneTBB -DBUILD_SHARED_LIBS=ON -DTBB_INSTALL=ON)
    metallic_add_dependency(OpenUSD -DBUILD_SHARED_LIBS=ON -DPXR_FIND_TBB_IN_CONFIG=ON)
endif()
file(CONFIGURE OUTPUT "${package_build}/metallic-dependencies.txt"
    CONTENT "@METALLIC_DEPENDENCY_IDENTITY@" @ONLY)
# Publish the completion marker only after every library and resource is installed.
add_custom_target(MetallicDependencies ALL
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
        "${package_build}/metallic-dependencies.txt"
        "${METALLIC_DEPENDENCY_ROOT}/metallic-dependencies.txt"
    DEPENDS ${previous_target}
    VERBATIM)
