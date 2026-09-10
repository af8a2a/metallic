if(APPLE)
    add_library(metal-cpp INTERFACE)
    target_include_directories(metal-cpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/metal-cpp)
    target_link_libraries(metal-cpp INTERFACE
        "-framework Metal"
        "-framework Foundation"
        "-framework QuartzCore"
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/meshoptimizer_build/CMakeLists.txt")
    add_subdirectory(meshoptimizer_build)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/stdexec/include/stdexec/execution.hpp")
    find_package(Threads REQUIRED)

    add_library(stdexec INTERFACE)
    add_library(STDEXEC::stdexec ALIAS stdexec)
    target_include_directories(stdexec SYSTEM INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/stdexec/include"
    )
    target_link_libraries(stdexec INTERFACE Threads::Threads)
    target_compile_features(stdexec INTERFACE cxx_std_20)
    target_compile_options(stdexec INTERFACE
        $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fcoroutines;-fconcepts-diagnostics-depth=10>
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/Zc:__cplusplus;/Zc:preprocessor;/Zc:externConstexpr;/bigobj>
    )
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/entt/entt/entity/registry.hpp")
    add_library(metallic_entt INTERFACE)
    add_library(EnTT::EnTT ALIAS metallic_entt)
    target_include_directories(metallic_entt SYSTEM INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/entt"
    )
else()
    find_package(EnTT CONFIG REQUIRED)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/volk/volk.c")
    add_library(volk STATIC
        volk/volk.c
    )
    add_library(metallic::volk ALIAS volk)
    target_include_directories(volk PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/volk"
    )

    set(METALLIC_VOLK_VULKAN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SDL3/src/video/khronos")
    if(EXISTS "${METALLIC_VOLK_VULKAN_INCLUDE_DIR}/vulkan/vulkan.h")
        target_include_directories(volk SYSTEM PUBLIC
            "${METALLIC_VOLK_VULKAN_INCLUDE_DIR}"
        )
        target_compile_definitions(volk PUBLIC
            "VOLK_VULKAN_H_PATH=\"${METALLIC_VOLK_VULKAN_INCLUDE_DIR}/vulkan/vulkan.h\""
        )
    endif()
endif()

if(METALLIC_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/googletest/CMakeLists.txt")
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    if(MSVC)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    endif()
    add_subdirectory(googletest)
endif()
# SDL3: prefer a complete vendored checkout, but allow package managers to
# provide SDL3 while the repository is being restructured.
if(METALLIC_DEPENDENCY_MODE STREQUAL "PREBUILT")
    set(SDL3_DIR "${METALLIC_DEPENDENCY_ROOT}/cmake")
    find_package(SDL3 CONFIG REQUIRED GLOBAL PATHS "${METALLIC_DEPENDENCY_ROOT}" NO_DEFAULT_PATH)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/SDL3/CMakeLists.txt")
    metallic_sdl_options()
    add_subdirectory(SDL3)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/SDL/CMakeLists.txt")
    metallic_sdl_options()
    add_subdirectory(SDL)
else()
    find_package(SDL3 CONFIG REQUIRED)
endif()

set(METALLIC_SDL_TARGET "")
foreach(METALLIC_SDL_CANDIDATE IN ITEMS
    SDL3::SDL3
    SDL3::SDL3-static
    SDL3::SDL3-shared
    SDL3
    SDL3-static
    SDL3-shared
)
    if(NOT METALLIC_SDL_TARGET AND TARGET ${METALLIC_SDL_CANDIDATE})
        set(METALLIC_SDL_TARGET ${METALLIC_SDL_CANDIDATE})
    endif()
endforeach()

if(NOT METALLIC_SDL_TARGET)
    message(FATAL_ERROR "SDL3 was found, but no usable SDL3 CMake target is available.")
endif()

add_library(metallic_sdl3 INTERFACE)
add_library(metallic::sdl3 ALIAS metallic_sdl3)
target_link_libraries(metallic_sdl3 INTERFACE ${METALLIC_SDL_TARGET})

# Dear ImGui
add_library(imgui STATIC
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/imgui_demo.cpp
    imgui/backends/imgui_impl_sdl3.cpp
)
target_include_directories(imgui PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends
)
target_link_libraries(imgui PUBLIC metallic::sdl3)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ImGuizmo/src/ImGuizmo.cpp")
    add_library(imguizmo STATIC
        ImGuizmo/src/ImGuizmo.cpp
    )
    add_library(imguizmo::imguizmo ALIAS imguizmo)
    target_include_directories(imguizmo PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/ImGuizmo/src"
    )
    target_link_libraries(imguizmo PUBLIC imgui)
endif()

if(APPLE)
    target_sources(imgui PRIVATE imgui/backends/imgui_impl_metal.mm)
    target_link_libraries(imgui PUBLIC metal-cpp "-framework Metal" "-framework AppKit")
elseif(WIN32)
    include("${CMAKE_SOURCE_DIR}/cmake/PrepareImGuiVulkanBackend.cmake")
    set(METALLIC_IMGUI_VULKAN_BACKEND "${CMAKE_CURRENT_BINARY_DIR}/imgui/imgui_impl_vulkan.cpp")
    metallic_prepare_imgui_vulkan_backend(
        "${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends/imgui_impl_vulkan.cpp"
        "${METALLIC_IMGUI_VULKAN_BACKEND}")
    if(TARGET metallic::volk)
        target_sources(imgui PRIVATE "${METALLIC_IMGUI_VULKAN_BACKEND}")
        target_compile_definitions(imgui PUBLIC IMGUI_IMPL_VULKAN_USE_VOLK)
        target_link_libraries(imgui PUBLIC metallic::volk)
    else()
        find_package(Vulkan QUIET)
        if(Vulkan_FOUND)
            target_sources(imgui PRIVATE "${METALLIC_IMGUI_VULKAN_BACKEND}")
            target_link_libraries(imgui PUBLIC Vulkan::Vulkan)
        endif()
    endif()
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/MathLib/CMakeLists.txt")
    add_subdirectory(MathLib)
endif()

if(METALLIC_ENABLE_NRD AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ShaderMakeBlob/ShaderBlob.cpp")
    add_library(ShaderMakeBlob STATIC
        ShaderMakeBlob/ShaderBlob.cpp
    )
    target_include_directories(ShaderMakeBlob PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/ShaderMakeBlob"
        "${CMAKE_CURRENT_SOURCE_DIR}/ShaderMakeBlob/ShaderMake"
    )
    target_compile_features(ShaderMakeBlob PUBLIC cxx_std_17)

    add_executable(ShaderMake
        ShaderMakeBlob/ShaderMake.cpp
        ShaderMakeBlob/argparse.c
        ShaderMakeBlob/ShaderBlob.cpp
    )
    target_include_directories(ShaderMake PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/ShaderMakeBlob"
        "${CMAKE_CURRENT_SOURCE_DIR}/ShaderMakeBlob/ShaderMake"
    )
    target_compile_features(ShaderMake PRIVATE cxx_std_17)
    if(WIN32)
        target_compile_definitions(ShaderMake PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    endif()
endif()

set(METALLIC_NRD_ROOT "" CACHE PATH "Path to the NVIDIA NRD SDK checkout")
if(NOT METALLIC_NRD_ROOT)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/nrd/CMakeLists.txt")
        set(METALLIC_NRD_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/nrd")
    endif()
endif()

if(METALLIC_ENABLE_NRD AND METALLIC_NRD_ROOT AND EXISTS "${METALLIC_NRD_ROOT}/CMakeLists.txt")
    set(NRD_STATIC_LIBRARY ON CACHE BOOL "" FORCE)
    set(NRD_NRI OFF CACHE BOOL "" FORCE)
    set(NRD_EMBEDS_SPIRV_SHADERS ON CACHE BOOL "" FORCE)
    set(NRD_EMBEDS_DXIL_SHADERS OFF CACHE BOOL "" FORCE)
    set(NRD_EMBEDS_DXBC_SHADERS OFF CACHE BOOL "" FORCE)
    set(NRD_NORMAL_ENCODING "2" CACHE STRING "" FORCE)
    set(NRD_ROUGHNESS_ENCODING "1" CACHE STRING "" FORCE)
    # Generate embedded shader blobs into the build tree; the source-tree
    # default may be write-protected in some deployments.
    set(NRD_SHADERS_PATH "${CMAKE_BINARY_DIR}/External/nrd-shaders" CACHE STRING "" FORCE)
    if(TARGET ShaderMake)
        set(SHADERMAKE_PATH "$<TARGET_FILE:ShaderMake>" CACHE STRING "" FORCE)
    endif()

    # Metallic vendors ShaderMake without ShaderMake's compiler-discovery
    # CMake. NRD then invokes:
    #   ShaderMake -p SPIRV --compiler "${SHADERMAKE_DXC_VK_PATH}" --flatten ...
    # An empty compiler path makes ShaderMake treat --flatten as the compiler
    # ("Compiler '--flatten' does not exist!") and never emit *.cs.spirv.h.
    set(_metallic_nrd_dxc "")
    foreach(_metallic_nrd_dxc_cand IN ITEMS
        "${SHADERMAKE_DXC_VK_PATH}"
        "${DXC_SPIRV_PATH}")
        if(NOT _metallic_nrd_dxc AND _metallic_nrd_dxc_cand AND EXISTS "${_metallic_nrd_dxc_cand}")
            set(_metallic_nrd_dxc "${_metallic_nrd_dxc_cand}")
        endif()
    endforeach()
    if(NOT _metallic_nrd_dxc)
        set(_metallic_nrd_dxc_hints)
        if(DEFINED ENV{VULKAN_SDK} AND NOT "$ENV{VULKAN_SDK}" STREQUAL "")
            list(APPEND _metallic_nrd_dxc_hints
                "$ENV{VULKAN_SDK}/Bin"
                "$ENV{VULKAN_SDK}/bin")
        endif()
        if(DEFINED ENV{VK_SDK_PATH} AND NOT "$ENV{VK_SDK_PATH}" STREQUAL "")
            list(APPEND _metallic_nrd_dxc_hints
                "$ENV{VK_SDK_PATH}/Bin"
                "$ENV{VK_SDK_PATH}/bin")
        endif()
        find_program(_metallic_nrd_dxc_prog
            NAMES dxc dxc.exe
            HINTS ${_metallic_nrd_dxc_hints}
            NO_CACHE)
        if(_metallic_nrd_dxc_prog)
            set(_metallic_nrd_dxc "${_metallic_nrd_dxc_prog}")
        endif()
        unset(_metallic_nrd_dxc_prog)
        unset(_metallic_nrd_dxc_hints)
    endif()
    if(NOT _metallic_nrd_dxc)
        set(_metallic_nrd_dxc_version "v1.8.2407")
        set(_metallic_nrd_dxc_date "2024_07_31")
        set(_metallic_nrd_dxc_root "${CMAKE_BINARY_DIR}/_deps/metallic-nrd-dxc")
        if(WIN32)
            if((CMAKE_GENERATOR_PLATFORM MATCHES "ARM64") OR (CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64"))
                set(_metallic_nrd_dxc_arch "arm64")
            else()
                set(_metallic_nrd_dxc_arch "x64")
            endif()
            set(_metallic_nrd_dxc_cand "${_metallic_nrd_dxc_root}/bin/${_metallic_nrd_dxc_arch}/dxc.exe")
            set(_metallic_nrd_dxc_url
                "https://github.com/microsoft/DirectXShaderCompiler/releases/download/${_metallic_nrd_dxc_version}/dxc_${_metallic_nrd_dxc_date}.zip")
            set(_metallic_nrd_dxc_archive "${CMAKE_BINARY_DIR}/_deps/metallic-nrd-dxc.zip")
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            set(_metallic_nrd_dxc_cand "${_metallic_nrd_dxc_root}/bin/dxc")
            set(_metallic_nrd_dxc_url
                "https://github.com/microsoft/DirectXShaderCompiler/releases/download/${_metallic_nrd_dxc_version}/linux_dxc_${_metallic_nrd_dxc_date}.x86_64.tar.gz")
            set(_metallic_nrd_dxc_archive "${CMAKE_BINARY_DIR}/_deps/metallic-nrd-dxc.tar.gz")
        else()
            set(_metallic_nrd_dxc_cand "")
            set(_metallic_nrd_dxc_url "")
        endif()
        if(_metallic_nrd_dxc_url AND NOT EXISTS "${_metallic_nrd_dxc_cand}")
            message(STATUS "NRD: DXC not found locally; downloading ${_metallic_nrd_dxc_version}")
            file(DOWNLOAD
                "${_metallic_nrd_dxc_url}"
                "${_metallic_nrd_dxc_archive}"
                SHOW_PROGRESS
                TLS_VERIFY ON
                STATUS _metallic_nrd_dxc_status)
            list(GET _metallic_nrd_dxc_status 0 _metallic_nrd_dxc_status_code)
            if(_metallic_nrd_dxc_status_code EQUAL 0)
                file(MAKE_DIRECTORY "${_metallic_nrd_dxc_root}")
                file(ARCHIVE_EXTRACT
                    INPUT "${_metallic_nrd_dxc_archive}"
                    DESTINATION "${_metallic_nrd_dxc_root}")
            else()
                list(GET _metallic_nrd_dxc_status 1 _metallic_nrd_dxc_status_msg)
                message(WARNING "NRD: DXC download failed: ${_metallic_nrd_dxc_status_msg}")
            endif()
            unset(_metallic_nrd_dxc_status)
            unset(_metallic_nrd_dxc_status_code)
            unset(_metallic_nrd_dxc_status_msg)
        endif()
        if(_metallic_nrd_dxc_cand AND EXISTS "${_metallic_nrd_dxc_cand}")
            set(_metallic_nrd_dxc "${_metallic_nrd_dxc_cand}")
        endif()
        unset(_metallic_nrd_dxc_version)
        unset(_metallic_nrd_dxc_date)
        unset(_metallic_nrd_dxc_root)
        unset(_metallic_nrd_dxc_arch)
        unset(_metallic_nrd_dxc_cand)
        unset(_metallic_nrd_dxc_url)
        unset(_metallic_nrd_dxc_archive)
    endif()

    set(_metallic_nrd_prebuilt_shaders "${METALLIC_NRD_ROOT}/_Shaders")
    file(MAKE_DIRECTORY "${NRD_SHADERS_PATH}")
    if(EXISTS "${_metallic_nrd_prebuilt_shaders}/Clear.cs.spirv.h")
        # Seed the build-tree output so NRD headers exist even if ShaderMake
        # is skipped. Fresh DXC output overwrites these files when present.
        file(COPY "${_metallic_nrd_prebuilt_shaders}/"
            DESTINATION "${NRD_SHADERS_PATH}"
            FILES_MATCHING PATTERN "*.spirv.h")
    endif()

    if(_metallic_nrd_dxc)
        set(SHADERMAKE_DXC_VK_PATH "${_metallic_nrd_dxc}" CACHE FILEPATH
            "DXC used to compile NRD SPIR-V shaders" FORCE)
        set(DXC_SPIRV_PATH "${_metallic_nrd_dxc}" CACHE FILEPATH "" FORCE)
        message(STATUS "NRD SPIR-V DXC: ${SHADERMAKE_DXC_VK_PATH}")
    elseif(EXISTS "${NRD_SHADERS_PATH}/Clear.cs.spirv.h")
        message(WARNING
            "DXC was not found; NRD will use precompiled SPIR-V headers from "
            "${_metallic_nrd_prebuilt_shaders}. Install the Vulkan SDK or set "
            "SHADERMAKE_DXC_VK_PATH to rebuild NRD shaders.")
    else()
        message(FATAL_ERROR
            "NRD shader compilation requires DXC with SPIR-V support, but none was found "
            "and no precompiled headers exist in ${_metallic_nrd_prebuilt_shaders}.\n"
            "Install the Vulkan SDK (provides Bin/dxc.exe) or set SHADERMAKE_DXC_VK_PATH "
            "to a DXC executable. An empty compiler path produces "
            "\"Compiler '--flatten' does not exist!\" from ShaderMake.")
    endif()
    unset(_metallic_nrd_dxc)
    unset(_metallic_nrd_prebuilt_shaders)

    add_subdirectory("${METALLIC_NRD_ROOT}" "${CMAKE_BINARY_DIR}/External/nrd")
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tracy/CMakeLists.txt")
    option(TRACY_ENABLE "Enable Tracy profiler" ON)
    option(TRACY_ON_DEMAND "Tracy on-demand mode" ON)
    add_subdirectory(tracy)
endif()

if(METALLIC_DEPENDENCY_MODE STREQUAL "PREBUILT")
    set(spdlog_DIR "${METALLIC_DEPENDENCY_ROOT}/lib/cmake/spdlog")
    find_package(spdlog CONFIG REQUIRED GLOBAL PATHS "${METALLIC_DEPENDENCY_ROOT}" NO_DEFAULT_PATH)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/spdlog/CMakeLists.txt")
    add_subdirectory(spdlog)
endif()

add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/nlohmann)

add_library(tinygltf INTERFACE)
target_include_directories(tinygltf INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/tinygltf)
target_compile_definitions(tinygltf INTERFACE
    TINYGLTF_NO_INCLUDE_JSON
    TINYGLTF_NO_INCLUDE_STB_IMAGE
    TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
    TINYGLTF_NO_STB_IMAGE_WRITE
)

include("${CMAKE_SOURCE_DIR}/cmake/SetupOpenUsd.cmake")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/imnodes/CMakeLists.txt")
    add_library(imnodes STATIC
        imnodes/imnodes.cpp
    )
    target_include_directories(imnodes PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/imnodes)
    target_link_libraries(imnodes PUBLIC imgui)
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/microprofile/CMakeLists.txt")
    set(MICROPROFILE_USE_CONFIG_FILE OFF CACHE BOOL "Use microprofile.config.h" FORCE)
    add_subdirectory(microprofile)
    target_compile_definitions(microprofile PUBLIC MICROPROFILE_USE_CONFIG=0)
endif()
