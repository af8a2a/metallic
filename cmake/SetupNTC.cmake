option(METALLIC_ENABLE_NTC
    "Build reserved NVIDIA Neural Texture Compression support for future VT work"
    OFF)
set(RTXNTC_ROOT "" CACHE PATH
    "RTXNTC checkout or RTXNTC-Library directory used by Metallic")

set(METALLIC_NTC_LIBRARY_ROOT "")
set(METALLIC_NTC_SHADER_INCLUDE_DIR "")
if(METALLIC_ENABLE_NTC)
    set(_METALLIC_NTC_CANDIDATES)
    if(RTXNTC_ROOT)
        list(APPEND _METALLIC_NTC_CANDIDATES
            "${RTXNTC_ROOT}"
            "${RTXNTC_ROOT}/libraries/RTXNTC-Library")
    endif()
    list(APPEND _METALLIC_NTC_CANDIDATES
        "${CMAKE_SOURCE_DIR}/External/RTXNTC-Library"
        "${CMAKE_SOURCE_DIR}/../RTXNTC/libraries/RTXNTC-Library")

    foreach(_METALLIC_NTC_CANDIDATE IN LISTS _METALLIC_NTC_CANDIDATES)
        if(EXISTS "${_METALLIC_NTC_CANDIDATE}/include/libntc/ntc.h" AND
           EXISTS "${_METALLIC_NTC_CANDIDATE}/CMakeLists.txt")
            get_filename_component(METALLIC_NTC_LIBRARY_ROOT
                "${_METALLIC_NTC_CANDIDATE}" ABSOLUTE)
            break()
        endif()
    endforeach()
    unset(_METALLIC_NTC_CANDIDATE)
    unset(_METALLIC_NTC_CANDIDATES)
endif()

if(METALLIC_NTC_LIBRARY_ROOT)
    # Reserved NTC experiments can perform inference in isolated Slang shaders. LibNTC's
    # Vulkan backend is used only to query and convert cooperative-vector
    # weights; CUDA, DX12, and precompiled decompression shaders remain unused.
    set(NTC_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(NTC_WITH_CUDA OFF CACHE BOOL "" FORCE)
    set(NTC_WITH_DX12 OFF CACHE BOOL "" FORCE)
    set(NTC_WITH_VULKAN ON CACHE BOOL "" FORCE)
    set(NTC_WITH_PREBUILT_SHADERS OFF CACHE BOOL "" FORCE)
    if(NOT TARGET Vulkan-Headers AND NOT TARGET Vulkan::Headers)
        find_path(METALLIC_NTC_VULKAN_HEADERS_DIR
            NAMES vulkan/vulkan.hpp
            HINTS "$ENV{VULKAN_SDK}"
            PATH_SUFFIXES Include include)
        if(NOT METALLIC_NTC_VULKAN_HEADERS_DIR)
            message(STATUS
                "RTXNTC disabled: Vulkan SDK headers (vulkan/vulkan.hpp) were not found")
            set(METALLIC_NTC_LIBRARY_ROOT "")
            return()
        endif()
        add_library(MetallicNTCVulkanHeaders INTERFACE)
        target_include_directories(MetallicNTCVulkanHeaders INTERFACE
            "${METALLIC_NTC_VULKAN_HEADERS_DIR}")
        add_library(Vulkan-Headers ALIAS MetallicNTCVulkanHeaders)
    endif()
    add_subdirectory(
        "${METALLIC_NTC_LIBRARY_ROOT}"
        "${CMAKE_BINARY_DIR}/External/RTXNTC-Library"
        EXCLUDE_FROM_ALL)
    if(MSVC AND TARGET libntc)
        # Metallic uses the DLL CRT. LibNTC defaults its static-library target
        # to /MT, which otherwise produces LNK2038 in Debug configurations.
        set_property(TARGET libntc PROPERTY MSVC_RUNTIME_LIBRARY
            "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    endif()
    set(METALLIC_NTC_SHADER_INCLUDE_DIR
        "${METALLIC_NTC_LIBRARY_ROOT}/include")
    message(STATUS "RTXNTC Library: ${METALLIC_NTC_LIBRARY_ROOT}")
else()
    set(METALLIC_NTC_SHADER_INCLUDE_DIR "")
    message(STATUS
        "Reserved NTC support is disabled or RTXNTC Library was not found")
endif()
