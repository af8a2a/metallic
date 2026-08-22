option(METALLIC_ENABLE_NTC
    "Enable NVIDIA Neural Texture Compression inference-on-sample support"
    ON)
set(RTXNTC_ROOT "" CACHE PATH
    "RTXNTC checkout or RTXNTC-Library directory used by Metallic")

set(METALLIC_NTC_LIBRARY_ROOT "")
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
    # Metallic performs Generic INT8 inference directly in its Slang shaders.
    # CUDA and LibNTC's precompiled decompression shaders are unnecessary for
    # this path and would add large toolchain/runtime dependencies.
    set(NTC_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(NTC_WITH_CUDA OFF CACHE BOOL "" FORCE)
    set(NTC_WITH_DX12 OFF CACHE BOOL "" FORCE)
    set(NTC_WITH_VULKAN OFF CACHE BOOL "" FORCE)
    set(NTC_WITH_PREBUILT_SHADERS OFF CACHE BOOL "" FORCE)
    add_subdirectory(
        "${METALLIC_NTC_LIBRARY_ROOT}"
        "${CMAKE_BINARY_DIR}/External/RTXNTC-Library")
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
        "RTXNTC Library was not found; Neural Texture Compression support is disabled")
endif()
