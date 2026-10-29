option(METALLIC_ENABLE_NVPERF "Enable optional Vulkan NvPerf range profiling" OFF)
set(METALLIC_NVPERF_SDK_ROOT "" CACHE PATH "Nsight Perf SDK root containing NvPerf/include and redist/NvPerfUtility")
add_library(metallic_nvperf INTERFACE)
add_library(metallic::nvperf ALIAS metallic_nvperf)
set(METALLIC_HAS_NVPERF 0)
if(METALLIC_ENABLE_NVPERF)
    if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
        message(STATUS "NvPerf disabled: the backend requires Windows x64")
    else()
        set(METALLIC_HAS_NVPERF 1)
        foreach(part IN ITEMS NvPerf/include/nvperf_vulkan_target.h
                NvPerf/include/windows-desktop-x64/nvperf_host_impl.h
                redist/NvPerfUtility/include/NvPerfVulkan.h NvPerf/bin/x64/nvperf_grfx_host.dll)
            if(NOT EXISTS "${METALLIC_NVPERF_SDK_ROOT}/${part}")
                set(METALLIC_HAS_NVPERF 0)
                message(STATUS "NvPerf disabled: SDK file not found: ${METALLIC_NVPERF_SDK_ROOT}/${part}")
                break()
            endif()
        endforeach()
    endif()
endif()
if(METALLIC_HAS_NVPERF)
    target_include_directories(metallic_nvperf SYSTEM INTERFACE
        "${METALLIC_NVPERF_SDK_ROOT}/NvPerf/include"
        "${METALLIC_NVPERF_SDK_ROOT}/NvPerf/include/windows-desktop-x64"
        "${METALLIC_NVPERF_SDK_ROOT}/redist/NvPerfUtility/include")
    # The SDK loader loads only on request; ordinary startup needs no SDK DLL.
    target_compile_definitions(metallic_nvperf INTERFACE
        METALLIC_NVPERF_LIBRARY_DIR="${METALLIC_NVPERF_SDK_ROOT}/NvPerf/bin/x64")
endif()
target_compile_definitions(metallic_nvperf INTERFACE METALLIC_HAS_NVPERF=${METALLIC_HAS_NVPERF})
