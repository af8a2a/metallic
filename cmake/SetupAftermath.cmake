option(METALLIC_ENABLE_NSIGHT_AFTERMATH "Enable NVIDIA Nsight Aftermath crash dump capture when the SDK is available" ON)
set(NsightAftermath_SDK "" CACHE PATH "Optional NVIDIA Nsight Aftermath SDK root")

add_library(metallic_aftermath INTERFACE)
add_library(metallic::aftermath ALIAS metallic_aftermath)

set(METALLIC_HAS_NSIGHT_AFTERMATH 0 CACHE INTERNAL "Whether Metallic found a usable NVIDIA Nsight Aftermath SDK")
set(METALLIC_NSIGHT_AFTERMATH_RUNTIME_FILES "" CACHE INTERNAL "NVIDIA Nsight Aftermath runtime files to copy next to executables")

function(metallic_set_aftermath_available value)
    set(METALLIC_HAS_NSIGHT_AFTERMATH ${value} CACHE INTERNAL "Whether Metallic found a usable NVIDIA Nsight Aftermath SDK")
    target_compile_definitions(metallic_aftermath INTERFACE METALLIC_HAS_NSIGHT_AFTERMATH=${value})
    add_compile_definitions(METALLIC_HAS_NSIGHT_AFTERMATH=${value})
endfunction()

function(metallic_copy_aftermath_runtime target_name)
    if(NOT METALLIC_HAS_NSIGHT_AFTERMATH)
        return()
    endif()

    foreach(METALLIC_NSIGHT_AFTERMATH_RUNTIME_FILE IN LISTS METALLIC_NSIGHT_AFTERMATH_RUNTIME_FILES)
        if(EXISTS "${METALLIC_NSIGHT_AFTERMATH_RUNTIME_FILE}")
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${METALLIC_NSIGHT_AFTERMATH_RUNTIME_FILE}"
                    "$<TARGET_FILE_DIR:${target_name}>"
            )
        endif()
    endforeach()
endfunction()

function(metallic_aftermath_is_usable sdk_root out_var)
    if(EXISTS "${sdk_root}/include/GFSDK_Aftermath.h" AND
       EXISTS "${sdk_root}/include/GFSDK_Aftermath_GpuCrashDump.h" AND
       EXISTS "${sdk_root}/include/GFSDK_Aftermath_GpuCrashDumpDecoding.h" AND
       EXISTS "${sdk_root}/lib/x64/GFSDK_Aftermath_Lib.x64.lib")
        set(${out_var} TRUE PARENT_SCOPE)
    else()
        set(${out_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

if(NOT METALLIC_ENABLE_NSIGHT_AFTERMATH)
    metallic_set_aftermath_available(0)
    message(STATUS "NVIDIA Nsight Aftermath disabled.")
    return()
endif()

if(NOT WIN32)
    metallic_set_aftermath_available(0)
    message(STATUS "NVIDIA Nsight Aftermath skipped: this integration currently targets Windows/Vulkan.")
    return()
endif()

set(METALLIC_NSIGHT_AFTERMATH_VENDORED_CANDIDATES
    "${CMAKE_SOURCE_DIR}/External/NsightAftermathSDK"
    "${CMAKE_SOURCE_DIR}/External/NsightAftermath"
    "${CMAKE_SOURCE_DIR}/External/nsight-aftermath"
    "${CMAKE_SOURCE_DIR}/External/aftermath"
)

set(METALLIC_NSIGHT_AFTERMATH_SDK_ROOT "")
foreach(METALLIC_NSIGHT_AFTERMATH_CANDIDATE IN LISTS METALLIC_NSIGHT_AFTERMATH_VENDORED_CANDIDATES)
    if(NOT METALLIC_NSIGHT_AFTERMATH_CANDIDATE)
        continue()
    endif()

    metallic_aftermath_is_usable("${METALLIC_NSIGHT_AFTERMATH_CANDIDATE}" METALLIC_NSIGHT_AFTERMATH_CANDIDATE_USABLE)
    if(METALLIC_NSIGHT_AFTERMATH_CANDIDATE_USABLE)
        set(METALLIC_NSIGHT_AFTERMATH_SDK_ROOT "${METALLIC_NSIGHT_AFTERMATH_CANDIDATE}")
        break()
    endif()
endforeach()

if(METALLIC_NSIGHT_AFTERMATH_SDK_ROOT STREQUAL "" AND NsightAftermath_SDK)
    metallic_aftermath_is_usable("${NsightAftermath_SDK}" METALLIC_NSIGHT_AFTERMATH_CANDIDATE_USABLE)
    if(METALLIC_NSIGHT_AFTERMATH_CANDIDATE_USABLE)
        set(METALLIC_NSIGHT_AFTERMATH_SDK_ROOT "${NsightAftermath_SDK}")
    endif()
endif()

if(METALLIC_NSIGHT_AFTERMATH_SDK_ROOT STREQUAL "")
    metallic_set_aftermath_available(0)
    return()
endif()

set(METALLIC_NSIGHT_AFTERMATH_INCLUDE_DIR "${METALLIC_NSIGHT_AFTERMATH_SDK_ROOT}/include")
set(METALLIC_NSIGHT_AFTERMATH_LIBRARY "${METALLIC_NSIGHT_AFTERMATH_SDK_ROOT}/lib/x64/GFSDK_Aftermath_Lib.x64.lib")
set(METALLIC_NSIGHT_AFTERMATH_DLL "${METALLIC_NSIGHT_AFTERMATH_SDK_ROOT}/lib/x64/GFSDK_Aftermath_Lib.x64.dll")

metallic_set_aftermath_available(1)
target_include_directories(metallic_aftermath SYSTEM INTERFACE "${METALLIC_NSIGHT_AFTERMATH_INCLUDE_DIR}")
target_link_libraries(metallic_aftermath INTERFACE "${METALLIC_NSIGHT_AFTERMATH_LIBRARY}")

if(EXISTS "${METALLIC_NSIGHT_AFTERMATH_DLL}")
    set(METALLIC_NSIGHT_AFTERMATH_RUNTIME_FILES
        "${METALLIC_NSIGHT_AFTERMATH_DLL}"
        CACHE INTERNAL "NVIDIA Nsight Aftermath runtime files to copy next to executables"
    )
endif()

message(STATUS "NVIDIA Nsight Aftermath enabled: ${METALLIC_NSIGHT_AFTERMATH_SDK_ROOT}")
