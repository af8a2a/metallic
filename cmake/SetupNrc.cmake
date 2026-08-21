option(METALLIC_ENABLE_NRC "Enable NVIDIA Neural Radiance Cache (NRC) integration when the SDK is available" ON)
set(METALLIC_NRC_ROOT "${CMAKE_SOURCE_DIR}/External/NRC" CACHE PATH "NVIDIA NRC SDK root containing Include, Lib and Bin directories")

add_library(metallic_nrc INTERFACE)
add_library(metallic::nrc ALIAS metallic_nrc)

function(metallic_nrc_is_usable sdk_root out_var)
    if(EXISTS "${sdk_root}/Include/NrcVk.h" AND
       EXISTS "${sdk_root}/Include/NrcCommon.h" AND
       EXISTS "${sdk_root}/Lib/NRC_Vulkan.lib" AND
       EXISTS "${sdk_root}/Bin/NRC_Vulkan.dll")
        set(${out_var} TRUE PARENT_SCOPE)
    else()
        set(${out_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(metallic_copy_nrc_runtime target_name)
    if(NOT METALLIC_HAS_NRC)
        return()
    endif()

    foreach(METALLIC_NRC_RUNTIME_FILE IN LISTS METALLIC_NRC_RUNTIME_FILES)
        if(EXISTS "${METALLIC_NRC_RUNTIME_FILE}")
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${METALLIC_NRC_RUNTIME_FILE}"
                    "$<TARGET_FILE_DIR:${target_name}>"
            )
        endif()
    endforeach()
endfunction()

set(METALLIC_HAS_NRC 0 CACHE INTERNAL "Whether Metallic found a usable NVIDIA NRC SDK")
set(METALLIC_NRC_RUNTIME_FILES "" CACHE INTERNAL "NVIDIA NRC runtime files to copy next to executables")

if(NOT METALLIC_ENABLE_NRC)
    message(STATUS "NVIDIA NRC disabled.")
    return()
endif()

if(NOT WIN32)
    message(STATUS "NVIDIA NRC skipped: this integration currently targets Windows/Vulkan.")
    return()
endif()

metallic_nrc_is_usable("${METALLIC_NRC_ROOT}" METALLIC_NRC_USABLE)
if(NOT METALLIC_NRC_USABLE)
    message(STATUS "NVIDIA NRC SDK binaries were not found at ${METALLIC_NRC_ROOT}; the neural radiance cache pass will compile as unsupported.")
    return()
endif()

file(TO_CMAKE_PATH "${METALLIC_NRC_ROOT}" METALLIC_NRC_ROOT_CMAKE)

set(METALLIC_HAS_NRC 1 CACHE INTERNAL "Whether Metallic found a usable NVIDIA NRC SDK" FORCE)
target_include_directories(metallic_nrc INTERFACE "${METALLIC_NRC_ROOT_CMAKE}/Include")
target_link_libraries(metallic_nrc INTERFACE "${METALLIC_NRC_ROOT_CMAKE}/Lib/NRC_Vulkan.lib")
target_compile_definitions(metallic_nrc INTERFACE METALLIC_HAS_NRC=1)

file(GLOB METALLIC_NRC_RUNTIME_DLLS
    "${METALLIC_NRC_ROOT_CMAKE}/Bin/*.dll"
)
set(METALLIC_NRC_RUNTIME_FILES
    ${METALLIC_NRC_RUNTIME_DLLS}
    CACHE INTERNAL "NVIDIA NRC runtime files to copy next to executables" FORCE
)
message(STATUS "NVIDIA NRC enabled: ${METALLIC_NRC_ROOT_CMAKE}")
