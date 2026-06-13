option(METALLIC_ENABLE_NSIGHT_EVENTS "Enable NVIDIA Nsight/NVTX profile markers" ON)
set(METALLIC_NVTX_ROOT "" CACHE PATH "Optional NVTX root containing c/include/nvtx3/nvToolsExt.h or include/nvtx3/nvToolsExt.h")

add_library(metallic_nsight_events INTERFACE)
add_library(metallic::nsight_events ALIAS metallic_nsight_events)

if(METALLIC_ENABLE_NSIGHT_EVENTS)
    set(METALLIC_NVTX_CANDIDATES
        "${METALLIC_NVTX_ROOT}"
        "$ENV{METALLIC_NVTX_ROOT}"
        "${CMAKE_SOURCE_DIR}/External/NVTX"
        "${CMAKE_SOURCE_DIR}/External/nvtx"
        "${CMAKE_SOURCE_DIR}/External/nvtx3"
        "${CMAKE_SOURCE_DIR}/../nvpro_core2/third_party/NVTX"
    )

    set(METALLIC_NVTX_INCLUDE_DIR "")
    foreach(METALLIC_NVTX_CANDIDATE IN LISTS METALLIC_NVTX_CANDIDATES)
        if(NOT METALLIC_NVTX_CANDIDATE)
            continue()
        endif()

        if(EXISTS "${METALLIC_NVTX_CANDIDATE}/c/include/nvtx3/nvToolsExt.h")
            set(METALLIC_NVTX_INCLUDE_DIR "${METALLIC_NVTX_CANDIDATE}/c/include")
            break()
        elseif(EXISTS "${METALLIC_NVTX_CANDIDATE}/include/nvtx3/nvToolsExt.h")
            set(METALLIC_NVTX_INCLUDE_DIR "${METALLIC_NVTX_CANDIDATE}/include")
            break()
        endif()
    endforeach()

    if(METALLIC_NVTX_INCLUDE_DIR)
        target_include_directories(metallic_nsight_events INTERFACE "${METALLIC_NVTX_INCLUDE_DIR}")
        target_compile_definitions(metallic_nsight_events INTERFACE METALLIC_NSIGHT_EVENTS_AVAILABLE=1)
        message(STATUS "Nsight Events enabled with NVTX headers: ${METALLIC_NVTX_INCLUDE_DIR}")
    else()
        message(STATUS "Nsight Events enabled, but NVTX headers were not found. Markers compile as no-ops.")
    endif()
else()
    message(STATUS "Nsight Events disabled.")
endif()
