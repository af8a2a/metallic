option(METALLIC_ENABLE_NSIGHT_NVTX
    "Enable NVTX profile markers for NVIDIA Nsight tools when NVTX3 headers are available"
    ON)

if(NOT TARGET metallic_nvtx3)
    add_library(metallic_nvtx3 INTERFACE)
    add_library(metallic::nvtx3 ALIAS metallic_nvtx3)
endif()

set(METALLIC_HAS_NVTX OFF)

if(NOT METALLIC_ENABLE_NSIGHT_NVTX)
    message(STATUS "Nsight NVTX markers disabled via METALLIC_ENABLE_NSIGHT_NVTX=OFF")
    return()
endif()

set(METALLIC_NVTX3_INCLUDE_DIR "" CACHE PATH
    "Optional path to an NVTX3 include directory containing nvtx3/nvToolsExt.h")

set(_metallic_nvtx_hints)

if(METALLIC_NVTX3_INCLUDE_DIR)
    list(APPEND _metallic_nvtx_hints "${METALLIC_NVTX3_INCLUDE_DIR}")
endif()

if(DEFINED ENV{METALLIC_NVTX3_INCLUDE_DIR})
    list(APPEND _metallic_nvtx_hints "$ENV{METALLIC_NVTX3_INCLUDE_DIR}")
endif()

if(DEFINED ENV{NVTX3_INCLUDE_DIR})
    list(APPEND _metallic_nvtx_hints "$ENV{NVTX3_INCLUDE_DIR}")
endif()

if(DEFINED ENV{NVTX3_ROOT})
    list(APPEND _metallic_nvtx_hints "$ENV{NVTX3_ROOT}")
endif()

if(DEFINED ENV{NVTX_ROOT})
    list(APPEND _metallic_nvtx_hints "$ENV{NVTX_ROOT}")
endif()

if(DEFINED ENV{NSIGHT_GRAPHICS_ROOT})
    list(APPEND _metallic_nvtx_hints "$ENV{NSIGHT_GRAPHICS_ROOT}")
endif()

if(WIN32)
    file(GLOB _metallic_nsight_graphics_nvtx_dirs LIST_DIRECTORIES true
        "C:/Program Files/NVIDIA Corporation/Nsight Graphics */target/Resources/NVTX3/include"
        "C:/Program Files/NVIDIA Corporation/NVIDIA Nsight Graphics */target/Resources/NVTX3/include"
    )
    list(APPEND _metallic_nvtx_hints ${_metallic_nsight_graphics_nvtx_dirs})
endif()

find_path(METALLIC_NVTX3_HEADER_DIR
    NAMES nvtx3/nvToolsExt.h
    HINTS ${_metallic_nvtx_hints}
    PATH_SUFFIXES "" include
)

if(METALLIC_NVTX3_HEADER_DIR)
    target_include_directories(metallic_nvtx3 INTERFACE "${METALLIC_NVTX3_HEADER_DIR}")
    target_compile_definitions(metallic_nvtx3 INTERFACE METALLIC_HAS_NVTX=1)
    set(METALLIC_HAS_NVTX ON)
    message(STATUS "Nsight NVTX markers enabled: ${METALLIC_NVTX3_HEADER_DIR}")
else()
    message(STATUS "NVTX3 headers not found. Nsight NVTX markers will be disabled.")
endif()

unset(_metallic_nvtx_hints)
unset(_metallic_nsight_graphics_nvtx_dirs)
