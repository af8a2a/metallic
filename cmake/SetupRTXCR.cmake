set(RTXCR_ROOT "" CACHE PATH
    "RTXCR checkout used by the RTXCR sample; individual submodules are preferred")
set(RTXCR_ASSETS_ROOT "" CACHE PATH
    "RTXCR-Assets checkout used by the licensed RTXCR reference sample")
set(METALLIC_RTXCR_MATERIAL_ROOT "")
set(METALLIC_RTXCR_GEOMETRY_ROOT "")
set(METALLIC_RTXCR_ASSETS_ROOT "")
set(METALLIC_RTXCR_SHADER_INCLUDE_DIR "")
set(METALLIC_RTXCR_GEOMETRY_SHADER_INCLUDE_DIR "")
if(NOT METALLIC_ENABLE_RTXCR)
    return()
endif()
set(_METALLIC_RTXCR_CANDIDATES)
if(RTXCR_ROOT)
    list(APPEND _METALLIC_RTXCR_CANDIDATES
        "${RTXCR_ROOT}"
        "${RTXCR_ROOT}/libraries/rtxcr/material")
endif()
list(APPEND _METALLIC_RTXCR_CANDIDATES
    "${CMAKE_SOURCE_DIR}/External/RTXCR-Material"
    "${CMAKE_SOURCE_DIR}/../RTXCR/libraries/rtxcr/material")

set(METALLIC_RTXCR_MATERIAL_ROOT "")
foreach(_METALLIC_RTXCR_CANDIDATE IN LISTS _METALLIC_RTXCR_CANDIDATES)
    if(EXISTS "${_METALLIC_RTXCR_CANDIDATE}/shaders/include/rtxcr/HairChiangBSDF.hlsli" AND
       EXISTS "${_METALLIC_RTXCR_CANDIDATE}/shaders/include/rtxcr/SubsurfaceScattering.hlsli")
        get_filename_component(METALLIC_RTXCR_MATERIAL_ROOT
            "${_METALLIC_RTXCR_CANDIDATE}" ABSOLUTE)
        break()
    endif()
endforeach()
unset(_METALLIC_RTXCR_CANDIDATE)
unset(_METALLIC_RTXCR_CANDIDATES)

if(METALLIC_RTXCR_MATERIAL_ROOT)
    set(METALLIC_RTXCR_SHADER_INCLUDE_DIR
        "${METALLIC_RTXCR_MATERIAL_ROOT}/shaders/include")
    message(STATUS
        "RTXCR Material Library: ${METALLIC_RTXCR_MATERIAL_ROOT}")
else()
    set(METALLIC_RTXCR_SHADER_INCLUDE_DIR "")
    message(STATUS
        "RTXCR Material Library was not found. MetallicRTXCRSample will build, "
        "but its shader cannot run until RTXCR_ROOT points to an RTXCR checkout "
        "or RTXCR-Material-Library is placed in External/RTXCR-Material.")
endif()

set(_METALLIC_RTXCR_GEOMETRY_CANDIDATES)
if(RTXCR_ROOT)
    list(APPEND _METALLIC_RTXCR_GEOMETRY_CANDIDATES
        "${RTXCR_ROOT}/libraries/rtxcr/geometry")
endif()
list(APPEND _METALLIC_RTXCR_GEOMETRY_CANDIDATES
    "${CMAKE_SOURCE_DIR}/External/RTXCR-Geometry"
    "${CMAKE_SOURCE_DIR}/../RTXCR/libraries/rtxcr/geometry")

set(METALLIC_RTXCR_GEOMETRY_ROOT "")
foreach(_METALLIC_RTXCR_GEOMETRY_CANDIDATE IN LISTS _METALLIC_RTXCR_GEOMETRY_CANDIDATES)
    if(EXISTS "${_METALLIC_RTXCR_GEOMETRY_CANDIDATE}/include/CurveTessellation.h" AND
       EXISTS "${_METALLIC_RTXCR_GEOMETRY_CANDIDATE}/shaders/include/rtxcr/geometry.hlsli")
        get_filename_component(METALLIC_RTXCR_GEOMETRY_ROOT
            "${_METALLIC_RTXCR_GEOMETRY_CANDIDATE}" ABSOLUTE)
        break()
    endif()
endforeach()
unset(_METALLIC_RTXCR_GEOMETRY_CANDIDATE)
unset(_METALLIC_RTXCR_GEOMETRY_CANDIDATES)

if(METALLIC_RTXCR_GEOMETRY_ROOT)
    set(METALLIC_RTXCR_GEOMETRY_SHADER_INCLUDE_DIR
        "${METALLIC_RTXCR_GEOMETRY_ROOT}/shaders/include")
    message(STATUS
        "RTXCR Geometry Library: ${METALLIC_RTXCR_GEOMETRY_ROOT}")
else()
    set(METALLIC_RTXCR_GEOMETRY_SHADER_INCLUDE_DIR "")
    message(STATUS
        "RTXCR Geometry Library was not found. RTXCR curve glTF assets cannot "
        "be converted to DOTS geometry until External/RTXCR-Geometry is initialized.")
endif()

set(_METALLIC_RTXCR_ASSET_CANDIDATES)
if(RTXCR_ASSETS_ROOT)
    list(APPEND _METALLIC_RTXCR_ASSET_CANDIDATES "${RTXCR_ASSETS_ROOT}")
endif()
if(RTXCR_ROOT)
    list(APPEND _METALLIC_RTXCR_ASSET_CANDIDATES "${RTXCR_ROOT}/assets")
endif()
list(APPEND _METALLIC_RTXCR_ASSET_CANDIDATES
    "${CMAKE_SOURCE_DIR}/External/RTXCR-Assets"
    "${CMAKE_SOURCE_DIR}/../RTXCR/assets")

set(METALLIC_RTXCR_ASSETS_ROOT "")
foreach(_METALLIC_RTXCR_ASSET_CANDIDATE IN LISTS _METALLIC_RTXCR_ASSET_CANDIDATES)
    set(_METALLIC_RTXCR_REFERENCE_GROOM
        "${_METALLIC_RTXCR_ASSET_CANDIDATE}/Claire/ponyTail_15vtx.gltf")
    set(_METALLIC_RTXCR_REFERENCE_GROOM_DATA
        "${_METALLIC_RTXCR_ASSET_CANDIDATE}/Claire/ponyTail_15vtx.bin")
    if(EXISTS "${_METALLIC_RTXCR_REFERENCE_GROOM}" AND
       EXISTS "${_METALLIC_RTXCR_REFERENCE_GROOM_DATA}" AND
       EXISTS "${_METALLIC_RTXCR_ASSET_CANDIDATE}/Claire/NVIDIA License for Claire Assets (2024.11.18) [FINAL].pdf")
        file(SIZE "${_METALLIC_RTXCR_REFERENCE_GROOM_DATA}" _METALLIC_RTXCR_GROOM_DATA_SIZE)
        if(_METALLIC_RTXCR_GROOM_DATA_SIZE GREATER 1000)
            get_filename_component(METALLIC_RTXCR_ASSETS_ROOT
                "${_METALLIC_RTXCR_ASSET_CANDIDATE}" ABSOLUTE)
            break()
        endif()
    endif()
endforeach()
unset(_METALLIC_RTXCR_REFERENCE_GROOM)
unset(_METALLIC_RTXCR_REFERENCE_GROOM_DATA)
unset(_METALLIC_RTXCR_GROOM_DATA_SIZE)
unset(_METALLIC_RTXCR_ASSET_CANDIDATE)
unset(_METALLIC_RTXCR_ASSET_CANDIDATES)

if(METALLIC_RTXCR_ASSETS_ROOT)
    message(STATUS "RTXCR licensed reference assets: ${METALLIC_RTXCR_ASSETS_ROOT}")
else()
    message(STATUS
        "RTXCR reference assets were not found, or their Git LFS payloads are "
        "missing. Run `git submodule update --init --recursive -- External/RTXCR-Assets` "
        "and `git -C External/RTXCR-Assets lfs pull` after accepting the included licenses.")
endif()
