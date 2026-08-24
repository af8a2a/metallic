option(
    METALLIC_ENABLE_NSIGHT_GRAPHICS_CAPTURE
    "Enable programmatic NVIDIA Nsight Graphics captures when the SDK is available"
    ON
)
set(
    METALLIC_NSIGHT_GRAPHICS_SDK_ROOT
    ""
    CACHE PATH
    "Optional Nsight Graphics SDK root containing include/NGFX_GraphicsCapture_Vulkan.h"
)
set(
    METALLIC_NSIGHT_GRAPHICS_INSTALLATION_ROOT
    ""
    CACHE PATH
    "Optional Nsight Graphics installation root containing target/windows-desktop-nomad-x64"
)

add_library(metallic_nsight_graphics_capture INTERFACE)
add_library(metallic::nsight_graphics_capture ALIAS metallic_nsight_graphics_capture)

set(
    METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
    0
    CACHE INTERNAL
    "Whether Metallic found a usable NVIDIA Nsight Graphics SDK and runtime"
)

function(metallic_set_nsight_graphics_capture_available value)
    set(
        METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
        ${value}
        CACHE INTERNAL
        "Whether Metallic found a usable NVIDIA Nsight Graphics SDK and runtime"
        FORCE
    )
    target_compile_definitions(
        metallic_nsight_graphics_capture
        INTERFACE METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE=${value}
    )
endfunction()

function(metallic_nsight_graphics_sdk_is_usable sdk_root out_var)
    if(EXISTS "${sdk_root}/include/NGFX_GraphicsCapture_Vulkan.h" AND
       EXISTS "${sdk_root}/include/NGFX_GraphicsCapture_Common.h" AND
       EXISTS "${sdk_root}/include/NGFX_Types.h")
        file(
            STRINGS
            "${sdk_root}/include/NGFX_GraphicsCapture_Common.h"
            METALLIC_NSIGHT_GRAPHICS_ARTIFACT_API
            REGEX "NGFX_GraphicsCapture_WaitForCaptureFilePath"
            LIMIT_COUNT 1
        )
        if(METALLIC_NSIGHT_GRAPHICS_ARTIFACT_API)
            set(${out_var} TRUE PARENT_SCOPE)
            return()
        endif()
    endif()

    set(${out_var} FALSE PARENT_SCOPE)
endfunction()

function(metallic_nsight_graphics_runtime_is_usable installation_root out_var)
    set(
        METALLIC_NSIGHT_GRAPHICS_TARGET_DIR
        "${installation_root}/target/windows-desktop-nomad-x64"
    )
    if(EXISTS "${METALLIC_NSIGHT_GRAPHICS_TARGET_DIR}/ngfx-api-bootstrap.dll" AND
       EXISTS "${METALLIC_NSIGHT_GRAPHICS_TARGET_DIR}/ngfx-capture-injection.dll" AND
       EXISTS "${METALLIC_NSIGHT_GRAPHICS_TARGET_DIR}/ngfx-capture-interception.dll")
        set(${out_var} TRUE PARENT_SCOPE)
    else()
        set(${out_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(metallic_nsight_graphics_find_sdk installation_root out_var)
    file(
        GLOB METALLIC_NSIGHT_GRAPHICS_SDK_CANDIDATES
        LIST_DIRECTORIES TRUE
        "${installation_root}/SDKs/NsightGraphicsSDK/*"
    )
    list(SORT METALLIC_NSIGHT_GRAPHICS_SDK_CANDIDATES COMPARE NATURAL ORDER DESCENDING)

    foreach(METALLIC_NSIGHT_GRAPHICS_SDK_CANDIDATE IN LISTS METALLIC_NSIGHT_GRAPHICS_SDK_CANDIDATES)
        metallic_nsight_graphics_sdk_is_usable(
            "${METALLIC_NSIGHT_GRAPHICS_SDK_CANDIDATE}"
            METALLIC_NSIGHT_GRAPHICS_SDK_CANDIDATE_USABLE
        )
        if(METALLIC_NSIGHT_GRAPHICS_SDK_CANDIDATE_USABLE)
            set(${out_var} "${METALLIC_NSIGHT_GRAPHICS_SDK_CANDIDATE}" PARENT_SCOPE)
            return()
        endif()
    endforeach()

    set(${out_var} "" PARENT_SCOPE)
endfunction()

function(metallic_nsight_graphics_derive_installation_root sdk_root out_var)
    get_filename_component(METALLIC_NSIGHT_GRAPHICS_SDK_VERSION_DIR "${sdk_root}" ABSOLUTE)
    get_filename_component(METALLIC_NSIGHT_GRAPHICS_SDK_FAMILY_DIR "${METALLIC_NSIGHT_GRAPHICS_SDK_VERSION_DIR}" DIRECTORY)
    get_filename_component(METALLIC_NSIGHT_GRAPHICS_SDKS_DIR "${METALLIC_NSIGHT_GRAPHICS_SDK_FAMILY_DIR}" DIRECTORY)
    get_filename_component(METALLIC_NSIGHT_GRAPHICS_DERIVED_INSTALLATION_ROOT "${METALLIC_NSIGHT_GRAPHICS_SDKS_DIR}" DIRECTORY)
    set(${out_var} "${METALLIC_NSIGHT_GRAPHICS_DERIVED_INSTALLATION_ROOT}" PARENT_SCOPE)
endfunction()

if(NOT METALLIC_ENABLE_NSIGHT_GRAPHICS_CAPTURE)
    metallic_set_nsight_graphics_capture_available(0)
    message(STATUS "NVIDIA Nsight Graphics Capture disabled.")
    return()
endif()

if(NOT WIN32)
    metallic_set_nsight_graphics_capture_available(0)
    message(STATUS "NVIDIA Nsight Graphics Capture skipped: self-injection currently targets Windows/Vulkan.")
    return()
endif()

set(METALLIC_NSIGHT_GRAPHICS_RESOLVED_SDK_ROOT "")
set(METALLIC_NSIGHT_GRAPHICS_RESOLVED_INSTALLATION_ROOT "")

metallic_nsight_graphics_sdk_is_usable(
    "${METALLIC_NSIGHT_GRAPHICS_SDK_ROOT}"
    METALLIC_NSIGHT_GRAPHICS_CONFIGURED_SDK_USABLE
)
if(METALLIC_NSIGHT_GRAPHICS_CONFIGURED_SDK_USABLE)
    get_filename_component(
        METALLIC_NSIGHT_GRAPHICS_RESOLVED_SDK_ROOT
        "${METALLIC_NSIGHT_GRAPHICS_SDK_ROOT}"
        ABSOLUTE
    )
endif()

metallic_nsight_graphics_runtime_is_usable(
    "${METALLIC_NSIGHT_GRAPHICS_INSTALLATION_ROOT}"
    METALLIC_NSIGHT_GRAPHICS_CONFIGURED_RUNTIME_USABLE
)
if(METALLIC_NSIGHT_GRAPHICS_CONFIGURED_RUNTIME_USABLE)
    get_filename_component(
        METALLIC_NSIGHT_GRAPHICS_RESOLVED_INSTALLATION_ROOT
        "${METALLIC_NSIGHT_GRAPHICS_INSTALLATION_ROOT}"
        ABSOLUTE
    )
endif()

if(METALLIC_NSIGHT_GRAPHICS_RESOLVED_SDK_ROOT AND
   NOT METALLIC_NSIGHT_GRAPHICS_RESOLVED_INSTALLATION_ROOT)
    metallic_nsight_graphics_derive_installation_root(
        "${METALLIC_NSIGHT_GRAPHICS_RESOLVED_SDK_ROOT}"
        METALLIC_NSIGHT_GRAPHICS_DERIVED_INSTALLATION_ROOT
    )
    metallic_nsight_graphics_runtime_is_usable(
        "${METALLIC_NSIGHT_GRAPHICS_DERIVED_INSTALLATION_ROOT}"
        METALLIC_NSIGHT_GRAPHICS_DERIVED_RUNTIME_USABLE
    )
    if(METALLIC_NSIGHT_GRAPHICS_DERIVED_RUNTIME_USABLE)
        set(
            METALLIC_NSIGHT_GRAPHICS_RESOLVED_INSTALLATION_ROOT
            "${METALLIC_NSIGHT_GRAPHICS_DERIVED_INSTALLATION_ROOT}"
        )
    endif()
endif()

if(METALLIC_NSIGHT_GRAPHICS_RESOLVED_INSTALLATION_ROOT AND
   NOT METALLIC_NSIGHT_GRAPHICS_RESOLVED_SDK_ROOT)
    metallic_nsight_graphics_find_sdk(
        "${METALLIC_NSIGHT_GRAPHICS_RESOLVED_INSTALLATION_ROOT}"
        METALLIC_NSIGHT_GRAPHICS_RESOLVED_SDK_ROOT
    )
endif()

if(NOT METALLIC_NSIGHT_GRAPHICS_RESOLVED_SDK_ROOT OR
   NOT METALLIC_NSIGHT_GRAPHICS_RESOLVED_INSTALLATION_ROOT)
    set(METALLIC_NSIGHT_GRAPHICS_INSTALLATION_CANDIDATES)
    if(DEFINED ENV{ProgramFiles} AND NOT "$ENV{ProgramFiles}" STREQUAL "")
        file(
            GLOB METALLIC_NSIGHT_GRAPHICS_PROGRAM_FILES_CANDIDATES
            LIST_DIRECTORIES TRUE
            "$ENV{ProgramFiles}/NVIDIA Corporation/Nsight Graphics *"
        )
        list(APPEND
            METALLIC_NSIGHT_GRAPHICS_INSTALLATION_CANDIDATES
            ${METALLIC_NSIGHT_GRAPHICS_PROGRAM_FILES_CANDIDATES}
        )
    endif()
    file(
        GLOB METALLIC_NSIGHT_GRAPHICS_DEFAULT_CANDIDATES
        LIST_DIRECTORIES TRUE
        "C:/Program Files/NVIDIA Corporation/Nsight Graphics *"
    )
    list(APPEND
        METALLIC_NSIGHT_GRAPHICS_INSTALLATION_CANDIDATES
        ${METALLIC_NSIGHT_GRAPHICS_DEFAULT_CANDIDATES}
    )
    list(REMOVE_DUPLICATES METALLIC_NSIGHT_GRAPHICS_INSTALLATION_CANDIDATES)
    list(SORT METALLIC_NSIGHT_GRAPHICS_INSTALLATION_CANDIDATES COMPARE NATURAL ORDER DESCENDING)

    foreach(METALLIC_NSIGHT_GRAPHICS_INSTALLATION_CANDIDATE IN LISTS METALLIC_NSIGHT_GRAPHICS_INSTALLATION_CANDIDATES)
        metallic_nsight_graphics_runtime_is_usable(
            "${METALLIC_NSIGHT_GRAPHICS_INSTALLATION_CANDIDATE}"
            METALLIC_NSIGHT_GRAPHICS_INSTALLATION_CANDIDATE_USABLE
        )
        if(NOT METALLIC_NSIGHT_GRAPHICS_INSTALLATION_CANDIDATE_USABLE)
            continue()
        endif()

        metallic_nsight_graphics_find_sdk(
            "${METALLIC_NSIGHT_GRAPHICS_INSTALLATION_CANDIDATE}"
            METALLIC_NSIGHT_GRAPHICS_SDK_CANDIDATE
        )
        if(METALLIC_NSIGHT_GRAPHICS_SDK_CANDIDATE)
            get_filename_component(
                METALLIC_NSIGHT_GRAPHICS_RESOLVED_INSTALLATION_ROOT
                "${METALLIC_NSIGHT_GRAPHICS_INSTALLATION_CANDIDATE}"
                ABSOLUTE
            )
            set(
                METALLIC_NSIGHT_GRAPHICS_RESOLVED_SDK_ROOT
                "${METALLIC_NSIGHT_GRAPHICS_SDK_CANDIDATE}"
            )
            break()
        endif()
    endforeach()
endif()

if(NOT METALLIC_NSIGHT_GRAPHICS_RESOLVED_SDK_ROOT OR
   NOT METALLIC_NSIGHT_GRAPHICS_RESOLVED_INSTALLATION_ROOT)
    metallic_set_nsight_graphics_capture_available(0)
    message(STATUS "NVIDIA Nsight Graphics Capture SDK/runtime not found; capture APIs compile as no-ops.")
    return()
endif()

set(
    METALLIC_NSIGHT_GRAPHICS_SDK_ROOT
    "${METALLIC_NSIGHT_GRAPHICS_RESOLVED_SDK_ROOT}"
    CACHE PATH
    "Optional Nsight Graphics SDK root containing include/NGFX_GraphicsCapture_Vulkan.h"
    FORCE
)
set(
    METALLIC_NSIGHT_GRAPHICS_INSTALLATION_ROOT
    "${METALLIC_NSIGHT_GRAPHICS_RESOLVED_INSTALLATION_ROOT}"
    CACHE PATH
    "Optional Nsight Graphics installation root containing target/windows-desktop-nomad-x64"
    FORCE
)

file(
    TO_CMAKE_PATH
    "${METALLIC_NSIGHT_GRAPHICS_RESOLVED_INSTALLATION_ROOT}"
    METALLIC_NSIGHT_GRAPHICS_INSTALLATION_ROOT_CMAKE
)
metallic_set_nsight_graphics_capture_available(1)
target_include_directories(
    metallic_nsight_graphics_capture
    SYSTEM INTERFACE "${METALLIC_NSIGHT_GRAPHICS_RESOLVED_SDK_ROOT}/include"
)
target_compile_definitions(
    metallic_nsight_graphics_capture
    INTERFACE
        "METALLIC_NSIGHT_GRAPHICS_DEFAULT_INSTALLATION_ROOT=\"${METALLIC_NSIGHT_GRAPHICS_INSTALLATION_ROOT_CMAKE}\""
)

message(
    STATUS
    "NVIDIA Nsight Graphics Capture enabled: SDK ${METALLIC_NSIGHT_GRAPHICS_RESOLVED_SDK_ROOT}, runtime ${METALLIC_NSIGHT_GRAPHICS_RESOLVED_INSTALLATION_ROOT}"
)
