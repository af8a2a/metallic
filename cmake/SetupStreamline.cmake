option(METALLIC_ENABLE_STREAMLINE "Enable NVIDIA Streamline integration when the SDK is available" ON)
option(METALLIC_STREAMLINE_AUTO_DOWNLOAD "Download NVIDIA Streamline release SDK binaries when they are missing" ON)
set(METALLIC_STREAMLINE_ROOT "${CMAKE_SOURCE_DIR}/External/streamline" CACHE PATH "NVIDIA Streamline source checkout or packaged SDK root")
set(METALLIC_STREAMLINE_RELEASE_TAG "" CACHE STRING "NVIDIA Streamline release tag to download; defaults to the tag checked out in METALLIC_STREAMLINE_ROOT")
set(METALLIC_STREAMLINE_RELEASE_BASE_URL "https://github.com/NVIDIA-RTX/Streamline/releases/download" CACHE STRING "Base URL for NVIDIA Streamline release downloads")

add_library(metallic_streamline INTERFACE)
add_library(metallic::streamline ALIAS metallic_streamline)

set(METALLIC_HAS_STREAMLINE 0 CACHE INTERNAL "Whether Metallic found a usable NVIDIA Streamline SDK")
set(METALLIC_STREAMLINE_RUNTIME_FILES "" CACHE INTERNAL "NVIDIA Streamline runtime files to copy next to executables")

function(metallic_set_streamline_available value)
    set(METALLIC_HAS_STREAMLINE ${value} CACHE INTERNAL "Whether Metallic found a usable NVIDIA Streamline SDK")
    target_compile_definitions(metallic_streamline INTERFACE METALLIC_HAS_STREAMLINE=${value})
    add_compile_definitions(METALLIC_HAS_STREAMLINE=${value})
endfunction()

function(metallic_copy_streamline_runtime target_name)
    if(NOT METALLIC_HAS_STREAMLINE)
        return()
    endif()

    foreach(METALLIC_STREAMLINE_RUNTIME_FILE IN LISTS METALLIC_STREAMLINE_RUNTIME_FILES)
        if(EXISTS "${METALLIC_STREAMLINE_RUNTIME_FILE}")
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${METALLIC_STREAMLINE_RUNTIME_FILE}"
                    "$<TARGET_FILE_DIR:${target_name}>"
            )
        endif()
    endforeach()
endfunction()

function(metallic_streamline_is_usable sdk_root out_var)
    if(EXISTS "${sdk_root}/include/sl.h" AND
       EXISTS "${sdk_root}/include/sl_dlss_d.h" AND
       EXISTS "${sdk_root}/lib/x64/sl.interposer.lib" AND
       EXISTS "${sdk_root}/bin/x64/sl.interposer.dll" AND
       EXISTS "${sdk_root}/bin/x64/sl.common.dll" AND
       EXISTS "${sdk_root}/bin/x64/sl.dlss_d.dll" AND
       EXISTS "${sdk_root}/bin/x64/nvngx_dlssd.dll")
        set(${out_var} TRUE PARENT_SCOPE)
    else()
        set(${out_var} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(metallic_streamline_find_existing_sdk root out_var)
    set(METALLIC_STREAMLINE_CANDIDATES
        "${root}"
        "${root}/_sdk"
    )

    foreach(METALLIC_STREAMLINE_CANDIDATE IN LISTS METALLIC_STREAMLINE_CANDIDATES)
        metallic_streamline_is_usable("${METALLIC_STREAMLINE_CANDIDATE}" METALLIC_STREAMLINE_CANDIDATE_USABLE)
        if(METALLIC_STREAMLINE_CANDIDATE_USABLE)
            set(${out_var} "${METALLIC_STREAMLINE_CANDIDATE}" PARENT_SCOPE)
            return()
        endif()
    endforeach()

    set(${out_var} "" PARENT_SCOPE)
endfunction()

function(metallic_streamline_release_tag root out_var)
    if(NOT METALLIC_STREAMLINE_RELEASE_TAG STREQUAL "")
        set(${out_var} "${METALLIC_STREAMLINE_RELEASE_TAG}" PARENT_SCOPE)
        return()
    endif()

    find_package(Git QUIET)
    if(GIT_FOUND AND EXISTS "${root}/.git")
        execute_process(
            COMMAND "${GIT_EXECUTABLE}" -C "${root}" describe --tags --exact-match
            OUTPUT_VARIABLE METALLIC_STREAMLINE_GIT_TAG
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(NOT METALLIC_STREAMLINE_GIT_TAG STREQUAL "")
            set(${out_var} "${METALLIC_STREAMLINE_GIT_TAG}" PARENT_SCOPE)
            return()
        endif()
    endif()

    set(${out_var} "" PARENT_SCOPE)
endfunction()

function(metallic_streamline_find_extracted_sdk extract_root out_var)
    file(GLOB_RECURSE METALLIC_STREAMLINE_EXTRACTED_LIBS
        LIST_DIRECTORIES FALSE
        "${extract_root}/sl.interposer.lib"
    )

    foreach(METALLIC_STREAMLINE_EXTRACTED_LIB IN LISTS METALLIC_STREAMLINE_EXTRACTED_LIBS)
        get_filename_component(METALLIC_STREAMLINE_X64_DIR "${METALLIC_STREAMLINE_EXTRACTED_LIB}" DIRECTORY)
        get_filename_component(METALLIC_STREAMLINE_LIB_DIR "${METALLIC_STREAMLINE_X64_DIR}" DIRECTORY)
        get_filename_component(METALLIC_STREAMLINE_CANDIDATE_ROOT "${METALLIC_STREAMLINE_LIB_DIR}" DIRECTORY)
        metallic_streamline_is_usable("${METALLIC_STREAMLINE_CANDIDATE_ROOT}" METALLIC_STREAMLINE_CANDIDATE_USABLE)
        if(METALLIC_STREAMLINE_CANDIDATE_USABLE)
            set(${out_var} "${METALLIC_STREAMLINE_CANDIDATE_ROOT}" PARENT_SCOPE)
            return()
        endif()
    endforeach()

    set(${out_var} "" PARENT_SCOPE)
endfunction()

function(metallic_streamline_download_sdk root out_var)
    set(${out_var} "" PARENT_SCOPE)

    metallic_streamline_release_tag("${root}" METALLIC_STREAMLINE_TAG)
    if(METALLIC_STREAMLINE_TAG STREQUAL "")
        message(STATUS "NVIDIA Streamline auto-download skipped: ${root} is not checked out at a release tag. Set METALLIC_STREAMLINE_RELEASE_TAG to override.")
        return()
    endif()

    set(METALLIC_STREAMLINE_DOWNLOAD_DIR "${root}/_artifacts/metallic_downloads")
    set(METALLIC_STREAMLINE_EXTRACT_DIR "${METALLIC_STREAMLINE_DOWNLOAD_DIR}/extract")
    set(METALLIC_STREAMLINE_ARCHIVE "${METALLIC_STREAMLINE_DOWNLOAD_DIR}/streamline-sdk-${METALLIC_STREAMLINE_TAG}.zip")
    set(METALLIC_STREAMLINE_URL "${METALLIC_STREAMLINE_RELEASE_BASE_URL}/${METALLIC_STREAMLINE_TAG}/streamline-sdk-${METALLIC_STREAMLINE_TAG}.zip")

    message(STATUS "Downloading NVIDIA Streamline SDK binaries: ${METALLIC_STREAMLINE_URL}")
    file(MAKE_DIRECTORY "${METALLIC_STREAMLINE_DOWNLOAD_DIR}")
    file(DOWNLOAD
        "${METALLIC_STREAMLINE_URL}"
        "${METALLIC_STREAMLINE_ARCHIVE}"
        SHOW_PROGRESS
        TLS_VERIFY ON
        STATUS METALLIC_STREAMLINE_DOWNLOAD_STATUS
    )
    list(GET METALLIC_STREAMLINE_DOWNLOAD_STATUS 0 METALLIC_STREAMLINE_DOWNLOAD_CODE)
    list(GET METALLIC_STREAMLINE_DOWNLOAD_STATUS 1 METALLIC_STREAMLINE_DOWNLOAD_MESSAGE)
    if(NOT METALLIC_STREAMLINE_DOWNLOAD_CODE EQUAL 0)
        message(STATUS "NVIDIA Streamline auto-download failed: ${METALLIC_STREAMLINE_DOWNLOAD_MESSAGE}")
        return()
    endif()

    file(REMOVE_RECURSE "${METALLIC_STREAMLINE_EXTRACT_DIR}")
    file(MAKE_DIRECTORY "${METALLIC_STREAMLINE_EXTRACT_DIR}")
    file(ARCHIVE_EXTRACT
        INPUT "${METALLIC_STREAMLINE_ARCHIVE}"
        DESTINATION "${METALLIC_STREAMLINE_EXTRACT_DIR}"
    )

    metallic_streamline_find_extracted_sdk("${METALLIC_STREAMLINE_EXTRACT_DIR}" METALLIC_STREAMLINE_EXTRACTED_SDK_ROOT)
    if(METALLIC_STREAMLINE_EXTRACTED_SDK_ROOT STREQUAL "")
        message(STATUS "NVIDIA Streamline auto-download failed: extracted archive did not contain a usable SDK layout.")
        return()
    endif()

    file(MAKE_DIRECTORY "${root}/_sdk")
    file(COPY "${METALLIC_STREAMLINE_EXTRACTED_SDK_ROOT}/" DESTINATION "${root}/_sdk")
    set(${out_var} "${root}/_sdk" PARENT_SCOPE)
endfunction()

if(NOT METALLIC_ENABLE_STREAMLINE)
    metallic_set_streamline_available(0)
    message(STATUS "NVIDIA Streamline disabled.")
    return()
endif()

if(NOT WIN32)
    metallic_set_streamline_available(0)
    message(STATUS "NVIDIA Streamline skipped: this integration currently targets Windows/Vulkan.")
    return()
endif()

metallic_streamline_find_existing_sdk("${METALLIC_STREAMLINE_ROOT}" METALLIC_STREAMLINE_SDK_ROOT)
if(METALLIC_STREAMLINE_SDK_ROOT STREQUAL "" AND METALLIC_STREAMLINE_AUTO_DOWNLOAD)
    metallic_streamline_download_sdk("${METALLIC_STREAMLINE_ROOT}" METALLIC_STREAMLINE_SDK_ROOT)
endif()

set(METALLIC_STREAMLINE_INCLUDE_DIR "${METALLIC_STREAMLINE_SDK_ROOT}/include")
set(METALLIC_STREAMLINE_LIBRARY "${METALLIC_STREAMLINE_SDK_ROOT}/lib/x64/sl.interposer.lib")
set(METALLIC_STREAMLINE_BIN_DIR "${METALLIC_STREAMLINE_SDK_ROOT}/bin/x64")
set(METALLIC_STREAMLINE_SCRIPTS_DIR "${METALLIC_STREAMLINE_SDK_ROOT}/scripts")

metallic_streamline_is_usable("${METALLIC_STREAMLINE_SDK_ROOT}" METALLIC_STREAMLINE_USABLE)
if(METALLIC_STREAMLINE_USABLE)
    file(TO_CMAKE_PATH "${METALLIC_STREAMLINE_BIN_DIR}" METALLIC_STREAMLINE_BIN_DIR_CMAKE)

    metallic_set_streamline_available(1)
    add_compile_definitions(
        STREAMLINE_FEATURE_DLSS_RR=1
        "METALLIC_STREAMLINE_BIN_DIR=\"${METALLIC_STREAMLINE_BIN_DIR_CMAKE}\""
        "METALLIC_STREAMLINE_INTERPOSER_DLL=\"sl.interposer.dll\""
    )
    target_include_directories(metallic_streamline INTERFACE "${METALLIC_STREAMLINE_INCLUDE_DIR}")
    target_link_libraries(metallic_streamline INTERFACE "${METALLIC_STREAMLINE_LIBRARY}")
    target_compile_definitions(metallic_streamline INTERFACE
        STREAMLINE_FEATURE_DLSS_RR=1
        "METALLIC_STREAMLINE_BIN_DIR=\"${METALLIC_STREAMLINE_BIN_DIR_CMAKE}\""
        "METALLIC_STREAMLINE_INTERPOSER_DLL=\"sl.interposer.dll\""
    )

    file(GLOB METALLIC_STREAMLINE_RUNTIME_DLLS
        "${METALLIC_STREAMLINE_BIN_DIR}/*.dll"
    )
    file(GLOB METALLIC_STREAMLINE_RUNTIME_JSONS
        "${METALLIC_STREAMLINE_SCRIPTS_DIR}/*.json"
    )
    set(METALLIC_STREAMLINE_RUNTIME_FILES
        ${METALLIC_STREAMLINE_RUNTIME_DLLS}
        ${METALLIC_STREAMLINE_RUNTIME_JSONS}
        CACHE INTERNAL "NVIDIA Streamline runtime files to copy next to executables"
    )
    message(STATUS "NVIDIA Streamline enabled: ${METALLIC_STREAMLINE_SDK_ROOT}")
else()
    metallic_set_streamline_available(0)
    message(STATUS "NVIDIA Streamline SDK binaries were not found at ${METALLIC_STREAMLINE_ROOT} or ${METALLIC_STREAMLINE_ROOT}/_sdk; DLSS-RR pass will compile as unsupported.")
endif()
