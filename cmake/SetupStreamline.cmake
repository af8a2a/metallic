# SetupStreamline.cmake — Download and configure NVIDIA Streamline SDK (Windows only)

if(NOT WIN32)
    return()
endif()

option(METALLIC_ENABLE_STREAMLINE "Enable NVIDIA Streamline (DLSS) integration" ON)

if(NOT METALLIC_ENABLE_STREAMLINE)
    return()
endif()

set(STREAMLINE_VERSION "2.10.3")
set(STREAMLINE_ROOT "${CMAKE_SOURCE_DIR}/External/streamline")
set(STREAMLINE_MARKER "${STREAMLINE_ROOT}/include/sl.h")

if(NOT EXISTS "${STREAMLINE_MARKER}")
    set(STREAMLINE_ARCHIVE "streamline-sdk-v${STREAMLINE_VERSION}.zip")
    set(STREAMLINE_URL "https://github.com/NVIDIA-RTX/Streamline/releases/download/v${STREAMLINE_VERSION}/${STREAMLINE_ARCHIVE}")
    set(STREAMLINE_DOWNLOAD_PATH "${CMAKE_BINARY_DIR}/${STREAMLINE_ARCHIVE}")

    file(MAKE_DIRECTORY "${STREAMLINE_ROOT}")

    message(STATUS "Downloading Streamline SDK ${STREAMLINE_VERSION} to ${STREAMLINE_ROOT}...")
    file(DOWNLOAD "${STREAMLINE_URL}" "${STREAMLINE_DOWNLOAD_PATH}"
        STATUS STREAMLINE_DOWNLOAD_STATUS
        SHOW_PROGRESS
        TLS_VERIFY ON
        INACTIVITY_TIMEOUT 60
        LOG STREAMLINE_DOWNLOAD_LOG
    )
    list(GET STREAMLINE_DOWNLOAD_STATUS 0 STREAMLINE_DOWNLOAD_ERROR)

    # Validate: check status code and file size
    set(STREAMLINE_DOWNLOAD_OK FALSE)
    if(STREAMLINE_DOWNLOAD_ERROR EQUAL 0 AND EXISTS "${STREAMLINE_DOWNLOAD_PATH}")
        file(SIZE "${STREAMLINE_DOWNLOAD_PATH}" STREAMLINE_DOWNLOAD_SIZE)
        if(STREAMLINE_DOWNLOAD_SIZE GREATER 1000000)
            set(STREAMLINE_DOWNLOAD_OK TRUE)
        endif()
    endif()

    # Fallback: use PowerShell on Windows (handles GitHub redirects better)
    if(NOT STREAMLINE_DOWNLOAD_OK)
        file(REMOVE "${STREAMLINE_DOWNLOAD_PATH}")
        message(STATUS "CMake download failed; trying PowerShell fallback...")
        execute_process(
            COMMAND powershell -Command
                "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '${STREAMLINE_URL}' -OutFile '${STREAMLINE_DOWNLOAD_PATH}' -UseBasicParsing"
            RESULT_VARIABLE STREAMLINE_PS_RESULT
            OUTPUT_VARIABLE STREAMLINE_PS_OUTPUT
            ERROR_VARIABLE STREAMLINE_PS_ERROR
            TIMEOUT 300
        )
        if(STREAMLINE_PS_RESULT EQUAL 0 AND EXISTS "${STREAMLINE_DOWNLOAD_PATH}")
            file(SIZE "${STREAMLINE_DOWNLOAD_PATH}" STREAMLINE_DOWNLOAD_SIZE)
            if(STREAMLINE_DOWNLOAD_SIZE GREATER 1000000)
                set(STREAMLINE_DOWNLOAD_OK TRUE)
            endif()
        endif()
    endif()

    if(NOT STREAMLINE_DOWNLOAD_OK)
        message(WARNING
            "Failed to download Streamline SDK.\n"
            "Please download manually from:\n  ${STREAMLINE_URL}\n"
            "and extract to: ${STREAMLINE_ROOT}\n"
            "DLSS integration will be disabled for now.")
        file(REMOVE "${STREAMLINE_DOWNLOAD_PATH}")
        set(METALLIC_ENABLE_STREAMLINE OFF CACHE BOOL "" FORCE)
        return()
    endif()

    message(STATUS "Extracting Streamline SDK to ${STREAMLINE_ROOT}...")
    file(ARCHIVE_EXTRACT INPUT "${STREAMLINE_DOWNLOAD_PATH}" DESTINATION "${STREAMLINE_ROOT}")

    # The zip may contain a top-level directory; flatten it if so
    file(GLOB STREAMLINE_SUBDIRS LIST_DIRECTORIES true "${STREAMLINE_ROOT}/*/include/sl.h")
    if(STREAMLINE_SUBDIRS)
        list(GET STREAMLINE_SUBDIRS 0 STREAMLINE_NESTED_MARKER)
        get_filename_component(STREAMLINE_NESTED_INCLUDE "${STREAMLINE_NESTED_MARKER}" DIRECTORY)
        get_filename_component(STREAMLINE_NESTED_ROOT "${STREAMLINE_NESTED_INCLUDE}" DIRECTORY)
        file(GLOB STREAMLINE_NESTED_CONTENTS "${STREAMLINE_NESTED_ROOT}/*")
        foreach(ITEM ${STREAMLINE_NESTED_CONTENTS})
            get_filename_component(ITEM_NAME "${ITEM}" NAME)
            if(NOT "${ITEM}" STREQUAL "${STREAMLINE_ROOT}/${ITEM_NAME}")
                file(RENAME "${ITEM}" "${STREAMLINE_ROOT}/${ITEM_NAME}")
            endif()
        endforeach()
        file(REMOVE_RECURSE "${STREAMLINE_NESTED_ROOT}")
    endif()

    file(REMOVE "${STREAMLINE_DOWNLOAD_PATH}")

    if(NOT EXISTS "${STREAMLINE_MARKER}")
        message(WARNING "Streamline SDK extraction did not produce expected headers. DLSS disabled.")
        set(METALLIC_ENABLE_STREAMLINE OFF CACHE BOOL "" FORCE)
        return()
    endif()

    message(STATUS "Streamline SDK ${STREAMLINE_VERSION} installed to ${STREAMLINE_ROOT}")
endif()

# Expose paths for consumers
set(STREAMLINE_INCLUDE_DIR "${STREAMLINE_ROOT}/include")
set(STREAMLINE_LIB_DIR "${STREAMLINE_ROOT}/lib/x64")
set(STREAMLINE_BIN_DIR "${STREAMLINE_ROOT}/bin/x64")

# Collect import library
set(STREAMLINE_LINK_LIBS "")
if(EXISTS "${STREAMLINE_LIB_DIR}/sl.interposer.lib")
    list(APPEND STREAMLINE_LINK_LIBS "${STREAMLINE_LIB_DIR}/sl.interposer.lib")
endif()

# Collect runtime DLLs to copy
file(GLOB METALLIC_STREAMLINE_RUNTIME_DLLS "${STREAMLINE_BIN_DIR}/*.dll")

# Create imported target
if(NOT TARGET streamline::streamline)
    add_library(streamline::streamline INTERFACE IMPORTED)
    target_include_directories(streamline::streamline INTERFACE "${STREAMLINE_INCLUDE_DIR}")
    if(STREAMLINE_LINK_LIBS)
        target_link_libraries(streamline::streamline INTERFACE ${STREAMLINE_LINK_LIBS})
    endif()
    target_compile_definitions(streamline::streamline INTERFACE METALLIC_HAS_STREAMLINE=1)
endif()
