set(SLANG_VERSION "2026.1.2")
set(SLANG_DIR "${CMAKE_SOURCE_DIR}/External/slang")
set(SLANG_MARKER "${SLANG_DIR}/include/slang.h")

if(NOT EXISTS "${SLANG_MARKER}")
    set(SLANG_ARCHIVE "slang-${SLANG_VERSION}-macos-aarch64.zip")
    set(SLANG_URL "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/${SLANG_ARCHIVE}")
    set(SLANG_DOWNLOAD_PATH "${CMAKE_BINARY_DIR}/${SLANG_ARCHIVE}")

    message(STATUS "Downloading Slang ${SLANG_VERSION}...")
    file(DOWNLOAD "${SLANG_URL}" "${SLANG_DOWNLOAD_PATH}"
        STATUS SLANG_DOWNLOAD_STATUS
        SHOW_PROGRESS
    )
    list(GET SLANG_DOWNLOAD_STATUS 0 SLANG_DOWNLOAD_ERROR)
    if(NOT SLANG_DOWNLOAD_ERROR EQUAL 0)
        list(GET SLANG_DOWNLOAD_STATUS 1 SLANG_DOWNLOAD_MSG)
        message(FATAL_ERROR "Failed to download Slang: ${SLANG_DOWNLOAD_MSG}")
    endif()

    message(STATUS "Extracting Slang to ${SLANG_DIR}...")
    file(ARCHIVE_EXTRACT INPUT "${SLANG_DOWNLOAD_PATH}" DESTINATION "${SLANG_DIR}")

    # The archive extracts into a subdirectory; move contents up
    set(SLANG_EXTRACTED_DIR "${SLANG_DIR}/slang-${SLANG_VERSION}-macos-aarch64")
    if(EXISTS "${SLANG_EXTRACTED_DIR}")
        file(GLOB SLANG_EXTRACTED_CONTENTS "${SLANG_EXTRACTED_DIR}/*")
        foreach(ITEM ${SLANG_EXTRACTED_CONTENTS})
            get_filename_component(ITEM_NAME "${ITEM}" NAME)
            file(RENAME "${ITEM}" "${SLANG_DIR}/${ITEM_NAME}")
        endforeach()
        file(REMOVE_RECURSE "${SLANG_EXTRACTED_DIR}")
    endif()

    file(REMOVE "${SLANG_DOWNLOAD_PATH}")
    message(STATUS "Slang ${SLANG_VERSION} installed to ${SLANG_DIR}")
endif()
