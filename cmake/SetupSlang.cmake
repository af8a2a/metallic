set(SLANG_VERSION "2026.1.2")

set(SLANG_ROOT "" CACHE PATH "Path to a Slang SDK root containing include/slang.h")

if(NOT SLANG_ROOT)
    if(DEFINED ENV{SLANG_ROOT} AND EXISTS "$ENV{SLANG_ROOT}/include/slang.h")
        set(SLANG_ROOT "$ENV{SLANG_ROOT}")
    elseif(DEFINED ENV{SLANG_DIR} AND EXISTS "$ENV{SLANG_DIR}/include/slang.h")
        set(SLANG_ROOT "$ENV{SLANG_DIR}")
    else()
        set(SLANG_ROOT "${CMAKE_SOURCE_DIR}/External/slang")
    endif()
endif()

set(SLANG_MARKER "${SLANG_ROOT}/include/slang.h")

if(NOT EXISTS "${SLANG_MARKER}")
    if(WIN32)
        set(SLANG_ARCHIVE "slang-${SLANG_VERSION}-windows-x86_64.zip")
        set(SLANG_EXTRACTED_DIR "${SLANG_ROOT}/slang-${SLANG_VERSION}-windows-x86_64")
    elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        set(SLANG_ARCHIVE "slang-${SLANG_VERSION}-macos-aarch64.zip")
        set(SLANG_EXTRACTED_DIR "${SLANG_ROOT}/slang-${SLANG_VERSION}-macos-aarch64")
    elseif(APPLE)
        set(SLANG_ARCHIVE "slang-${SLANG_VERSION}-macos-x86_64.zip")
        set(SLANG_EXTRACTED_DIR "${SLANG_ROOT}/slang-${SLANG_VERSION}-macos-x86_64")
    elseif(UNIX AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(SLANG_ARCHIVE "slang-${SLANG_VERSION}-linux-x86_64.zip")
        set(SLANG_EXTRACTED_DIR "${SLANG_ROOT}/slang-${SLANG_VERSION}-linux-x86_64")
    else()
        message(FATAL_ERROR "Unsupported platform for automatic Slang download. Set SLANG_ROOT to an existing Slang SDK.")
    endif()

    set(SLANG_URL "https://github.com/shader-slang/slang/releases/download/v${SLANG_VERSION}/${SLANG_ARCHIVE}")
    set(SLANG_DOWNLOAD_PATH "${CMAKE_BINARY_DIR}/${SLANG_ARCHIVE}")

    file(MAKE_DIRECTORY "${SLANG_ROOT}")

    message(STATUS "Downloading Slang ${SLANG_VERSION} to ${SLANG_ROOT}...")
    file(DOWNLOAD "${SLANG_URL}" "${SLANG_DOWNLOAD_PATH}"
        STATUS SLANG_DOWNLOAD_STATUS
        SHOW_PROGRESS
    )
    list(GET SLANG_DOWNLOAD_STATUS 0 SLANG_DOWNLOAD_ERROR)
    if(NOT SLANG_DOWNLOAD_ERROR EQUAL 0)
        list(GET SLANG_DOWNLOAD_STATUS 1 SLANG_DOWNLOAD_MSG)
        message(FATAL_ERROR
            "Failed to download Slang: ${SLANG_DOWNLOAD_MSG}\n"
            "Provide a local SDK with -DSLANG_ROOT=<path> or set the SLANG_ROOT/SLANG_DIR environment variable.")
    endif()

    message(STATUS "Extracting Slang to ${SLANG_ROOT}...")
    file(ARCHIVE_EXTRACT INPUT "${SLANG_DOWNLOAD_PATH}" DESTINATION "${SLANG_ROOT}")

    if(EXISTS "${SLANG_EXTRACTED_DIR}")
        file(GLOB SLANG_EXTRACTED_CONTENTS "${SLANG_EXTRACTED_DIR}/*")
        foreach(ITEM ${SLANG_EXTRACTED_CONTENTS})
            get_filename_component(ITEM_NAME "${ITEM}" NAME)
            file(RENAME "${ITEM}" "${SLANG_ROOT}/${ITEM_NAME}")
        endforeach()
        file(REMOVE_RECURSE "${SLANG_EXTRACTED_DIR}")
    endif()

    file(REMOVE "${SLANG_DOWNLOAD_PATH}")
    message(STATUS "Slang ${SLANG_VERSION} installed to ${SLANG_ROOT}")
endif()

if(NOT TARGET slang::slang)
    add_library(slang::slang INTERFACE IMPORTED)

    set(SLANG_INCLUDE_DIR "${SLANG_ROOT}/include")
    set(SLANG_LINK_LIBS "")
    set(METALLIC_SLANG_RUNTIME_LIBS "")

    if(WIN32)
        foreach(LIB_NAME slang-compiler.lib slang.lib slang-rt.lib gfx.lib)
            if(EXISTS "${SLANG_ROOT}/lib/${LIB_NAME}")
                list(APPEND SLANG_LINK_LIBS "${SLANG_ROOT}/lib/${LIB_NAME}")
            endif()
        endforeach()
        file(GLOB METALLIC_SLANG_RUNTIME_LIBS
            "${SLANG_ROOT}/bin/*.dll"
            "${SLANG_ROOT}/lib/*.dll")
    elseif(APPLE)
        foreach(LIB_NAME libslang-compiler.dylib libslang.dylib libslang-rt.dylib libgfx.dylib)
            if(EXISTS "${SLANG_ROOT}/lib/${LIB_NAME}")
                list(APPEND SLANG_LINK_LIBS "${SLANG_ROOT}/lib/${LIB_NAME}")
            endif()
        endforeach()
        file(GLOB METALLIC_SLANG_RUNTIME_LIBS "${SLANG_ROOT}/lib/*.dylib")
    else()
        foreach(LIB_NAME libslang-compiler.so libslang.so libslang-rt.so libgfx.so)
            if(EXISTS "${SLANG_ROOT}/lib/${LIB_NAME}")
                list(APPEND SLANG_LINK_LIBS "${SLANG_ROOT}/lib/${LIB_NAME}")
            endif()
        endforeach()
        file(GLOB METALLIC_SLANG_RUNTIME_LIBS "${SLANG_ROOT}/lib/*.so*")
    endif()

    if(NOT EXISTS "${SLANG_INCLUDE_DIR}/slang.h")
        message(FATAL_ERROR "Slang headers were not found under ${SLANG_ROOT}")
    endif()

    if(NOT SLANG_LINK_LIBS)
        message(FATAL_ERROR "No Slang libraries were found under ${SLANG_ROOT}/lib")
    endif()

    target_include_directories(slang::slang INTERFACE "${SLANG_INCLUDE_DIR}")
    target_link_libraries(slang::slang INTERFACE ${SLANG_LINK_LIBS})
endif()

