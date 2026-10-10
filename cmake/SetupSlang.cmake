option(METALLIC_SLANG_AUTO_DOWNLOAD "Install the pinned Slang SDK when the default installation is missing" ON)
set(METALLIC_SLANG_ROOT "${CMAKE_SOURCE_DIR}/External/slang")
if(DEFINED SLANG_ROOT)
    # An explicit SDK is user-managed, including when the path is invalid.
    set(METALLIC_SLANG_ROOT "${SLANG_ROOT}")
    return()
endif()

function(metallic_ensure_slang)
    if(EXISTS "${METALLIC_SLANG_ROOT}/cmake/slangConfig.cmake")
        return()
    endif()
    if(NOT METALLIC_SLANG_AUTO_DOWNLOAD)
        message(FATAL_ERROR "Slang is missing. Enable METALLIC_SLANG_AUTO_DOWNLOAD or set SLANG_ROOT to a Slang 2026.18.2 SDK.")
    endif()
    if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8 OR CMAKE_CROSSCOMPILING
            OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64)$")
        message(FATAL_ERROR "Automatic Slang installation supports native Windows x64 builds. Set SLANG_ROOT to a Slang 2026.18.2 SDK for this platform.")
    endif()

    find_program(METALLIC_POWERSHELL_EXECUTABLE NAMES pwsh powershell REQUIRED)
    # Multiple CLion presets can configure concurrently against the same SDK.
    file(MAKE_DIRECTORY "${CMAKE_SOURCE_DIR}/.cache/slang")
    file(LOCK "${CMAKE_SOURCE_DIR}/.cache/slang/install.lock"
        GUARD FUNCTION TIMEOUT 600 RESULT_VARIABLE lock_result)
    if(NOT lock_result STREQUAL "0")
        message(FATAL_ERROR "Cannot lock the Slang installation: ${lock_result}")
    endif()
    if(EXISTS "${METALLIC_SLANG_ROOT}/cmake/slangConfig.cmake")
        return()
    endif()

    message(STATUS "Installing Slang 2026.18.2 (verified archive; cached in .cache/slang)")
    execute_process(
        COMMAND "${METALLIC_POWERSHELL_EXECUTABLE}" -NoProfile -NonInteractive
            -ExecutionPolicy Bypass -File "${CMAKE_SOURCE_DIR}/scripts/InstallSlang.ps1"
            -Destination "${METALLIC_SLANG_ROOT}"
        RESULT_VARIABLE install_result)
    if(NOT install_result STREQUAL "0" OR NOT EXISTS "${METALLIC_SLANG_ROOT}/cmake/slangConfig.cmake")
        message(FATAL_ERROR
            "Slang installation failed (${install_result}). See installer output above. "
            "Existing installations are not overwritten; set SLANG_ROOT to a complete Slang 2026.18.2 SDK.")
    endif()
endfunction()

metallic_ensure_slang()
