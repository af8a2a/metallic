option(METALLIC_ENABLE_DLSS_NR "Enable experimental DLSS-NR when the user-provided DLL is available (Windows/Vulkan)" ON)
if(METALLIC_STREAMLINE_SDK_ROOT)
    set(METALLIC_DLSS_NR_DEFAULT_NGX_ROOT "${METALLIC_STREAMLINE_SDK_ROOT}/external/ngx-sdk")
else()
    set(METALLIC_DLSS_NR_DEFAULT_NGX_ROOT "${CMAKE_SOURCE_DIR}/External/streamline/external/ngx-sdk")
endif()
set(METALLIC_DLSS_NR_NGX_ROOT "${METALLIC_DLSS_NR_DEFAULT_NGX_ROOT}" CACHE PATH "NGX SDK used by experimental DLSS-NR")
# NR has no public runtime distribution. Only a manually installed local DLL
# opts this checkout in; old cache overrides must not load a sibling checkout.
unset(METALLIC_DLSS_NR_RUNTIME CACHE)
set(METALLIC_DLSS_NR_RUNTIME "${CMAKE_SOURCE_DIR}/External/nvngx_dlssnr.dll")
# Reconfigure on the next build when the user adds or removes this exact file.
file(GLOB METALLIC_DLSS_NR_RUNTIME_FILES LIST_DIRECTORIES false CONFIGURE_DEPENDS
    "${METALLIC_DLSS_NR_RUNTIME}")

add_library(metallic_dlss_nr INTERFACE)
add_library(metallic::dlss_nr ALIAS metallic_dlss_nr)
set(METALLIC_HAS_DLSS_NR 0)
set(METALLIC_DLSS_NR_UNAVAILABLE_REASON "")
if(NOT METALLIC_ENABLE_DLSS_NR)
    set(METALLIC_DLSS_NR_UNAVAILABLE_REASON "METALLIC_ENABLE_DLSS_NR=OFF")
elseif(NOT METALLIC_DLSS_NR_RUNTIME_FILES)
    set(METALLIC_DLSS_NR_UNAVAILABLE_REASON "manually place nvngx_dlssnr.dll in ${CMAKE_SOURCE_DIR}/External to enable it")
elseif(NOT WIN32 OR NOT MSVC OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(METALLIC_DLSS_NR_UNAVAILABLE_REASON "Windows x64/MSVC is required")
elseif(NOT METALLIC_HAS_STREAMLINE)
    set(METALLIC_DLSS_NR_UNAVAILABLE_REASON "a usable Streamline SDK is required")
else()
    foreach(required_file include/nvsdk_ngx_vk.h include/nvsdk_ngx_params.h)
        if(NOT EXISTS "${METALLIC_DLSS_NR_NGX_ROOT}/${required_file}")
            set(METALLIC_DLSS_NR_UNAVAILABLE_REASON "missing NGX SDK file: ${METALLIC_DLSS_NR_NGX_ROOT}/${required_file}")
            break()
        endif()
        file(READ "${METALLIC_DLSS_NR_NGX_ROOT}/${required_file}" file_prefix LIMIT 128)
        if(file_prefix MATCHES "version https://git-lfs.github.com/spec/v1")
            set(METALLIC_DLSS_NR_UNAVAILABLE_REASON "NGX SDK file is an LFS pointer: ${required_file}; use the packaged Streamline SDK")
            break()
        endif()
    endforeach()
endif()

if(METALLIC_DLSS_NR_UNAVAILABLE_REASON STREQUAL "")
    set(METALLIC_HAS_DLSS_NR 1)
    target_include_directories(metallic_dlss_nr INTERFACE "${METALLIC_DLSS_NR_NGX_ROOT}/include")
    message(STATUS "Experimental DLSS-NR enabled: ${METALLIC_DLSS_NR_RUNTIME}")
else()
    message(STATUS "Experimental DLSS-NR disabled (METALLIC_HAS_DLSS_NR=0): ${METALLIC_DLSS_NR_UNAVAILABLE_REASON}")
endif()
target_compile_definitions(metallic_dlss_nr INTERFACE METALLIC_HAS_DLSS_NR=${METALLIC_HAS_DLSS_NR})

function(metallic_copy_dlss_nr_runtime target_name)
    if(METALLIC_HAS_DLSS_NR)
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${METALLIC_DLSS_NR_RUNTIME}" "$<TARGET_FILE_DIR:${target_name}>/nvngx_dlssnr.dll"
            VERBATIM)
    else()
        # Do not leave an old experimental runtime in a newly disabled build.
        add_custom_command(TARGET ${target_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E rm -f "$<TARGET_FILE_DIR:${target_name}>/nvngx_dlssnr.dll"
            VERBATIM)
    endif()
endfunction()
