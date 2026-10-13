# Optional in-process SPIR-V cleanup. Finding the optimizer does not enable any
# shader transform: the default compiler policy continues to emit Slang output.
option(METALLIC_ENABLE_SPIRV_TOOLS "Use installed SPIRV-Tools for optional shader cleanup" ON)

function(metallic_configure_spirv_tools target_name)
    set(_optimizer_target "")
    if(METALLIC_ENABLE_SPIRV_TOOLS)
        if(NOT TARGET SPIRV-Tools-opt AND NOT TARGET SPIRV-Tools::SPIRV-Tools-opt)
            # Windows Vulkan SDK exports are relocated one directory deeper than
            # their generated prefixes expect. Use SDK libraries below instead
            # of loading those exports, which fail before a fallback can run.
            if(NOT WIN32 OR DEFINED SPIRV-Tools-opt_DIR)
                find_package(SPIRV-Tools-opt CONFIG QUIET)
            endif()
        endif()
        if(TARGET SPIRV-Tools-opt)
            set(_optimizer_target SPIRV-Tools-opt)
        elseif(TARGET SPIRV-Tools::SPIRV-Tools-opt)
            set(_optimizer_target SPIRV-Tools::SPIRV-Tools-opt)
        elseif(TARGET MetallicInstalledSPIRVToolsOpt)
            set(_optimizer_target MetallicInstalledSPIRVToolsOpt)
        else()
            find_path(METALLIC_SPIRV_TOOLS_INCLUDE_DIR spirv-tools/optimizer.hpp
                HINTS ${Vulkan_INCLUDE_DIR} ${Vulkan_INCLUDE_DIRS} "$ENV{VULKAN_SDK}/include")
            set(_library_hints "$ENV{VULKAN_SDK}/Lib" "$ENV{VULKAN_SDK}/lib")
            foreach(_include_dir IN LISTS Vulkan_INCLUDE_DIR Vulkan_INCLUDE_DIRS METALLIC_SPIRV_TOOLS_INCLUDE_DIR)
                get_filename_component(_sdk_root "${_include_dir}" DIRECTORY)
                list(APPEND _library_hints "${_sdk_root}/Lib" "${_sdk_root}/lib" "${_sdk_root}/lib64")
            endforeach()
            find_library(METALLIC_SPIRV_TOOLS_OPT_LIBRARY NAMES SPIRV-Tools-opt HINTS ${_library_hints})
            find_library(METALLIC_SPIRV_TOOLS_LIBRARY NAMES SPIRV-Tools SPIRV-Tools-static HINTS ${_library_hints})
            find_library(METALLIC_SPIRV_TOOLS_OPT_DEBUG_LIBRARY NAMES SPIRV-Tools-optd HINTS ${_library_hints})
            find_library(METALLIC_SPIRV_TOOLS_DEBUG_LIBRARY NAMES SPIRV-Toolsd HINTS ${_library_hints})
            set(_compatible TRUE)
            if(MSVC AND (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_CONFIGURATION_TYPES) AND
                (NOT METALLIC_SPIRV_TOOLS_OPT_DEBUG_LIBRARY OR NOT METALLIC_SPIRV_TOOLS_DEBUG_LIBRARY))
                set(_compatible FALSE)
            endif()
            if(_compatible AND METALLIC_SPIRV_TOOLS_INCLUDE_DIR AND
                METALLIC_SPIRV_TOOLS_OPT_LIBRARY AND METALLIC_SPIRV_TOOLS_LIBRARY)
                add_library(MetallicInstalledSPIRVTools STATIC IMPORTED)
                set_target_properties(MetallicInstalledSPIRVTools PROPERTIES
                    IMPORTED_LOCATION "${METALLIC_SPIRV_TOOLS_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${METALLIC_SPIRV_TOOLS_INCLUDE_DIR}")
                add_library(MetallicInstalledSPIRVToolsOpt STATIC IMPORTED)
                set_target_properties(MetallicInstalledSPIRVToolsOpt PROPERTIES
                    IMPORTED_LOCATION "${METALLIC_SPIRV_TOOLS_OPT_LIBRARY}"
                    INTERFACE_LINK_LIBRARIES MetallicInstalledSPIRVTools)
                if(METALLIC_SPIRV_TOOLS_OPT_DEBUG_LIBRARY AND METALLIC_SPIRV_TOOLS_DEBUG_LIBRARY)
                    set_target_properties(MetallicInstalledSPIRVTools PROPERTIES
                        IMPORTED_LOCATION_DEBUG "${METALLIC_SPIRV_TOOLS_DEBUG_LIBRARY}")
                    set_target_properties(MetallicInstalledSPIRVToolsOpt PROPERTIES
                        IMPORTED_LOCATION_DEBUG "${METALLIC_SPIRV_TOOLS_OPT_DEBUG_LIBRARY}")
                endif()
                set(_optimizer_target MetallicInstalledSPIRVToolsOpt)
            endif()
            mark_as_advanced(METALLIC_SPIRV_TOOLS_INCLUDE_DIR METALLIC_SPIRV_TOOLS_OPT_LIBRARY
                METALLIC_SPIRV_TOOLS_LIBRARY METALLIC_SPIRV_TOOLS_OPT_DEBUG_LIBRARY METALLIC_SPIRV_TOOLS_DEBUG_LIBRARY)
        endif()
    endif()
    if(_optimizer_target)
        target_link_libraries(${target_name} PRIVATE ${_optimizer_target})
        target_compile_definitions(${target_name} PRIVATE METALLIC_HAS_SPIRV_TOOLS=1)
        message(STATUS "Optional SPIRV-Tools shader cleanup: ${_optimizer_target}")
    else()
        target_compile_definitions(${target_name} PRIVATE METALLIC_HAS_SPIRV_TOOLS=0)
        message(STATUS "Optional SPIRV-Tools shader cleanup unavailable; explicit requests will fail")
    endif()
endfunction()
