add_library(metallic_openusd INTERFACE)
add_library(metallic::openusd ALIAS metallic_openusd)
target_compile_definitions(metallic_openusd INTERFACE
    METALLIC_HAS_OPENUSD=$<BOOL:${METALLIC_ENABLE_OPENUSD}>)

function(metallic_copy_openusd_runtime target_name)
    # No deployment when USD support is disabled.
endfunction()

if(NOT METALLIC_ENABLE_OPENUSD)
    return()
endif()

if(METALLIC_DEPENDENCY_MODE STREQUAL "PREBUILT")
    set(TBB_DIR "${METALLIC_DEPENDENCY_ROOT}/lib/cmake/TBB")
    find_package(TBB CONFIG REQUIRED GLOBAL COMPONENTS tbb
        PATHS "${METALLIC_DEPENDENCY_ROOT}" NO_DEFAULT_PATH)
    set(PXR_FIND_TBB_IN_CONFIG ON)
    set(pxr_DIR "${METALLIC_DEPENDENCY_ROOT}")
    find_package(pxr CONFIG REQUIRED GLOBAL PATHS "${METALLIC_DEPENDENCY_ROOT}" NO_DEFAULT_PATH)
    if(NOT TARGET usd_m)
        message(FATAL_ERROR "Metallic requires a monolithic OpenUSD package (usd_m)")
    endif()
    target_link_libraries(metallic_openusd INTERFACE usd_m)
    target_include_directories(metallic_openusd SYSTEM INTERFACE "${PXR_INCLUDE_DIRS}")

    function(metallic_copy_openusd_runtime target_name)
        if(NOT WIN32)
            # Imported shared libraries carry the installed library paths in RPATH.
            return()
        endif()
        add_custom_target(${target_name}OpenUsdRuntime
            COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:${target_name}>"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_FILE:usd_m>" "$<TARGET_FILE_DIR:${target_name}>"
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_FILE:TBB::tbb>" "$<TARGET_FILE_DIR:${target_name}>"
            COMMAND "${CMAKE_COMMAND}" -E copy_directory
                "${METALLIC_DEPENDENCY_ROOT}/lib/usd" "$<TARGET_FILE_DIR:${target_name}>/usd"
            COMMAND "${CMAKE_COMMAND}" -E copy_directory
                "${METALLIC_DEPENDENCY_ROOT}/plugin/usd" "$<TARGET_FILE_DIR:${target_name}>/../plugin/usd"
            VERBATIM)
        add_dependencies(${target_name} ${target_name}OpenUsdRuntime)
    endfunction()
    return()
endif()

if(METALLIC_DEPENDENCY_MODE STREQUAL "SOURCE" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/oneTBB/CMakeLists.txt")
    set(_METALLIC_BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS}")
    set(BUILD_SHARED_LIBS ON)
    metallic_tbb_options()
    set(_METALLIC_CMAKE_POLICY_VERSION_MINIMUM "${CMAKE_POLICY_VERSION_MINIMUM}")
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    add_subdirectory(oneTBB)
    set(CMAKE_POLICY_VERSION_MINIMUM "${_METALLIC_CMAKE_POLICY_VERSION_MINIMUM}")
    unset(_METALLIC_CMAKE_POLICY_VERSION_MINIMUM)
endif()

if(METALLIC_DEPENDENCY_MODE STREQUAL "SOURCE" AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/OpenUSD/CMakeLists.txt")
    if(NOT TARGET TBB::tbb)
        message(FATAL_ERROR "OpenUSD requires the vendored oneTBB target")
    endif()

    set(TBB_FOUND TRUE)
    set(PXR_FIND_TBB_IN_CONFIG OFF CACHE BOOL "Use the in-tree oneTBB target" FORCE)
    metallic_openusd_options()
    add_subdirectory(OpenUSD)

    set(BUILD_SHARED_LIBS "${_METALLIC_BUILD_SHARED_LIBS}")
    unset(_METALLIC_BUILD_SHARED_LIBS)

    target_link_libraries(metallic_openusd INTERFACE usd_m)
    target_include_directories(metallic_openusd SYSTEM INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/OpenUSD"
        "${CMAKE_CURRENT_BINARY_DIR}/OpenUSD/include"
    )

    function(metallic_copy_openusd_runtime target_name)
        if(NOT TARGET ${target_name} OR NOT TARGET usd_m OR NOT TARGET tbb)
            return()
        endif()

        set(openusd_binary_root "${CMAKE_BINARY_DIR}/External/OpenUSD")
        set(openusd_source_root "${CMAKE_SOURCE_DIR}/External/OpenUSD")
        set(runtime_root "$<TARGET_FILE_DIR:${target_name}>")
        set(core_resource_root "${runtime_root}/usd")
        set(plugin_resource_root "${runtime_root}/../plugin/usd")

        set(runtime_commands
            COMMAND ${CMAKE_COMMAND} -E make_directory "${runtime_root}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:usd_m>"
                "${runtime_root}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:tbb>"
                "${runtime_root}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${core_resource_root}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${openusd_binary_root}/plugins_plugInfo.json"
                "${core_resource_root}/plugInfo.json"
        )

        file(GLOB openusd_core_plug_infos
            RELATIVE "${openusd_binary_root}/pxr/usd"
            "${openusd_binary_root}/pxr/usd/*/plugInfo.json"
        )
        foreach(relative_plug_info IN LISTS openusd_core_plug_infos)
            get_filename_component(module_name "${relative_plug_info}" DIRECTORY)
            set(module_resource_root "${core_resource_root}/${module_name}/resources")
            list(APPEND runtime_commands
                COMMAND ${CMAKE_COMMAND} -E make_directory "${module_resource_root}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${openusd_binary_root}/pxr/usd/${relative_plug_info}"
                    "${module_resource_root}/plugInfo.json"
            )

            set(module_source_root "${openusd_source_root}/pxr/usd/${module_name}")
            if(EXISTS "${module_source_root}/generatedSchema.usda")
                list(APPEND runtime_commands
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${module_source_root}/generatedSchema.usda"
                        "${module_resource_root}/generatedSchema.usda"
                )
            endif()
            if(EXISTS "${module_source_root}/schema.usda")
                list(APPEND runtime_commands
                    COMMAND ${CMAKE_COMMAND} -E make_directory
                        "${module_resource_root}/${module_name}"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${module_source_root}/schema.usda"
                        "${module_resource_root}/${module_name}/schema.usda"
                )
            endif()
        endforeach()

        set(usd_shaders_binary_root
            "${openusd_binary_root}/pxr/usd/plugin/usdShaders")
        set(usd_shaders_source_root
            "${openusd_source_root}/pxr/usd/plugin/usdShaders")
        if(EXISTS "${usd_shaders_binary_root}/plugInfo.json")
            list(APPEND runtime_commands
                COMMAND ${CMAKE_COMMAND} -E make_directory "${plugin_resource_root}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${openusd_binary_root}/usd_plugInfo.json"
                    "${plugin_resource_root}/plugInfo.json"
                COMMAND ${CMAKE_COMMAND} -E make_directory
                    "${plugin_resource_root}/usdShaders/resources"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${usd_shaders_binary_root}/plugInfo.json"
                    "${plugin_resource_root}/usdShaders/resources/plugInfo.json"
                COMMAND ${CMAKE_COMMAND} -E copy_directory
                    "${usd_shaders_source_root}/shaders"
                    "${plugin_resource_root}/usdShaders/resources/shaders"
            )
        endif()

        add_custom_target(${target_name}OpenUsdRuntime
            ${runtime_commands}
            DEPENDS usd_m tbb
            COMMENT "Deploying OpenUSD runtime for ${target_name}"
            VERBATIM
        )
        add_dependencies(${target_name} ${target_name}OpenUsdRuntime)
    endfunction()
endif()

