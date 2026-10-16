include_guard(GLOBAL)

function(_metallic_shader_identity_link_inputs target visited files targets complete)
    set(_visited "${${visited}}")
    set(_files "${${files}}")
    set(_targets "${${targets}}")
    set(_complete "${${complete}}")
    if(TARGET "${target}" AND NOT "${target}" IN_LIST _visited)
        list(APPEND _visited "${target}")
        get_target_property(_alias "${target}" ALIASED_TARGET)
        if(_alias)
            set(target "${_alias}")
        endif()
        get_target_property(_type "${target}" TYPE)
        if(_type STREQUAL "STATIC_LIBRARY" OR _type STREQUAL "SHARED_LIBRARY" OR _type STREQUAL "UNKNOWN_LIBRARY")
            list(APPEND _files "$<TARGET_LINKER_FILE:${target}>")
            list(APPEND _targets "${target}")
        elseif(NOT _type STREQUAL "INTERFACE_LIBRARY")
            set(_complete FALSE)
        endif()
        foreach(_property IN ITEMS LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
            get_target_property(_links "${target}" "${_property}")
            if(NOT _links)
                continue()
            endif()
            foreach(_link IN LISTS _links)
                if(_link MATCHES "^\\$<LINK_ONLY:([^<>]+)>$")
                    set(_link "${CMAKE_MATCH_1}")
                endif()
                if(TARGET "${_link}")
                    _metallic_shader_identity_link_inputs("${_link}" _visited _files _targets _complete)
                elseif(IS_ABSOLUTE "${_link}")
                    list(APPEND _files "${_link}")
                elseif(NOT _link STREQUAL "")
                    # Unsupported package expressions or raw linker switches
                    # retain ordinary host-specific identities, never a guessed
                    # complete shared compiler domain.
                    set(_complete FALSE)
                endif()
            endforeach()
        endforeach()
    elseif(NOT TARGET "${target}")
        set(_complete FALSE)
    endif()
    set(${visited} "${_visited}" PARENT_SCOPE)
    set(${files} "${_files}" PARENT_SCOPE)
    set(${targets} "${_targets}" PARENT_SCOPE)
    set(${complete} "${_complete}" PARENT_SCOPE)
endfunction()

function(_metallic_shader_identity_targets directory result)
    get_property(_targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    get_property(_children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(_child IN LISTS _children)
        _metallic_shader_identity_targets("${_child}" _child_targets)
        list(APPEND _targets ${_child_targets})
    endforeach()
    set(${result} "${_targets}" PARENT_SCOPE)
endfunction()

function(metallic_prepare_shader_compiler_identity)
    set(_directory "${CMAKE_BINARY_DIR}/generated/shader-compiler/$<CONFIG>")
    set(_provider "${_directory}/ShaderCompilerBuildIdentity.cpp")
    set_source_files_properties("${_provider}" PROPERTIES GENERATED TRUE)
    add_library(MetallicShaderCompilerBuildIdentity STATIC EXCLUDE_FROM_ALL "${_provider}")
    target_include_directories(MetallicShaderCompilerBuildIdentity PRIVATE "${CMAKE_SOURCE_DIR}/Source")
    add_library(MetallicShaderCompilerShared INTERFACE)
    target_link_libraries(MetallicShaderCompilerShared INTERFACE
        MetallicShaderWarmupCore MetallicShaderCompilerBuildIdentity)
    # Core never links Identity: the provider is generated from the completed
    # Core archive, excluding its own output and every host main/PDB.
    # Imported SDK targets are local to Source/. Resolve their graph in that
    # scope instead of promoting third-party targets to the global namespace.
    cmake_language(DEFER DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" CALL metallic_finalize_shader_compiler_identity)
endfunction()

function(_metallic_shader_identity_risk_content targets result)
    set(_content "")
    foreach(_target IN LISTS targets)
        if(_target STREQUAL "MetallicShaderCompilerBuildIdentity")
            continue()
        endif()
        # Property values can contain semicolons and empty cells. Keep them as
        # scalar strings; parsing delimiter-separated records loses boundaries.
        string(APPEND _content "string(APPEND identity_target_ipo [==[$<TARGET_PROPERTY:${_target},INTERPROCEDURAL_OPTIMIZATION>;$<TARGET_PROPERTY:${_target},INTERPROCEDURAL_OPTIMIZATION_DEBUG>;$<TARGET_PROPERTY:${_target},INTERPROCEDURAL_OPTIMIZATION_RELEASE>;$<TARGET_PROPERTY:${_target},INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO>;$<TARGET_PROPERTY:${_target},INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL>;]==])\n")
        string(APPEND _content "string(APPEND identity_target_compile [==[$<TARGET_GENEX_EVAL:${_target},$<TARGET_PROPERTY:${_target},COMPILE_OPTIONS>>;$<TARGET_GENEX_EVAL:${_target},$<TARGET_PROPERTY:${_target},INTERFACE_COMPILE_OPTIONS>>;]==])\n")
        string(APPEND _content "string(APPEND identity_target_link [==[$<TARGET_GENEX_EVAL:${_target},$<TARGET_PROPERTY:${_target},LINK_OPTIONS>>;$<TARGET_GENEX_EVAL:${_target},$<TARGET_PROPERTY:${_target},INTERFACE_LINK_OPTIONS>>;$<TARGET_PROPERTY:${_target},LINK_FLAGS>;$<TARGET_PROPERTY:${_target},LINK_FLAGS_DEBUG>;$<TARGET_PROPERTY:${_target},LINK_FLAGS_RELEASE>;$<TARGET_PROPERTY:${_target},LINK_FLAGS_RELWITHDEBINFO>;$<TARGET_PROPERTY:${_target},LINK_FLAGS_MINSIZEREL>;]==])\n")
    endforeach()
    set(${result} "${_content}" PARENT_SCOPE)
endfunction()

function(metallic_finalize_shader_compiler_identity_risks)
    set(_directory "${CMAKE_BINARY_DIR}/generated/shader-compiler/$<CONFIG>")
    # Run at the end of the root so tests and every host consumer exist too.
    _metallic_shader_identity_targets("${CMAKE_SOURCE_DIR}" _all_targets)
    _metallic_shader_identity_risk_content("${_all_targets}" _target_content)
    set(_risk_content "set(identity_global_link [==[${CMAKE_EXE_LINKER_FLAGS};${CMAKE_EXE_LINKER_FLAGS_DEBUG};${CMAKE_EXE_LINKER_FLAGS_RELEASE};${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO};${CMAKE_EXE_LINKER_FLAGS_MINSIZEREL};${CMAKE_SHARED_LINKER_FLAGS}]==])\nset(identity_global_ipo [==[${CMAKE_INTERPROCEDURAL_OPTIMIZATION};${CMAKE_INTERPROCEDURAL_OPTIMIZATION_DEBUG};${CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE};${CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO};${CMAKE_INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL}]==])\n")
    string(APPEND _risk_content "${_target_content}include([==[${_directory}/BackendRisks.cmake]==])\n")
    file(GENERATE OUTPUT "${_directory}/LinkRisks.cmake" CONTENT "${_risk_content}"
        CONDITION "$<COMPILE_LANGUAGE:CXX>" TARGET MetallicShaderWarmupCore)
endfunction()

function(metallic_finalize_shader_compiler_identity)
    set(_directory "${CMAKE_BINARY_DIR}/generated/shader-compiler/$<CONFIG>")
    set(_provider "${_directory}/ShaderCompilerBuildIdentity.cpp")
    set(_inputs "${_directory}/Inputs.cmake")
    set(_risks "${_directory}/LinkRisks.cmake")
    set(_script "${CMAKE_SOURCE_DIR}/cmake/GenerateShaderCompilerBuildIdentity.cmake")
    set(_visited)
    set(_backend_files)
    set(_backend_targets)
    set(_backend_complete TRUE)
    get_target_property(_roots MetallicShaderWarmupCore METALLIC_SHADER_IDENTITY_BACKEND_TARGETS)
    foreach(_target IN LISTS _roots)
        _metallic_shader_identity_link_inputs("${_target}" _visited _backend_files _backend_targets _backend_complete)
    endforeach()
    list(REMOVE_DUPLICATES _backend_files)
    list(REMOVE_DUPLICATES _backend_targets)
    set(_tools "${CMAKE_CXX_COMPILER}" "${CMAKE_AR}" "${CMAKE_LINKER}" "${CMAKE_COMMAND}"
        "${CMAKE_SOURCE_DIR}/cmake/RunMsvcCompiler.cmake")
    get_filename_component(_compiler_directory "${CMAKE_CXX_COMPILER}" DIRECTORY)
    if(MSVC)
        file(GLOB _compiler_components CONFIGURE_DEPENDS
            "${_compiler_directory}/c1*.dll" "${_compiler_directory}/c2*.dll" "${_compiler_directory}/mspdb*.dll")
        list(APPEND _tools ${_compiler_components})
    endif()
    set(_launcher_complete TRUE)
    if(METALLIC_USE_SCCACHE)
        list(APPEND _tools "${METALLIC_SCCACHE_EXECUTABLE}")
    endif()
    foreach(_launcher IN LISTS CMAKE_CXX_COMPILER_LAUNCHER)
        if(IS_ABSOLUTE "${_launcher}" AND EXISTS "${_launcher}")
            list(APPEND _tools "${_launcher}")
        elseif(NOT _launcher STREQUAL "-P" AND NOT _launcher STREQUAL "--" AND NOT _launcher STREQUAL "")
            set(_launcher_complete FALSE)
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _tools)
    file(GENERATE OUTPUT "${_inputs}" CONTENT
"set(identity_files [==[$<TARGET_FILE:MetallicShaderWarmupCore>;${_backend_files};${_tools}]==])
set(identity_semantics [==[shader-compiler-build-v1;compiler=${CMAKE_CXX_COMPILER_ID}/${CMAKE_CXX_COMPILER_VERSION}/${CMAKE_CXX_COMPILER_FRONTEND_VARIANT};config=$<CONFIG>;platform=${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/${CMAKE_SIZEOF_VOID_P};cxx=${CMAKE_CXX_STANDARD};runtime=$<TARGET_PROPERTY:MetallicShaderWarmupCore,MSVC_RUNTIME_LIBRARY>;global-flags=${CMAKE_CXX_FLAGS};debug=${CMAKE_CXX_FLAGS_DEBUG};release=${CMAKE_CXX_FLAGS_RELEASE};relwithdebinfo=${CMAKE_CXX_FLAGS_RELWITHDEBINFO};minsize=${CMAKE_CXX_FLAGS_MINSIZEREL};options=$<TARGET_GENEX_EVAL:MetallicShaderWarmupCore,$<TARGET_PROPERTY:MetallicShaderWarmupCore,COMPILE_OPTIONS>>;definitions=$<TARGET_GENEX_EVAL:MetallicShaderWarmupCore,$<TARGET_PROPERTY:MetallicShaderWarmupCore,COMPILE_DEFINITIONS>>;static-link=${CMAKE_STATIC_LINKER_FLAGS};static-debug=${CMAKE_STATIC_LINKER_FLAGS_DEBUG};static-release=${CMAKE_STATIC_LINKER_FLAGS_RELEASE};static-relwithdebinfo=${CMAKE_STATIC_LINKER_FLAGS_RELWITHDEBINFO};static-minsize=${CMAKE_STATIC_LINKER_FLAGS_MINSIZEREL};static-options=$<TARGET_GENEX_EVAL:MetallicShaderWarmupCore,$<TARGET_PROPERTY:MetallicShaderWarmupCore,STATIC_LIBRARY_OPTIONS>>;launcher=${CMAKE_CXX_COMPILER_LAUNCHER}]==])
set(identity_runtime [==[$<TARGET_GENEX_EVAL:MetallicShaderWarmupCore,$<TARGET_PROPERTY:MetallicShaderWarmupCore,MSVC_RUNTIME_LIBRARY>>]==])
set(identity_compiler [==[${CMAKE_CXX_COMPILER}]==])
set(identity_supported [==[${MSVC}/${CMAKE_GENERATOR}/${_backend_complete}/${_launcher_complete}]==])
" CONDITION "$<COMPILE_LANGUAGE:CXX>" TARGET MetallicShaderWarmupCore)

    _metallic_shader_identity_risk_content("${_backend_targets}" _backend_risk_content)
    file(GENERATE OUTPUT "${_directory}/BackendRisks.cmake" CONTENT "${_backend_risk_content}"
        CONDITION "$<COMPILE_LANGUAGE:CXX>" TARGET MetallicShaderWarmupCore)
    cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL metallic_finalize_shader_compiler_identity_risks)
    add_custom_command(OUTPUT "${_provider}"
        COMMAND "${CMAKE_COMMAND}" "-DIDENTITY_INPUTS=${_inputs}" "-DIDENTITY_RISKS=${_risks}"
            "-DIDENTITY_OUTPUT=${_provider}" "-DIDENTITY_DEPFILE=${_directory}/Identity.d" -P "${_script}"
        DEPENDS MetallicShaderWarmupCore "$<TARGET_FILE:MetallicShaderWarmupCore>"
            ${_backend_targets} ${_backend_files} ${_tools} "${_inputs}" "${_risks}"
            "${_directory}/BackendRisks.cmake" "${_script}"
        DEPFILE "${_directory}/Identity.d"
        COMMENT "Generating shared shader compiler build identity"
        VERBATIM)
    # Make the completed archive -> provider -> identity archive DAG explicit.
    add_custom_target(MetallicShaderCompilerIdentityGeneration DEPENDS "${_provider}")
    add_dependencies(MetallicShaderCompilerBuildIdentity MetallicShaderCompilerIdentityGeneration)
endfunction()
