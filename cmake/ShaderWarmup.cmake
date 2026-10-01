add_library(MetallicShaderWarmupCore STATIC EXCLUDE_FROM_ALL
    "${CMAKE_SOURCE_DIR}/Tools/ShaderWarmup.cpp"
    "${CMAKE_SOURCE_DIR}/Source/Runtime/Render/Core/SlangCompiler.cpp"
)
target_include_directories(MetallicShaderWarmupCore PUBLIC "${CMAKE_SOURCE_DIR}/Source")
find_package(Threads REQUIRED)
target_link_libraries(MetallicShaderWarmupCore PUBLIC slang::slang spdlog::spdlog Threads::Threads)
target_compile_definitions(MetallicShaderWarmupCore PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    METALLIC_RTXCR_SHADER_INCLUDE_DIR="${METALLIC_RTXCR_SHADER_INCLUDE_DIR}"
)
if(MSVC)
    target_compile_options(MetallicShaderWarmupCore PRIVATE /utf-8 /EHsc)
    target_compile_definitions(MetallicShaderWarmupCore PRIVATE NOMINMAX)
endif()
# Applications call the shared implementation at startup. Building an
# application does not execute the separate manual warmup target.
add_executable(MetallicShaderCompiler EXCLUDE_FROM_ALL
    "${CMAKE_SOURCE_DIR}/Tools/ShaderWarmupMain.cpp")
target_link_libraries(MetallicShaderCompiler PRIVATE MetallicShaderWarmupCore)
metallic_copy_spdlog_runtime(MetallicShaderCompiler)
if(WIN32)
    file(GLOB _metallic_warmup_slang_dlls "${METALLIC_SLANG_ROOT}/bin/*.dll")
    add_custom_target(MetallicShaderCompilerRuntime
        COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:MetallicShaderCompiler>"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            ${_metallic_warmup_slang_dlls} "$<TARGET_FILE_DIR:MetallicShaderCompiler>"
        VERBATIM
    )
    add_dependencies(MetallicShaderCompiler MetallicShaderCompilerRuntime)
endif()

set(METALLIC_SHADER_WARMUP_ARGS "" CACHE STRING
    "Optional shader warmup arguments (semicolon-separated), e.g. --filter;FinalBlit")
add_custom_target(MetallicShaderWarmup
    COMMAND "$<TARGET_FILE:MetallicShaderCompiler>" ${METALLIC_SHADER_WARMUP_ARGS}
    DEPENDS MetallicShaderCompiler
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    COMMENT "Warming the optional Metallic SPIR-V cache"
    USES_TERMINAL
    VERBATIM
)
