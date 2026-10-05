if(NOT DEFINED SOURCE_DIRECTORY)
    message(FATAL_ERROR "SOURCE_DIRECTORY is required")
endif()
file(GLOB_RECURSE shader_clients
    "${SOURCE_DIRECTORY}/Source/*.cpp" "${SOURCE_DIRECTORY}/Source/*.h"
    "${SOURCE_DIRECTORY}/Tools/*.cpp" "${SOURCE_DIRECTORY}/Tools/*.h")
foreach(path IN LISTS shader_clients)
    file(RELATIVE_PATH relative "${SOURCE_DIRECTORY}" "${path}")
    if(relative MATCHES "^Source/Runtime/Render/GAPI/" OR
       relative MATCHES "^Source/Runtime/Render/Core/(SlangCompiler\\.(cpp|h)|ShaderRegistry(Compiler)?\\.cpp|ShaderRegistry\\.h)$")
        continue()
    endif()
    file(READ "${path}" source)
    # Ignore logs/comments; reject actual compiler and device factory calls.
    string(REPLACE "\\\\" "" source "${source}")
    string(REPLACE "\\\"" "" source "${source}")
    string(REGEX REPLACE "\"[^\"]*\"" "\"\"" source "${source}")
    string(REGEX REPLACE "//[^\n]*|/\\*([^*]|\\*+[^*/])*\\*+/" "" source "${source}")
    if(source MATCHES "compileSlangShaderToSpirv[ \r\n\t]*\\(" OR
       source MATCHES "(->|\\.)[ \r\n\t]*create(ShaderModule|ComputePipeline|GraphicsPipeline|GraphicsShaderObjectProgram|PipelineCache)[ \r\n\t]*\\(" OR
       source MATCHES "vkCreate(ShaderModule|ComputePipelines|GraphicsPipelines)[ \r\n\t]*\\(")
        message(FATAL_ERROR "${relative} bypasses ShaderRegistry (${CMAKE_MATCH_0}); acquire shaders/pipelines through its unified API")
    endif()
endforeach()
message(STATUS "ShaderRegistry usage audit passed")
