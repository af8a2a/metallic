# Pipeline Editor - Standalone tool for editing render pipeline configurations

add_executable(PipelineEditorTool
    main.cpp
    pipeline_editor.cpp
    pass_registrations_editor.cpp
)

target_include_directories(PipelineEditorTool PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/Source/PipelineEditor
    ${METALLIC_GENERATED_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/Source/RHI
    ${CMAKE_SOURCE_DIR}/Source/RHI/Helpers
    ${CMAKE_SOURCE_DIR}/Source/Helpers
    ${CMAKE_SOURCE_DIR}/Source/Asset
    ${CMAKE_SOURCE_DIR}/Source/Scene
    ${CMAKE_SOURCE_DIR}/Source/Rendering
    ${CMAKE_SOURCE_DIR}/Source/Rendering/Helpers
    ${CMAKE_SOURCE_DIR}/Source/Rendering/Passes
    ${CMAKE_SOURCE_DIR}/Source/Platform
    ${CMAKE_SOURCE_DIR}/External/MathLib
    ${CMAKE_SOURCE_DIR}/External/meshoptimizer/src
    ${CMAKE_SOURCE_DIR}/External/stb
)

target_compile_definitions(PipelineEditorTool PRIVATE
    METALLIC_PASS_REGISTRATION_METADATA_ONLY
)

if(WIN32)
    find_package(Vulkan REQUIRED)
    target_include_directories(PipelineEditorTool PRIVATE
        ${CMAKE_SOURCE_DIR}/Source/RHI/Vulkan
        ${CMAKE_SOURCE_DIR}/Source/RHI/Vulkan/Helpers
        ${CMAKE_SOURCE_DIR}/External/VulkanMemoryAllocator/include
    )
    target_link_libraries(PipelineEditorTool PRIVATE Vulkan::Vulkan)
endif()

# Find OpenGL for the standalone editor
find_package(OpenGL REQUIRED)

# Embed project root path so the editor reads/writes source tree pipelines
target_compile_definitions(PipelineEditorTool PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

target_link_libraries(PipelineEditorTool PRIVATE
    PipelineEditorLib
    MathLib
    imgui_opengl
    imnodes
    glfw
    OpenGL::GL
    spdlog::spdlog_header_only
)
