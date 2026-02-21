# Pipeline Editor - Standalone tool for editing render pipeline configurations

add_executable(PipelineEditorTool
    main.cpp
    pipeline_editor.cpp
    pass_registrations_editor.cpp
)

target_include_directories(PipelineEditorTool PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/Source/PipelineEditor
)

# Find OpenGL for the standalone editor
find_package(OpenGL REQUIRED)

# Embed project root path so the editor reads/writes source tree pipelines
target_compile_definitions(PipelineEditorTool PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

target_link_libraries(PipelineEditorTool PRIVATE
    PipelineEditorLib
    imgui_opengl
    imnodes
    glfw
    OpenGL::GL
    spdlog::spdlog_header_only
)

