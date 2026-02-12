# Pipeline Editor - Standalone tool for editing render pipeline configurations

add_executable(PipelineEditorTool
    main.cpp
    pipeline_editor.cpp
)

target_include_directories(PipelineEditorTool PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/Source/PipelineEditor
)

# Find OpenGL for the standalone editor
find_package(OpenGL REQUIRED)

target_link_libraries(PipelineEditorTool PRIVATE
    PipelineEditorLib
    imgui_opengl
    imnodes
    glfw
    OpenGL::GL
    spdlog::spdlog_header_only
)

# Copy Pipelines directory to build output
add_custom_command(TARGET PipelineEditorTool POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/Pipelines"
        "$<TARGET_FILE_DIR:PipelineEditorTool>/Pipelines"
)
