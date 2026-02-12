# PipelineEditorLib - Core library for pipeline asset loading/saving
# Used by both Metallic renderer and the standalone Pipeline Editor tool

add_library(PipelineEditorLib STATIC
    pass_registry.cpp
    pipeline_asset.cpp
    pipeline_builder.cpp
)

target_include_directories(PipelineEditorLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Need access to Rendering headers for frame_graph.h, render_pass.h
target_include_directories(PipelineEditorLib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../Rendering
    ${CMAKE_CURRENT_SOURCE_DIR}/../Asset
    ${CMAKE_CURRENT_SOURCE_DIR}/../Scene
    ${CMAKE_CURRENT_SOURCE_DIR}/../Platform
)

target_link_libraries(PipelineEditorLib PUBLIC
    nlohmann_json
    spdlog::spdlog_header_only
    Tracy::TracyClient
)

# Need Metal for pipeline_builder (MTL::PixelFormat)
target_link_libraries(PipelineEditorLib PRIVATE
    metal-cpp
    MathLib
)
