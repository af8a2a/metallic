# PipelineEditor library
add_subdirectory(PipelineEditor)

add_executable(Metallic
    main.cpp
    Platform/metal_impl.cpp
    Platform/glfw_metal_bridge.mm
    Platform/imgui_metal_bridge.mm
    Platform/tracy_metal.mm
    Asset/mesh_loader.cpp
    Asset/meshlet_builder.cpp
    Asset/material_loader.cpp
    Scene/scene_graph.cpp
    Scene/scene_graph_ui.cpp
    Rendering/input.cpp
    Rendering/frame_graph.cpp
    Rendering/raytraced_shadows.cpp
    Rendering/render_pass.cpp
    Rendering/pass_registrations.cpp
    Rendering/slang_compiler.cpp
    Rendering/shader_manager.cpp
)
target_include_directories(Metallic PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/Platform"
    "${CMAKE_CURRENT_SOURCE_DIR}/Asset"
    "${CMAKE_CURRENT_SOURCE_DIR}/Scene"
    "${CMAKE_CURRENT_SOURCE_DIR}/Rendering"
    "${CMAKE_CURRENT_SOURCE_DIR}/Rendering/Passes"
    "${CMAKE_SOURCE_DIR}/External/cgltf"
    "${CMAKE_SOURCE_DIR}/External/stb"
)
target_link_libraries(Metallic PRIVATE
    metal-cpp
    glfw
    meshoptimizer
    imgui
    slang::slang
    MathLib
    Tracy::TracyClient
    spdlog::spdlog_header_only
    PipelineEditorLib
    "-framework AppKit"
)

# Embed project root path so the app can load shaders from the source tree
target_compile_definitions(Metallic PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# TracyMetal requires ARC (Automatic Reference Counting)
set_source_files_properties(Platform/tracy_metal.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

# Copy Slang dylibs next to executable at build time
set(SLANG_LIB_DIR "${CMAKE_SOURCE_DIR}/External/slang/lib")
add_custom_command(TARGET Metallic POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SLANG_LIB_DIR}/libslang-compiler.dylib"
        "$<TARGET_FILE_DIR:Metallic>/libslang-compiler.dylib"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SLANG_LIB_DIR}/libslang-rt.dylib"
        "$<TARGET_FILE_DIR:Metallic>/libslang-rt.dylib"
)

# Copy shader files next to executable
add_custom_command(TARGET Metallic POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/Shaders"
        "$<TARGET_FILE_DIR:Metallic>/Shaders"
)

# Copy shared visibility constants header for runtime Slang includes
add_custom_command(TARGET Metallic POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "$<TARGET_FILE_DIR:Metallic>/Source/Rendering"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/Source/Rendering/visibility_constants.h"
        "$<TARGET_FILE_DIR:Metallic>/Source/Rendering/visibility_constants.h"
)

# Copy asset files next to executable
add_custom_command(TARGET Metallic POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/Asset"
        "$<TARGET_FILE_DIR:Metallic>/Asset"
)

# Copy pipeline JSON files next to executable
add_custom_command(TARGET Metallic POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/Pipelines"
        "$<TARGET_FILE_DIR:Metallic>/Pipelines"
)
