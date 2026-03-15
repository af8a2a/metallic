if(APPLE)
    add_subdirectory(PipelineEditor)

    add_executable(Metallic
        main.cpp
        Platform/metal_impl.cpp
        Platform/metal_raytracing_utils.cpp
        Platform/metal_resource_utils.cpp
        Platform/metal_shader_utils.cpp
        Platform/metal_runtime.mm
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
        Rendering/scene_context.cpp
        RHI/rhi_resource_utils.cpp
        RHI/rhi_shader_utils.cpp
        RHI/rhi_raytracing_utils.cpp
        RHI/Metal/metal_window_runtime.cpp
        RHI/Metal/metal_frame_graph.cpp
    )
    target_include_directories(Metallic PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/RHI"
        "${CMAKE_CURRENT_SOURCE_DIR}/RHI/Metal"
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

    set_source_files_properties(Platform/tracy_metal.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
elseif(WIN32)
    find_package(Vulkan REQUIRED)

    add_executable(Metallic
        main_vulkan.cpp
        Rendering/slang_compiler.cpp
        Rendering/shader_manager.cpp
        RHI/rhi_backend.cpp
        RHI/rhi_resource_utils.cpp
        RHI/rhi_shader_utils.cpp
        RHI/Vulkan/vulkan_backend.cpp
        RHI/Vulkan/vulkan_frame_graph.cpp
        RHI/Vulkan/vulkan_descriptor_manager.cpp
    )
    target_include_directories(Metallic PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/RHI"
        "${CMAKE_CURRENT_SOURCE_DIR}/RHI/Vulkan"
        "${CMAKE_CURRENT_SOURCE_DIR}/Rendering"
        "${CMAKE_CURRENT_SOURCE_DIR}/Platform"
        "${CMAKE_SOURCE_DIR}/External/MathLib"
        "${CMAKE_SOURCE_DIR}/External/VulkanMemoryAllocator/include"
    )
    target_link_libraries(Metallic PRIVATE
        glfw
        imgui
        slang::slang
        spdlog::spdlog_header_only
        Vulkan::Vulkan
    )
endif()

target_compile_definitions(Metallic PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

foreach(SLANG_RUNTIME_LIB ${METALLIC_SLANG_RUNTIME_LIBS})
    add_custom_command(TARGET Metallic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SLANG_RUNTIME_LIB}"
            "$<TARGET_FILE_DIR:Metallic>"
    )
endforeach()

add_custom_command(TARGET Metallic POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/Shaders"
        "$<TARGET_FILE_DIR:Metallic>/Shaders"
)

if(APPLE)
    add_custom_command(TARGET Metallic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "$<TARGET_FILE_DIR:Metallic>/Source/Rendering"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_SOURCE_DIR}/Source/Rendering/visibility_constants.h"
            "$<TARGET_FILE_DIR:Metallic>/Source/Rendering/visibility_constants.h"
    )

    add_custom_command(TARGET Metallic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/Asset"
            "$<TARGET_FILE_DIR:Metallic>/Asset"
    )

    add_custom_command(TARGET Metallic POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/Pipelines"
            "$<TARGET_FILE_DIR:Metallic>/Pipelines"
    )
endif()
