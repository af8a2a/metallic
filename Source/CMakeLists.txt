set(METALLIC_SLANG_ROOT "${CMAKE_SOURCE_DIR}/External/slang")
if(DEFINED SLANG_ROOT)
    set(METALLIC_SLANG_ROOT "${SLANG_ROOT}")
endif()
find_package(slang CONFIG REQUIRED PATHS "${METALLIC_SLANG_ROOT}" NO_DEFAULT_PATH)

add_library(MetallicRuntimeRender STATIC
    Runtime/Render/HistoryResources.cpp
    Runtime/Render/RenderSample.cpp
    Runtime/Render/SlangCompiler.cpp
    Runtime/Render/RenderGraph/RenderGraph.cpp
    Runtime/Render/RenderGraph/RenderGraphNode.cpp
    Runtime/Render/RenderGraph/RenderGraphExecutor.cpp
    Runtime/Render/RenderPass/BuiltinRenderPasses.cpp
    Runtime/Render/RenderPass/BuiltinPass/BunnyWireframePass.cpp
    Runtime/Render/RenderPass/BuiltinPass/ClearColorPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/CopyColorPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/ImageSamplePass.cpp
    Runtime/Render/RenderPass/BuiltinPass/NrdDenoisePass.cpp
    Runtime/Render/RenderPass/BuiltinPass/RenderGraphBufferPasses.cpp
    Runtime/Render/RenderPass/BuiltinPass/SceneMaterialShaderObjectPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/SceneMaterialVisualizationPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp
    Runtime/Render/RenderPass/BuiltinPass/SceneRayQueryVisualizationPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/TriangleRasterPass.cpp
    Runtime/Render/RenderPass/ScenePathTraceResources.cpp
    Runtime/Render/GAPI/Vulkan/VulkanRhi.cpp
    Runtime/Render/GAPI/Vulkan/VulkanNrdWrapper.cpp
    Runtime/Render/GAPI/Vulkan/VulkanSceneRtx.cpp
)

target_include_directories(MetallicRuntimeRender PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(MetallicRuntimeRender SYSTEM PRIVATE
    "${CMAKE_SOURCE_DIR}/External/stb"
    "${CMAKE_SOURCE_DIR}/External/VulkanMemoryAllocator/include"
)

target_link_libraries(MetallicRuntimeRender PUBLIC
    metallic::volk
    metallic::sdl3
    metallic::nsight_events
    nlohmann_json
    slang::slang
)

if(TARGET NRD)
    target_link_libraries(MetallicRuntimeRender PUBLIC NRD)
    target_compile_definitions(MetallicRuntimeRender PUBLIC METALLIC_HAS_NRD=1)
else()
    target_compile_definitions(MetallicRuntimeRender PUBLIC METALLIC_HAS_NRD=0)
endif()

target_compile_definitions(MetallicRuntimeRender PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

if(WIN32)
    target_compile_definitions(MetallicRuntimeRender PRIVATE NOMINMAX)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(MetallicRuntimeRender PRIVATE -Wno-nullability-completeness)
endif()

add_library(MetallicRuntimeScene STATIC
    Runtime/Scene/Scene.cpp
    Runtime/Scene/TinyGltfImpl.cpp
)

target_include_directories(MetallicRuntimeScene PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(MetallicRuntimeScene SYSTEM PRIVATE
    "${CMAKE_SOURCE_DIR}/External/stb"
)

target_link_libraries(MetallicRuntimeScene PUBLIC
    MathLib
    nlohmann_json
    tinygltf
)

if(WIN32)
    target_compile_definitions(MetallicRuntimeScene PRIVATE NOMINMAX)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU" AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    target_compile_options(MetallicRuntimeScene PUBLIC -msse3)
endif()

target_link_libraries(MetallicRuntimeRender PUBLIC
    MetallicRuntimeScene
)

add_executable(Metallic
    Main.cpp
    Editor/EditorApplication.cpp
)

target_include_directories(Metallic PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(Metallic PRIVATE
    imgui
    imnodes
    metallic::sdl3
    MetallicRuntimeRender
    MetallicRuntimeScene
)

target_compile_definitions(Metallic PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

if(WIN32)
    target_compile_definitions(Metallic PRIVATE NOMINMAX)

    file(GLOB METALLIC_SLANG_RUNTIME_DLLS "${METALLIC_SLANG_ROOT}/bin/*.dll")
    foreach(METALLIC_SLANG_RUNTIME_DLL IN LISTS METALLIC_SLANG_RUNTIME_DLLS)
        add_custom_command(TARGET Metallic POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${METALLIC_SLANG_RUNTIME_DLL}"
                "$<TARGET_FILE_DIR:Metallic>"
        )
    endforeach()

    if(TARGET SDL3-shared)
        add_custom_command(TARGET Metallic POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SDL3-shared>"
                "$<TARGET_FILE_DIR:Metallic>"
        )
    elseif(TARGET SDL3::SDL3-shared)
        add_custom_command(TARGET Metallic POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SDL3::SDL3-shared>"
                "$<TARGET_FILE_DIR:Metallic>"
        )
    endif()
endif()

add_executable(MetallicMaterialVisualizationSample
    Samples/MaterialVisualizationSampleMain.cpp
    Editor/EditorApplication.cpp
)

target_include_directories(MetallicMaterialVisualizationSample PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(MetallicMaterialVisualizationSample PRIVATE
    imgui
    imnodes
    metallic::sdl3
    MetallicRuntimeRender
    MetallicRuntimeScene
)

target_compile_definitions(MetallicMaterialVisualizationSample PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

if(WIN32)
    target_compile_definitions(MetallicMaterialVisualizationSample PRIVATE NOMINMAX)

    foreach(METALLIC_SLANG_RUNTIME_DLL IN LISTS METALLIC_SLANG_RUNTIME_DLLS)
        add_custom_command(TARGET MetallicMaterialVisualizationSample POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${METALLIC_SLANG_RUNTIME_DLL}"
                "$<TARGET_FILE_DIR:MetallicMaterialVisualizationSample>"
        )
    endforeach()

    if(TARGET SDL3-shared)
        add_custom_command(TARGET MetallicMaterialVisualizationSample POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SDL3-shared>"
                "$<TARGET_FILE_DIR:MetallicMaterialVisualizationSample>"
        )
    elseif(TARGET SDL3::SDL3-shared)
        add_custom_command(TARGET MetallicMaterialVisualizationSample POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SDL3::SDL3-shared>"
                "$<TARGET_FILE_DIR:MetallicMaterialVisualizationSample>"
        )
    endif()
endif()
