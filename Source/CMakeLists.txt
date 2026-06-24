set(METALLIC_SLANG_ROOT "${CMAKE_SOURCE_DIR}/External/slang")
if(DEFINED SLANG_ROOT)
    set(METALLIC_SLANG_ROOT "${SLANG_ROOT}")
endif()
find_package(slang CONFIG REQUIRED PATHS "${METALLIC_SLANG_ROOT}" NO_DEFAULT_PATH)

add_library(MetallicRuntimeRender STATIC
    Runtime/Render/HistoryResources.cpp
    Runtime/Render/MeshletStreamResidency.cpp
    Runtime/Render/GAPI/Streamer.cpp
    Runtime/Render/RenderSample.cpp
    Runtime/Render/SlangCompiler.cpp
    Runtime/Render/RenderGraph/RenderGraph.cpp
    Runtime/Render/RenderGraph/RenderGraphNode.cpp
    Runtime/Render/RenderGraph/RenderGraphExecutor.cpp
    Runtime/Render/RenderGraph/RenderGraphStreamingSubsystem.cpp
    Runtime/Render/RenderPass/BuiltinRenderPasses.cpp
    Runtime/Render/RenderPass/BuiltinPass/BunnyWireframePass.cpp
    Runtime/Render/RenderPass/BuiltinPass/ClearColorPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/CopyColorPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/GPUDrivenPreviewPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/GPUDrivenStreamAssetPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/ImageSamplePass.cpp
    Runtime/Render/RenderPass/BuiltinPass/NrdDenoisePass.cpp
    Runtime/Render/RenderPass/BuiltinPass/RenderGraphBufferPasses.cpp
    Runtime/Render/RenderPass/BuiltinPass/SceneMaterialShaderObjectPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/SceneMaterialVisualizationPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp
    Runtime/Render/RenderPass/BuiltinPass/SceneRayQueryVisualizationPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/StreamlineDlssRrPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/TriangleRasterPass.cpp
    Runtime/Render/RenderPass/ScenePathTraceResources.cpp
    Runtime/Render/Profiling/NsightAftermath.cpp
    Runtime/Render/GAPI/Vulkan/VulkanRhi.cpp
    Runtime/Render/GAPI/Vulkan/VulkanNrdWrapper.cpp
    Runtime/Render/GAPI/Vulkan/VulkanSceneRtx.cpp
    Runtime/Render/GAPI/Vulkan/VulkanStreamline.cpp
)

target_include_directories(MetallicRuntimeRender PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(MetallicRuntimeRender SYSTEM PRIVATE
    "${CMAKE_SOURCE_DIR}/External/openpbr-bsdf"
    "${CMAKE_SOURCE_DIR}/External/stb"
    "${CMAKE_SOURCE_DIR}/External/VulkanMemoryAllocator/include"
)

target_link_libraries(MetallicRuntimeRender PUBLIC
    metallic::volk
    metallic::sdl3
    metallic::aftermath
    metallic::nsight_events
    metallic::streamline
    nlohmann_json
    slang::slang
    spdlog::spdlog
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
    Runtime/Scene/MeshletStreamAsset.cpp
    Runtime/Scene/Scene.cpp
    Runtime/Scene/TinyGltfImpl.cpp
)

target_include_directories(MetallicRuntimeScene PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(MetallicRuntimeScene SYSTEM PRIVATE
    "${CMAKE_SOURCE_DIR}/External/meshoptimizer/demo"
    "${CMAKE_SOURCE_DIR}/External/stb"
)

target_link_libraries(MetallicRuntimeScene PUBLIC
    MathLib
    meshoptimizer
    nlohmann_json
    spdlog::spdlog
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

set(METALLIC_EDITOR_SOURCES
    Editor/EditorApplication.cpp
    Editor/EditorProfiler.cpp
    Editor/NvmlMonitor.cpp
)

add_executable(Metallic
    Main.cpp
    ${METALLIC_EDITOR_SOURCES}
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
    target_link_libraries(Metallic PRIVATE Comdlg32)

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

    metallic_copy_streamline_runtime(Metallic)
    metallic_copy_aftermath_runtime(Metallic)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

function(configure_metallic_editor_sample target_name)
    target_include_directories(${target_name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    target_link_libraries(${target_name} PRIVATE
        imgui
        imnodes
        metallic::sdl3
        MetallicRuntimeRender
        MetallicRuntimeScene
    )

    target_compile_definitions(${target_name} PRIVATE
        PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    )

    set_target_properties(${target_name} PROPERTIES FOLDER "Samples")

    if(WIN32)
        target_compile_definitions(${target_name} PRIVATE NOMINMAX)
        target_link_libraries(${target_name} PRIVATE Comdlg32)

        foreach(METALLIC_SLANG_RUNTIME_DLL IN LISTS METALLIC_SLANG_RUNTIME_DLLS)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${METALLIC_SLANG_RUNTIME_DLL}"
                    "$<TARGET_FILE_DIR:${target_name}>"
            )
        endforeach()

        if(TARGET SDL3-shared)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:SDL3-shared>"
                    "$<TARGET_FILE_DIR:${target_name}>"
            )
        elseif(TARGET SDL3::SDL3-shared)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "$<TARGET_FILE:SDL3::SDL3-shared>"
                    "$<TARGET_FILE_DIR:${target_name}>"
            )
        endif()

        metallic_copy_streamline_runtime(${target_name})
        metallic_copy_aftermath_runtime(${target_name})
    endif()
endfunction()

add_executable(MetallicMaterialVisualizationSample
    Samples/MaterialVisualizationSampleMain.cpp
    ${METALLIC_EDITOR_SOURCES}
)
configure_metallic_editor_sample(MetallicMaterialVisualizationSample)

add_executable(MetallicPathTracingSample
    Samples/PathTracingSampleMain.cpp
    ${METALLIC_EDITOR_SOURCES}
)
configure_metallic_editor_sample(MetallicPathTracingSample)

add_executable(MetallicGPUDrivenSample
    Samples/GPUDrivenSampleMain.cpp
    ${METALLIC_EDITOR_SOURCES}
)
configure_metallic_editor_sample(MetallicGPUDrivenSample)
