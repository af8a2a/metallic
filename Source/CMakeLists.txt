set(METALLIC_SLANG_ROOT "${CMAKE_SOURCE_DIR}/External/slang")
if(DEFINED SLANG_ROOT)
    set(METALLIC_SLANG_ROOT "${SLANG_ROOT}")
endif()
find_package(slang CONFIG REQUIRED PATHS "${METALLIC_SLANG_ROOT}" NO_DEFAULT_PATH)

add_library(MetallicRuntimeTask STATIC
    Runtime/Task/TaskSystem.cpp
)

target_include_directories(MetallicRuntimeTask PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(MetallicRuntimeTask PUBLIC
    STDEXEC::stdexec
)

if(WIN32)
    target_compile_definitions(MetallicRuntimeTask PRIVATE NOMINMAX)
    target_compile_options(MetallicRuntimeTask PRIVATE /EHsc)
endif()

add_library(MetallicRuntimeRender STATIC
    Runtime/Render/ComputeProgram.cpp
    Runtime/Render/HistoryResources.cpp
    Runtime/Render/ImportanceSampling.cpp
    Runtime/Render/ReGIR.cpp
    Runtime/Render/SceneResourceManager.cpp
    Runtime/Render/Subsystem/BuiltinRenderSubsystems.cpp
    Runtime/Render/Subsystem/EnvironmentLightingSubsystem.cpp
    Runtime/Render/Subsystem/GPUScene.cpp
    Runtime/Render/Subsystem/GPUSceneSubsystem.cpp
    Runtime/Render/Subsystem/RenderSubsystem.cpp
    Runtime/Render/Subsystem/RenderWorld.cpp
    Runtime/Render/MeshletStreamClas.cpp
    Runtime/Render/MeshletStreamClasPool.cpp
    Runtime/Render/MeshletStreamPageLoader.cpp
    Runtime/Render/MeshletStreamResidency.cpp
    Runtime/Render/MeshletStreamRuntime.cpp
    Runtime/Render/NeuralTextureResources.cpp
    Runtime/Render/RayTracing/SceneAccelerationStructure.cpp
    Runtime/Render/GAPI/PipelineCacheFile.cpp
    Runtime/Render/GAPI/PipelineStateHash.cpp
    Runtime/Render/GAPI/Streamer.cpp
    Runtime/Render/RenderSample.cpp
    Runtime/Render/SlangCompiler.cpp
    Runtime/Render/RenderGraph/RenderGraph.cpp
    Runtime/Render/RenderGraph/RenderGraphNode.cpp
    Runtime/Render/RenderGraph/RenderGraphExecutor.cpp
    Runtime/Render/RenderGraph/RenderGraphStreamingSubsystem.cpp
    Runtime/Render/RenderGraph/NrdRuntime.cpp
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
    Runtime/Render/RenderPass/BuiltinPass/RtxdiConfidencePass.cpp
    Runtime/Render/RenderPass/BuiltinPass/RtxdiCompositePass.cpp
    Runtime/Render/RenderPass/BuiltinPass/RtxcrMaterialSamplePass.cpp
    Runtime/Render/RenderPass/BuiltinPass/SceneRtxdiPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/SceneRayQueryVisualizationPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/StreamlineDlssRrPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/TriangleRasterPass.cpp
    Runtime/Render/RenderPass/ScenePathTraceResources.cpp
    Runtime/Render/Profiling/NsightAftermath.cpp
    Runtime/Render/Profiling/NsightGraphicsCapture.cpp
    Runtime/Render/GAPI/Vulkan/VulkanRhi.cpp
    Runtime/Render/GAPI/Vulkan/VulkanNrcWrapper.cpp
    Runtime/Render/RayTracing/SceneAccelerationStructureExtensions.cpp
    Runtime/Render/GAPI/Vulkan/VulkanStreamline.cpp
)

target_include_directories(MetallicRuntimeRender PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(MetallicRuntimeRender BEFORE PRIVATE
    "${METALLIC_SLANG_ROOT}/include"
)
set_property(TARGET MetallicRuntimeRender PROPERTY NO_SYSTEM_FROM_IMPORTED ON)

target_include_directories(MetallicRuntimeRender SYSTEM PRIVATE
    "${CMAKE_SOURCE_DIR}/External/openpbr-bsdf"
    "${CMAKE_SOURCE_DIR}/External/stb"
    "${CMAKE_SOURCE_DIR}/External/VulkanMemoryAllocator/include"
)

target_link_libraries(MetallicRuntimeRender PUBLIC
    MetallicRuntimeTask
    metallic::volk
    metallic::sdl3
    metallic::aftermath
    metallic::nsight_events
    metallic::streamline
    metallic::nrc
    nlohmann_json
    slang::slang
    spdlog::spdlog
)

target_link_libraries(MetallicRuntimeRender PRIVATE
    metallic::nsight_graphics_capture
)

if(TARGET libntc)
    target_link_libraries(MetallicRuntimeRender PRIVATE libntc)
    target_compile_definitions(MetallicRuntimeRender PRIVATE
        METALLIC_HAS_NTC=1
        METALLIC_NTC_SHADER_INCLUDE_DIR="${METALLIC_NTC_SHADER_INCLUDE_DIR}"
    )
else()
    target_compile_definitions(MetallicRuntimeRender PRIVATE
        METALLIC_HAS_NTC=0
        METALLIC_NTC_SHADER_INCLUDE_DIR=""
    )
endif()

if(TARGET NRD)
    target_link_libraries(MetallicRuntimeRender PUBLIC NRD)
    target_compile_definitions(MetallicRuntimeRender PUBLIC METALLIC_HAS_NRD=1)
else()
    target_compile_definitions(MetallicRuntimeRender PUBLIC METALLIC_HAS_NRD=0)
endif()

target_compile_definitions(MetallicRuntimeRender PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

if(METALLIC_RTXCR_MATERIAL_ROOT)
    target_compile_definitions(MetallicRuntimeRender PUBLIC
        METALLIC_HAS_RTXCR=1
        METALLIC_RTXCR_SHADER_INCLUDE_DIR="${METALLIC_RTXCR_SHADER_INCLUDE_DIR}"
    )
else()
    target_compile_definitions(MetallicRuntimeRender PUBLIC
        METALLIC_HAS_RTXCR=0
        METALLIC_RTXCR_SHADER_INCLUDE_DIR=""
    )
endif()

if(METALLIC_RTXCR_GEOMETRY_ROOT)
    target_compile_definitions(MetallicRuntimeRender PUBLIC
        METALLIC_HAS_RTXCR_GEOMETRY=1
        METALLIC_RTXCR_GEOMETRY_SHADER_INCLUDE_DIR="${METALLIC_RTXCR_GEOMETRY_SHADER_INCLUDE_DIR}"
    )
else()
    target_compile_definitions(MetallicRuntimeRender PUBLIC
        METALLIC_HAS_RTXCR_GEOMETRY=0
        METALLIC_RTXCR_GEOMETRY_SHADER_INCLUDE_DIR=""
    )
endif()

if(METALLIC_RTXCR_ASSETS_ROOT)
    target_compile_definitions(MetallicRuntimeRender PUBLIC
        METALLIC_HAS_RTXCR_ASSETS=1
        METALLIC_RTXCR_ASSETS_ROOT="${METALLIC_RTXCR_ASSETS_ROOT}"
    )
else()
    target_compile_definitions(MetallicRuntimeRender PUBLIC
        METALLIC_HAS_RTXCR_ASSETS=0
        METALLIC_RTXCR_ASSETS_ROOT=""
    )
endif()

if(WIN32)
    target_compile_definitions(MetallicRuntimeRender PRIVATE NOMINMAX)
    target_compile_options(MetallicRuntimeRender PRIVATE /EHsc)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(MetallicRuntimeRender PRIVATE -Wno-nullability-completeness)
endif()

add_library(MetallicRuntimeScene STATIC
    Runtime/Scene/MeshletStreamAsset.cpp
    Runtime/Scene/Scene.cpp
    Runtime/Scene/SceneDocument.cpp
    Runtime/Scene/SceneGraph.cpp
    Runtime/Scene/SceneLoader.cpp
    Runtime/Scene/ScenePicker.cpp
    Runtime/Scene/TinyGltfImpl.cpp
)

target_include_directories(MetallicRuntimeScene PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_include_directories(MetallicRuntimeScene SYSTEM PRIVATE
    "${CMAKE_SOURCE_DIR}/External/meshoptimizer/demo"
    "${CMAKE_SOURCE_DIR}/External/stb"
)

if(METALLIC_RTXCR_GEOMETRY_ROOT)
    target_include_directories(MetallicRuntimeScene SYSTEM PRIVATE
        "${METALLIC_RTXCR_GEOMETRY_ROOT}/include"
    )
    target_compile_definitions(MetallicRuntimeScene PUBLIC
        METALLIC_HAS_RTXCR_GEOMETRY=1
    )
else()
    target_compile_definitions(MetallicRuntimeScene PUBLIC
        METALLIC_HAS_RTXCR_GEOMETRY=0
    )
endif()

target_compile_definitions(MetallicRuntimeScene PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    METALLIC_RTXCR_ASSETS_ROOT="${METALLIC_RTXCR_ASSETS_ROOT}"
)

target_link_libraries(MetallicRuntimeScene PUBLIC
    EnTT::EnTT
    MetallicRuntimeTask
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
    imguizmo::imguizmo
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
    target_compile_options(Metallic PRIVATE /EHsc)
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
    metallic_copy_nrc_runtime(Metallic)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

function(configure_metallic_editor_sample target_name)
    target_include_directories(${target_name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    target_link_libraries(${target_name} PRIVATE
        imgui
        imguizmo::imguizmo
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
        metallic_copy_nrc_runtime(${target_name})
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

add_executable(MetallicRtxdiSample
    Samples/RtxdiSampleMain.cpp
    ${METALLIC_EDITOR_SOURCES}
)
configure_metallic_editor_sample(MetallicRtxdiSample)

add_executable(MetallicGPUDrivenSample
    Samples/GPUDrivenSampleMain.cpp
    ${METALLIC_EDITOR_SOURCES}
)
configure_metallic_editor_sample(MetallicGPUDrivenSample)

add_executable(MetallicRtxcrSample
    Samples/RtxcrSampleMain.cpp
    ${METALLIC_EDITOR_SOURCES}
)
configure_metallic_editor_sample(MetallicRtxcrSample)
