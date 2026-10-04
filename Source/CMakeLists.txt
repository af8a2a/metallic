set(METALLIC_SLANG_ROOT "${CMAKE_SOURCE_DIR}/External/slang")
include(${CMAKE_SOURCE_DIR}/cmake/SetupGDeflate.cmake)
if(DEFINED SLANG_ROOT)
    set(METALLIC_SLANG_ROOT "${SLANG_ROOT}")
endif()
find_package(slang 2026.18.2 EXACT CONFIG REQUIRED PATHS "${METALLIC_SLANG_ROOT}" NO_DEFAULT_PATH)
include("${CMAKE_SOURCE_DIR}/cmake/ShaderWarmup.cmake")

add_library(MetallicDebugCore STATIC
    Runtime/Debug/DebugTypes.cpp
    Runtime/Debug/DebugEval.cpp
    Runtime/Debug/DebugCore.cpp
    Runtime/Debug/DebugHash.cpp
    Runtime/Debug/ShaderTraceCore.cpp
    Runtime/Debug/DebugAnalysis.cpp
    Runtime/Debug/DebugProbe.cpp
    Runtime/Debug/DebugWatch.cpp
    Runtime/Debug/DebugTransport.cpp
)
target_include_directories(MetallicDebugCore PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(MetallicDebugCore PUBLIC nlohmann_json)
add_executable(MetallicCtl Tools/MetallicCtl.cpp)
target_link_libraries(MetallicCtl PRIVATE MetallicDebugCore)
set_target_properties(MetallicCtl PROPERTIES OUTPUT_NAME metallicctl)
if(WIN32)
    target_compile_definitions(MetallicDebugCore PRIVATE NOMINMAX)
    target_link_libraries(MetallicDebugCore PRIVATE advapi32)
endif()

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
    Runtime/Render/Debug/RenderDebug.cpp
    Runtime/Render/Debug/ShaderTraceRuntime.cpp
    Runtime/Render/Debug/WorkControlShaderTrace.cpp
    Runtime/Render/Debug/RenderDebugProviders.cpp
    Runtime/Render/Debug/GPUDebugProbe.cpp
    Runtime/Render/Core/ComputeProgram.cpp
    Runtime/Render/Core/ColorGrading.cpp
    Runtime/Render/Core/ACESTables.cpp
    Runtime/Render/Core/ComputeKernel.cpp
    Runtime/Render/Core/ResourceRegistry.cpp
    Runtime/Render/VisibilityHybridRasterizer.cpp
    Runtime/Render/MeshletLOD.cpp
    Runtime/Render/ResidentMeshletLOD.cpp
    Runtime/Render/MaterialBinning.cpp
    Runtime/Render/ClusterLightGrid.cpp
    Runtime/Render/Core/RenderFrameContext.cpp
    Runtime/Render/GAPI/CommandSubmission.cpp
    Runtime/Render/Core/RHISmokeTests.cpp
    Runtime/Render/Core/HistoryResources.cpp
    Runtime/Render/ImportanceSampling.cpp
    Runtime/Render/ReGIR.cpp
    Runtime/Render/Streamer/SceneResourceManager.cpp
    Runtime/Render/SceneLightResources.cpp
    Runtime/Render/Subsystem/BuiltinRenderSubsystems.cpp
    Runtime/Render/Subsystem/EnvironmentLightingSubsystem.cpp
    Runtime/Render/Subsystem/GPUScene.cpp
    Runtime/Render/Subsystem/GPUSceneSubsystem.cpp
    Runtime/Render/Subsystem/RenderSubsystem.cpp
    Runtime/Render/Subsystem/RenderWorld.cpp
    Runtime/Render/Core/RenderView.cpp
    Runtime/Render/Streamer/MeshletStreamCLAS.cpp
    Runtime/Render/Streamer/MeshletStreamCLASPool.cpp
    Runtime/Render/Streamer/MeshletStreamCompactCLASPool.cpp
    Runtime/Render/Streamer/MeshletStreamInitialLoader.cpp
    Runtime/Render/Streamer/MeshletStreamPageLoader.cpp
    Runtime/Render/Streamer/MeshletStreamResidency.cpp
    Runtime/Render/Streamer/MeshletStreamRuntime.cpp
    Runtime/Render/NeuralTextureResources.cpp
    Runtime/Render/RayTracing/SceneAccelerationStructure.cpp
    Runtime/Render/RayTracing/OpacityMicromapBake.cpp
    Runtime/Render/GAPI/PipelineCacheFile.cpp
    Runtime/Render/GAPI/PipelineStateHash.cpp
    Runtime/Render/Streamer/UploadStreamer.cpp
    Runtime/Render/RenderSample.cpp
    Runtime/Render/RenderGraph/RenderGraph.cpp
    Runtime/Render/RenderGraph/RenderGraphNode.cpp
    Runtime/Render/RenderGraph/RenderGraphAccessPlan.cpp
    Runtime/Render/RenderGraph/RenderGraphResourceAliasPlan.cpp
    Runtime/Render/RenderGraph/RenderGraphTextureAliasPlan.cpp
    Runtime/Render/RenderGraph/RenderGraphComputeStages.cpp
    Runtime/Render/RenderGraph/RenderGraphExecutor.cpp
    Runtime/Render/Streamer/StreamingUploads.cpp
    Runtime/Render/Streamer/StreamerSubsystem.cpp
    Runtime/Render/RenderGraph/NRDRuntime.cpp
    Runtime/Render/ScreenSpaceShadows.cpp
    Runtime/Render/RenderPass/BuiltinRenderPasses.cpp
    Runtime/Render/RenderPass/BuiltinPass/BunnyWireframePass.cpp
    Runtime/Render/RenderPass/BuiltinPass/ClearColorPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/LightGridDebugPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/CopyColorPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/FinalBlitPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/ColorGradingLUTPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferMaterialPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/AutoExposurePass.cpp
    Runtime/Render/RenderPass/BuiltinPass/SliderDebugPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/ScreenSpaceShadowPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/VisibilityBufferPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/GPUDrivenStreamAssetPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/ImageSamplePass.cpp
    Runtime/Render/RenderPass/BuiltinPass/NRDDenoisePass.cpp
    Runtime/Render/RenderPass/BuiltinPass/RenderGraphBufferPasses.cpp
    Runtime/Render/RenderPass/BuiltinPass/SceneMaterialShaderObjectPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/SceneMaterialVisualizationPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/ScenePathTracePass.cpp
    Runtime/Render/RenderPass/BuiltinPass/RTXDIConfidencePass.cpp
    Runtime/Render/RenderPass/BuiltinPass/RTXDICompositePass.cpp
    Runtime/Render/RenderPass/BuiltinPass/RTXCRMaterialSamplePass.cpp
    Runtime/Render/RenderPass/BuiltinPass/SceneRTXDIPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/SceneRayQueryVisualizationPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/SceneAccelerationStructurePass.cpp
    Runtime/Render/RenderPass/BuiltinPass/StreamlineDLSSRRPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/DLSSNRPass.cpp
    Runtime/Render/RenderPass/BuiltinPass/TriangleRasterPass.cpp
    Runtime/Render/Streamer/ScenePathTraceResources.cpp
    Runtime/Render/Material/MaterialRuntime.cpp
    Runtime/Render/Material/MaterialCoverageProgram.cpp
    Runtime/Render/Material/MaterialValueProgram.cpp
    Runtime/Render/Material/MaterialExecutable.cpp
    Runtime/Render/Streamer/Ktx2Texture.cpp
    Runtime/Render/Profiling/NsightAftermath.cpp
    Runtime/Render/Profiling/NsightGraphicsCapture.cpp
    Runtime/Render/Profiling/PacingTrace.cpp
    Runtime/Render/Profiling/NvPerf.cpp
    Runtime/Render/Profiling/WorkControlReplay.cpp
    Runtime/Render/Profiling/TracyProfiler.cpp
    Runtime/Render/GAPI/Vulkan/VulkanSynchronization.cpp
    Runtime/Render/GAPI/Vulkan/VulkanTrace.cpp
    Runtime/Render/GAPI/Vulkan/VulkanRHI.cpp
    Runtime/Render/GAPI/Vulkan/VulkanResult.cpp
    Runtime/Render/GAPI/Vulkan/VulkanDeviceProperties.cpp
    Runtime/Render/GAPI/Vulkan/VulkanShaderPrintf.cpp
    Runtime/Render/GAPI/Vulkan/VulkanGeneratedCommands.cpp
    Runtime/Render/GAPI/Vulkan/VulkanNRCWrapper.cpp
    Runtime/Render/RayTracing/SceneAccelerationStructureExtensions.cpp
    Runtime/Render/GAPI/Vulkan/VulkanStreamline.cpp
    Runtime/Render/GAPI/Vulkan/VulkanDLSSNR.cpp
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
    MetallicDebugCore
    MetallicRuntimeTask
    metallic::volk
    metallic::sdl3
    metallic::aftermath
    metallic::nsight_events
    metallic::streamline
    metallic::dlss_nr
    metallic::nrc
    nlohmann_json
    MetallicShaderWarmupCore
    slang::slang
    spdlog::spdlog
)

include("${CMAKE_SOURCE_DIR}/cmake/Zstd.cmake")
target_link_libraries(MetallicRuntimeRender PRIVATE
    libzstd_static
    metallic::nsight_graphics_capture
    metallic::nvperf
)

if(METALLIC_ENABLE_TRACY AND TRACY_ENABLE)
    target_link_libraries(MetallicRuntimeRender PUBLIC Tracy::TracyClient)
    target_compile_definitions(MetallicRuntimeRender PUBLIC METALLIC_HAS_TRACY=1)
else()
    target_compile_definitions(MetallicRuntimeRender PUBLIC METALLIC_HAS_TRACY=0)
endif()

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

if(METALLIC_ENABLE_NRD)
    target_sources(MetallicRuntimeRender PRIVATE
        Runtime/Render/Denoising/NRDPlan.cpp
        Runtime/Render/Denoising/NRDReblur.cpp
        Runtime/Render/Denoising/NRDRelax.cpp
        Runtime/Render/Denoising/NRDReference.cpp
        Runtime/Render/Denoising/NRDSigma.cpp
    )
    target_link_libraries(MetallicRuntimeRender PRIVATE MathLib)
    target_include_directories(MetallicRuntimeRender PRIVATE "${CMAKE_SOURCE_DIR}")
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
    target_link_libraries(MetallicRuntimeRender PRIVATE advapi32)
    target_compile_options(MetallicRuntimeRender PRIVATE /EHsc)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(MetallicRuntimeRender PRIVATE -Wno-nullability-completeness)
endif()

add_library(MetallicRuntimeScene STATIC
    Runtime/Material/MaterialAsset.cpp
    Runtime/Material/MaterialFeatures.cpp
    Runtime/Material/MaterialValueIR.cpp
    Runtime/Material/MaterialClosureIR.cpp
    Runtime/Material/MaterialClosureClassification.cpp
    Runtime/Scene/glTFGPUInstancing.cpp
    Runtime/Scene/GeometryAttributes.cpp
    Runtime/Scene/MeshletStreamAsset.cpp
    Runtime/Scene/MeshletStreamGPUCodec.cpp
    Runtime/Scene/MeshletStreamReferenceCodec.cpp
    Runtime/Scene/MeshletBuildParallel.cpp
    Runtime/Scene/Scene.cpp
    Runtime/Scene/SceneDocument.cpp
    Runtime/Scene/SceneGraph.cpp
    Runtime/Scene/SceneLoader.cpp
    Runtime/Scene/ScenePicker.cpp
    Runtime/Scene/TinyGLTFImpl.cpp
)
if(METALLIC_ENABLE_OPENUSD)
    target_sources(MetallicRuntimeScene PRIVATE Runtime/Scene/USDSceneImporter.cpp)
else()
    target_sources(MetallicRuntimeScene PRIVATE Runtime/Scene/USDSceneImporterStub.cpp)
endif()

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
    MetallicGDeflate
    EnTT::EnTT
    MetallicRuntimeTask
    MathLib
    meshoptimizer
    nlohmann_json
    spdlog::spdlog
    tinygltf
    metallic::openusd
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

# CPU-only offline cooker. Avoids editor/Vulkan/RTAS startup and can be built
# independently while processing assets larger than resident scene memory.
add_executable(MetallicMeshletCook Tools/MeshletCookMain.cpp)
add_executable(MetallicMeshletTranscode Tools/MeshletTranscodeMain.cpp)
target_link_libraries(MetallicMeshletTranscode PRIVATE MetallicRuntimeScene)
metallic_copy_openusd_runtime(MetallicMeshletTranscode)
metallic_copy_spdlog_runtime(MetallicMeshletTranscode)
target_link_libraries(MetallicMeshletCook PRIVATE MetallicRuntimeScene)
if(WIN32)
    target_compile_definitions(MetallicMeshletCook PRIVATE NOMINMAX)
    target_link_libraries(MetallicMeshletCook PRIVATE Psapi)
endif()
metallic_copy_openusd_runtime(MetallicMeshletCook)
metallic_copy_spdlog_runtime(MetallicMeshletCook)

# Resident cache migration keeps the MTLMSHLT payload separate from meshstream.
add_executable(MetallicResidentMeshletCook Tools/MeshletResidentCookMain.cpp)
target_link_libraries(MetallicResidentMeshletCook PRIVATE MetallicRuntimeScene)
target_compile_definitions(MetallicResidentMeshletCook PRIVATE
    METALLIC_ASSET_ROOT="${PROJECT_SOURCE_DIR}/Asset")
if(WIN32)
    target_compile_definitions(MetallicResidentMeshletCook PRIVATE NOMINMAX)
endif()
metallic_copy_openusd_runtime(MetallicResidentMeshletCook)
metallic_copy_spdlog_runtime(MetallicResidentMeshletCook)

# Keep shader symbols independent of opt-in Nsight capture injection.
set_property(SOURCE Editor/EditorApplication.cpp APPEND PROPERTY COMPILE_DEFINITIONS
    METALLIC_DEFAULT_SHADER_CAPTURE_SYMBOLS=$<CONFIG:RelWithDebInfo>
)

set(METALLIC_EDITOR_SOURCES
    Editor/EditorApplication.cpp
    Editor/EditorDisplayRenderer.cpp
    Editor/EditorApplicationSmokeTests.cpp
    Editor/EditorFullRoamBenchmark.cpp
    Editor/EditorRasterComparison.cpp
    Editor/EditorMaterialInspector.cpp
    Editor/EditorMaterialInspectorSmoke.cpp
    Editor/EditorSliderDebug.cpp
    Editor/EditorStreamlineDebug.cpp
    Editor/EditorProfiler.cpp
    Editor/EditorRenderGraphViewer.cpp
    Editor/NVMLMonitor.cpp
)

if(WIN32 AND METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE)
    add_executable(MetallicNsightReplay Tools/NsightReplayMain.cpp)
    target_include_directories(MetallicNsightReplay PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_definitions(MetallicNsightReplay PRIVATE NOMINMAX)
endif()

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
    metallic_copy_dlss_nr_runtime(Metallic)
    metallic_copy_aftermath_runtime(Metallic)
    metallic_copy_nrc_runtime(Metallic)
endif()
metallic_copy_openusd_runtime(Metallic)
metallic_copy_spdlog_runtime(Metallic)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
if(TARGET MetallicNsightReplay)
    add_dependencies(Metallic MetallicNsightReplay)
endif()

function(configure_metallic_editor_sample target_name)
    if(TARGET MetallicNsightReplay)
        add_dependencies(${target_name} MetallicNsightReplay)
    endif()
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
        metallic_copy_dlss_nr_runtime(${target_name})
        metallic_copy_aftermath_runtime(${target_name})
        metallic_copy_nrc_runtime(${target_name})
    endif()
    metallic_copy_openusd_runtime(${target_name})
    metallic_copy_spdlog_runtime(${target_name})
endfunction()

add_executable(MetallicMaterialVisualizationSample
    Samples/MaterialVisualizationSampleMain.cpp
    ${METALLIC_EDITOR_SOURCES}
)
configure_metallic_editor_sample(MetallicMaterialVisualizationSample)

add_executable(LookDev
    LookDev/LookDevMain.cpp
    ${METALLIC_EDITOR_SOURCES}
)
configure_metallic_editor_sample(LookDev)
if(WIN32)
    target_compile_options(LookDev PRIVATE /EHsc)
endif()

add_executable(MetallicPathTracingSample
    Samples/PathTracingSampleMain.cpp
    ${METALLIC_EDITOR_SOURCES}
)
configure_metallic_editor_sample(MetallicPathTracingSample)

add_executable(MetallicRTXDISample
    Samples/RTXDISampleMain.cpp
    ${METALLIC_EDITOR_SOURCES}
)
configure_metallic_editor_sample(MetallicRTXDISample)

add_executable(MetallicGPUDrivenSample
    Samples/GPUDrivenSampleMain.cpp
    ${METALLIC_EDITOR_SOURCES}
)
configure_metallic_editor_sample(MetallicGPUDrivenSample)

add_executable(MetallicRTXCRSample
    Samples/RTXCRSampleMain.cpp
    ${METALLIC_EDITOR_SOURCES}
)
configure_metallic_editor_sample(MetallicRTXCRSample)

# Manual diagnostics only; never a dependency of editor/sample/default builds.
add_executable(MetallicShaderPrintfProbe EXCLUDE_FROM_ALL Tools/ShaderPrintfProbe.cpp)
target_link_libraries(MetallicShaderPrintfProbe PRIVATE MetallicRuntimeRender)
target_include_directories(MetallicShaderPrintfProbe BEFORE PRIVATE "${METALLIC_SLANG_ROOT}/include")
target_compile_definitions(MetallicShaderPrintfProbe PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(WIN32)
    target_compile_definitions(MetallicShaderPrintfProbe PRIVATE NOMINMAX)
    target_compile_options(MetallicShaderPrintfProbe PRIVATE /EHsc /utf-8)
    foreach(_dll IN LISTS METALLIC_SLANG_RUNTIME_DLLS)
        add_custom_command(TARGET MetallicShaderPrintfProbe POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${_dll}" "$<TARGET_FILE_DIR:MetallicShaderPrintfProbe>")
    endforeach()
    if(TARGET SDL3-shared)
        set(_printf_sdl SDL3-shared)
    else()
        set(_printf_sdl SDL3::SDL3-shared)
    endif()
    add_custom_command(TARGET MetallicShaderPrintfProbe POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:${_printf_sdl}>" "$<TARGET_FILE_DIR:MetallicShaderPrintfProbe>")
    metallic_copy_streamline_runtime(MetallicShaderPrintfProbe)
    metallic_copy_dlss_nr_runtime(MetallicShaderPrintfProbe)
    metallic_copy_aftermath_runtime(MetallicShaderPrintfProbe)
    metallic_copy_nrc_runtime(MetallicShaderPrintfProbe)
endif()
metallic_copy_spdlog_runtime(MetallicShaderPrintfProbe)

# One definition for the runtime and every consumer; public RHI layouts are unchanged.
target_compile_definitions(MetallicRuntimeRender PUBLIC METALLIC_RHI_DIAGNOSTICS=$<BOOL:${METALLIC_RHI_DIAGNOSTICS}>)
