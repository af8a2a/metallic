include(GoogleTest)

find_package(Threads REQUIRED)
add_executable(MetallicShaderWarmupStartupTests
    editor/ShaderWarmupStartupTests.cpp
    "${CMAKE_SOURCE_DIR}/Tools/ShaderWarmup.cpp")
target_include_directories(MetallicShaderWarmupStartupTests PRIVATE
    "${CMAKE_SOURCE_DIR}" "${CMAKE_SOURCE_DIR}/Source")
target_compile_definitions(MetallicShaderWarmupStartupTests PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    METALLIC_RTXCR_SHADER_INCLUDE_DIR="${METALLIC_RTXCR_SHADER_INCLUDE_DIR}")
target_link_libraries(MetallicShaderWarmupStartupTests PRIVATE GTest::gtest_main spdlog::spdlog Threads::Threads)
if(MSVC)
    target_compile_options(MetallicShaderWarmupStartupTests PRIVATE /EHsc /utf-8)
endif()
metallic_copy_spdlog_runtime(MetallicShaderWarmupStartupTests)
add_test(NAME MetallicShaderWarmupStartupTests COMMAND MetallicShaderWarmupStartupTests)
set_tests_properties(MetallicShaderWarmupStartupTests PROPERTIES LABELS "editor;shader" TIMEOUT 30)


add_executable(MetallicShaderRequestsTests editor/ShaderRequestsTests.cpp)
target_include_directories(MetallicShaderRequestsTests PRIVATE "${CMAKE_SOURCE_DIR}/Source")
target_compile_definitions(MetallicShaderRequestsTests PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    TEST_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    METALLIC_RTXCR_SHADER_INCLUDE_DIR="${METALLIC_RTXCR_SHADER_INCLUDE_DIR}")
target_link_libraries(MetallicShaderRequestsTests PRIVATE MetallicShaderWarmupCore GTest::gtest_main)
if(MSVC)
    target_compile_options(MetallicShaderRequestsTests PRIVATE /EHsc /utf-8)
endif()
if(WIN32)
    set(_shader_request_test_slang_root "${CMAKE_SOURCE_DIR}/External/slang")
    if(DEFINED SLANG_ROOT)
        set(_shader_request_test_slang_root "${SLANG_ROOT}")
    endif()
    file(GLOB _shader_request_test_slang_dlls "${_shader_request_test_slang_root}/bin/*.dll")
    foreach(_dll IN LISTS _shader_request_test_slang_dlls)
        add_custom_command(TARGET MetallicShaderRequestsTests POST_BUILD COMMAND ${CMAKE_COMMAND} -E
            copy_if_different "${_dll}" "$<TARGET_FILE_DIR:MetallicShaderRequestsTests>")
    endforeach()
endif()
metallic_copy_spdlog_runtime(MetallicShaderRequestsTests)
add_test(NAME MetallicShaderRequestsTests COMMAND MetallicShaderRequestsTests)
set_tests_properties(MetallicShaderRequestsTests PROPERTIES LABELS "editor;shader" TIMEOUT 300)


add_subdirectory(perf)

if(METALLIC_ENABLE_NRD)
    add_executable(MetallicNRDTests rhi/NRDTests.cpp)
    target_link_libraries(MetallicNRDTests PRIVATE MetallicRuntimeRender MathLib GTest::gtest_main)
    target_include_directories(MetallicNRDTests PRIVATE "${CMAKE_SOURCE_DIR}")
    target_compile_definitions(MetallicNRDTests PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    if(WIN32)
        target_compile_definitions(MetallicNRDTests PRIVATE NOMINMAX)
        set(_nrd_test_slang_root "${CMAKE_SOURCE_DIR}/External/slang")
        if(DEFINED SLANG_ROOT)
            set(_nrd_test_slang_root "${SLANG_ROOT}")
        endif()
        file(GLOB _nrd_test_slang_dlls "${_nrd_test_slang_root}/bin/*.dll")
        foreach(_dll IN LISTS _nrd_test_slang_dlls)
            add_custom_command(TARGET MetallicNRDTests POST_BUILD COMMAND ${CMAKE_COMMAND} -E
                copy_if_different "${_dll}" "$<TARGET_FILE_DIR:MetallicNRDTests>")
        endforeach()
        metallic_copy_spdlog_runtime(MetallicNRDTests)
        metallic_copy_streamline_runtime(MetallicNRDTests)
    endif()
    add_test(NAME MetallicNRDTests COMMAND MetallicNRDTests --gtest_color=no)
    set_tests_properties(MetallicNRDTests PROPERTIES LABELS "rhi;nrd" TIMEOUT 300)
endif()

add_executable(MetallicNsightLaunchOptionsTests editor/NsightLaunchOptionsTests.cpp)
target_include_directories(MetallicNsightLaunchOptionsTests PRIVATE "${CMAKE_SOURCE_DIR}/Source")
target_link_libraries(MetallicNsightLaunchOptionsTests PRIVATE GTest::gtest_main)
add_test(NAME MetallicNsightLaunchOptionsTests COMMAND MetallicNsightLaunchOptionsTests)
set_tests_properties(MetallicNsightLaunchOptionsTests PROPERTIES LABELS "editor;profiling")

option(METALLIC_NSIGHT_INTEGRATION_TESTS "Run installed Nsight host/GPU integration regressions" OFF)
if(WIN32 AND METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE AND METALLIC_NSIGHT_INTEGRATION_TESTS)
    foreach(nsight_case IN ITEMS disabled export collection invalid-metrics)
        add_test(NAME MetallicNsightGPUTrace.${nsight_case}
            COMMAND ${CMAKE_COMMAND}
                "-DEDITOR_EXECUTABLE=$<TARGET_FILE:LookDev>"
                "-DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/nsight-gputrace-${nsight_case}"
                "-DCASE=${nsight_case}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/editor/RunNsightGPUTraceStartup.cmake")
        set_tests_properties(MetallicNsightGPUTrace.${nsight_case} PROPERTIES
            LABELS "editor;profiling;gpu" TIMEOUT 200 RESOURCE_LOCK MetallicEditor)
        if(nsight_case STREQUAL "collection")
            set_tests_properties(MetallicNsightGPUTrace.${nsight_case} PROPERTIES TIMEOUT 620)
        endif()
    endforeach()
endif()

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.27)
    foreach(dependency_case IN ITEMS reuse boolean missing compiler configuration crt flags features multiconfig)
        add_test(NAME MetallicDependencyCache.${dependency_case}
            COMMAND ${CMAKE_COMMAND}
                "-DCASE=${dependency_case}"
                "-DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/dependency-cache-tests"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/DependencyCacheTests.cmake")
        set_tests_properties(MetallicDependencyCache.${dependency_case} PROPERTIES LABELS "build")
    endforeach()
endif()

if(WIN32)
    add_test(NAME MetallicDLSSNRConfiguration
        COMMAND ${CMAKE_COMMAND}
            "-DSOURCE_DIRECTORY=${CMAKE_SOURCE_DIR}"
            "-DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/dlss-nr-configuration"
            "-DTEST_GENERATOR=${CMAKE_GENERATOR}"
            "-DTEST_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/DLSSNRConfigurationTests.cmake")
    set_tests_properties(MetallicDLSSNRConfiguration PROPERTIES LABELS "build;dlss-nr" TIMEOUT 120)
    add_test(NAME MetallicStreamlineReflexConfiguration
        COMMAND ${CMAKE_COMMAND}
            "-DSOURCE_DIRECTORY=${CMAKE_SOURCE_DIR}"
            "-DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/streamline-reflex-configuration"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/StreamlineReflexTests.cmake")
    set_tests_properties(MetallicStreamlineReflexConfiguration PROPERTIES LABELS "build;reflex")
    if(METALLIC_HAS_NV_LOW_LATENCY)
        foreach(reflex_mode IN ITEMS off on boost)
            add_test(NAME MetallicEditorReflex.${reflex_mode}
                COMMAND ${CMAKE_COMMAND}
                    "-DEDITOR_EXECUTABLE=$<TARGET_FILE:Metallic>"
                    "-DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/editor-reflex-${reflex_mode}"
                    "-DSOURCE_DIRECTORY=${CMAKE_SOURCE_DIR}"
                    "-DREFLEX_MODE=${reflex_mode}"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/editor/RunReflexSmoke.cmake")
            set_tests_properties(MetallicEditorReflex.${reflex_mode} PROPERTIES
                LABELS "editor;reflex" TIMEOUT 180 RESOURCE_LOCK MetallicEditor
                SKIP_REGULAR_EXPRESSION "Reflex unavailable; skipping")
        endforeach()
    endif()
    if(METALLIC_ENABLE_STREAMLINE)
        add_test(NAME MetallicEditorDLSSCameraSmoke
            COMMAND ${CMAKE_COMMAND}
                "-DEDITOR_EXECUTABLE=$<TARGET_FILE:Metallic>"
                "-DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/editor-dlss-camera"
                "-DSOURCE_DIRECTORY=${CMAKE_SOURCE_DIR}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/editor/RunDLSSCameraSmoke.cmake"
        )
        set_tests_properties(MetallicEditorDLSSCameraSmoke PROPERTIES
            LABELS "editor;dlss" TIMEOUT 90 RESOURCE_LOCK MetallicEditor
        )
    endif()
    add_test(NAME MetallicLookDevMaterialInspectorSmoke
        COMMAND ${CMAKE_COMMAND}
            "-DLOOKDEV_EXECUTABLE=$<TARGET_FILE:LookDev>"
            "-DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/lookdev-material-inspector"
            "-DSOURCE_DIRECTORY=${CMAKE_SOURCE_DIR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/editor/RunMaterialInspectorSmoke.cmake"
    )
    set_tests_properties(MetallicLookDevMaterialInspectorSmoke PROPERTIES
        LABELS "editor" TIMEOUT 90 RESOURCE_LOCK MetallicEditor
    )
    add_test(NAME MetallicLookDevSceneSwitchSmoke
        COMMAND ${CMAKE_COMMAND}
            "-DLOOKDEV_EXECUTABLE=$<TARGET_FILE:LookDev>"
            "-DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/lookdev-scene-switch"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/editor/RunSceneSwitchSmoke.cmake"
    )
    set_tests_properties(MetallicLookDevSceneSwitchSmoke PROPERTIES
        LABELS "editor" TIMEOUT 180 RESOURCE_LOCK MetallicEditor
    )
    add_test(NAME MetallicLookDevVBufferSmoke
        COMMAND ${CMAKE_COMMAND}
            "-DLOOKDEV_EXECUTABLE=$<TARGET_FILE:LookDev>"
            "-DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/lookdev-vbuffer"
            -DSAMPLE_ID=lookdev-vbuffer
            -P "${CMAKE_CURRENT_SOURCE_DIR}/editor/RunSliderDebugSmoke.cmake"
    )
    set_tests_properties(MetallicLookDevVBufferSmoke PROPERTIES
        LABELS "editor" TIMEOUT 90 RESOURCE_LOCK MetallicEditor
    )
    add_test(NAME MetallicLookDevSliderSmoke
        COMMAND ${CMAKE_COMMAND}
            "-DLOOKDEV_EXECUTABLE=$<TARGET_FILE:LookDev>"
            "-DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/lookdev-slider"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/editor/RunSliderDebugSmoke.cmake"
    )
    set_tests_properties(MetallicLookDevSliderSmoke PROPERTIES
        LABELS "editor" TIMEOUT 90 RESOURCE_LOCK MetallicEditor
    )
    add_test(NAME MetallicEditorFinalBlitSmoke
        COMMAND ${CMAKE_COMMAND}
            "-DEDITOR_EXECUTABLE=$<TARGET_FILE:Metallic>"
            "-DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/editor-final-blit"
            -DFINAL_BLIT=ON
            -P "${CMAKE_CURRENT_SOURCE_DIR}/editor/RunMultiViewportSmoke.cmake"
    )
    set_tests_properties(MetallicEditorFinalBlitSmoke PROPERTIES
        LABELS "editor"
        TIMEOUT 90
        RESOURCE_LOCK MetallicEditor
    )
    add_test(NAME MetallicEditorMultiViewportSmoke
        COMMAND ${CMAKE_COMMAND}
            "-DEDITOR_EXECUTABLE=$<TARGET_FILE:Metallic>"
            "-DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/editor-multi-viewport"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/editor/RunMultiViewportSmoke.cmake"
    )
    set_tests_properties(MetallicEditorMultiViewportSmoke PROPERTIES
        LABELS "editor"
        TIMEOUT 90
        RESOURCE_LOCK MetallicEditor
    )
endif()

add_executable(MetallicDebugTests debug/DebugCoreTests.cpp debug/ShaderTraceTests.cpp)
target_link_libraries(MetallicDebugTests PRIVATE MetallicDebugCore GTest::gtest_main)
if(WIN32)
    target_compile_definitions(MetallicDebugTests PRIVATE NOMINMAX)
endif()
add_test(NAME MetallicDebugTests COMMAND MetallicDebugTests --gtest_color=no)
set_tests_properties(MetallicDebugTests PROPERTIES LABELS "debug")

add_executable(MetallicTaskTests
    task/TaskGraphTests.cpp
)

target_include_directories(MetallicTaskTests PRIVATE
    "${CMAKE_SOURCE_DIR}/Source"
)

target_link_libraries(MetallicTaskTests PRIVATE
    MetallicRuntimeTask
    GTest::gtest_main
)

if(WIN32)
    target_compile_definitions(MetallicTaskTests PRIVATE NOMINMAX)
    target_compile_options(MetallicTaskTests PRIVATE /EHsc)
endif()

add_test(
    NAME MetallicTaskTests
    COMMAND MetallicTaskTests --gtest_color=no
)
set_tests_properties(MetallicTaskTests PROPERTIES
    LABELS "task"
)

add_executable(MetallicSceneTests
    scene/MaterialAssetTests.cpp
    scene/glTFInstancingTests.cpp
    scene/GeometryAttributesTests.cpp
    scene/MeshletStreamReferenceCodecTests.cpp
    scene/MeshletStreamPrefetchTests.cpp
    scene/MeshletBuildParallelTests.cpp
    scene/MeshletReferenceSceneBuildTests.cpp
    scene/Main.cpp
    scene/SceneGraphTests.cpp
)

target_include_directories(MetallicSceneTests PRIVATE
    "${CMAKE_SOURCE_DIR}/Source"
)

target_link_libraries(MetallicSceneTests PRIVATE
    MetallicRuntimeScene
    GTest::gtest_main
)

target_compile_definitions(MetallicSceneTests PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

if(WIN32)
    target_compile_definitions(MetallicSceneTests PRIVATE NOMINMAX)
endif()
metallic_copy_openusd_runtime(MetallicSceneTests)
metallic_copy_spdlog_runtime(MetallicSceneTests)

add_test(
    NAME MetallicSceneTests
    COMMAND MetallicSceneTests
)
set_tests_properties(MetallicSceneTests PROPERTIES
    LABELS "scene"
)

add_executable(MetallicGPUPageTests
    scene/GPUPageCodecTests.cpp
    rhi/GPUMemoryDecompressionTests.cpp
    ${CMAKE_SOURCE_DIR}/Source/Runtime/Render/Streamer/MeshletStreamCLAS.cpp
    ${CMAKE_SOURCE_DIR}/Source/Runtime/Render/Streamer/MeshletStreamPageLoader.cpp
)
target_link_libraries(MetallicGPUPageTests PRIVATE MetallicRuntimeScene volk GTest::gtest_main)
target_include_directories(MetallicGPUPageTests PRIVATE ${CMAKE_SOURCE_DIR}/External/volk)
target_compile_definitions(MetallicGPUPageTests PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}" NOMINMAX)
metallic_copy_spdlog_runtime(MetallicGPUPageTests)
metallic_copy_openusd_runtime(MetallicGPUPageTests)
add_test(NAME MetallicGPUPageTests COMMAND MetallicGPUPageTests)
set_tests_properties(MetallicGPUPageTests PROPERTIES LABELS "scene;rhi")

# Embed the offline report at build time; no server, Python or copied runtime assets.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/rhi/harness/HtmlReport.html")
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/rhi/harness/HtmlReport.html" METALLIC_REPORT_HTML)
configure_file(rhi/harness/HtmlReportTemplate.h.in "${CMAKE_CURRENT_BINARY_DIR}/generated/HtmlReportTemplate.h" @ONLY)

add_executable(MetallicRHITests
    rhi/RHIHandleTests.cpp
    rhi/GeneratedCommandsTests.cpp
    rhi/DebugControlTests.cpp
    rhi/Main.cpp
    rhi/RHITest.h
    rhi/harness/Requirements.cpp
    rhi/harness/Evidence.cpp
    rhi/harness/HtmlReport.cpp
    rhi/harness/HtmlReportTests.cpp
    rhi/harness/GTestHtmlReport.cpp
    rhi/harness/ValidationRecorder.cpp
    rhi/harness/VulkanDiagnostics.cpp
    rhi/harness/ProcessRunner.cpp
    rhi/harness/BufferSequence.cpp
    rhi/harness/SequenceShrinker.cpp
    rhi/harness/DiagnosticCases.cpp
    rhi/harness/TraceRecorder.cpp
    rhi/harness/Runner.cpp
    rhi/harness/ExtensionCases.cpp
    rhi/harness/CoreCases.cpp
    rhi/harness/SampleCases.cpp
    rhi/harness/HarnessTests.cpp
    rhi/NativeDescriptorHeapTests.cpp
    rhi/NativeDescriptorHeapSPIRVTests.cpp
    rhi/BindlessBufferTests.cpp
    rhi/DescriptorHeapCodePatternTests.cpp
    rhi/DLSSMotionVectorTests.cpp
    rhi/TextureFootprintTests.cpp
    rhi/DLSSNRTests.cpp
    rhi/AutoExposureTests.cpp
    rhi/OpenPBRLookDevTests.cpp
    rhi/SliderDebugTests.cpp
    rhi/VisibilityBufferDeferredTests.cpp
    rhi/StreamMaterialTests.cpp
    rhi/ZorahFullFirstFrameTests.cpp
    rhi/MaterialBinningTests.cpp
    rhi/MaterialRuntimeTests.cpp
    rhi/MaterialProgramTests.cpp
    rhi/MaterialClosureTests.cpp
    rhi/MaterialAssetTests.cpp
    rhi/MaterialBaselineTests.cpp
    rhi/MaterialValueProgramTests.cpp
    rhi/ResourceRegistryTests.cpp
    rhi/ResourceMemoryInfoTests.cpp
    rhi/DeferredTransmissionTests.cpp
    rhi/ClusterLightGridTests.cpp
    rhi/CommandSubmissionTests.cpp
    rhi/RHIRefinementTests.cpp
    rhi/GPUProfilingTests.cpp
    rhi/EditorProfilerTests.cpp
    rhi/RenderGraphViewerTests.cpp
    "${CMAKE_SOURCE_DIR}/Source/Editor/EditorProfiler.cpp"
    "${CMAKE_SOURCE_DIR}/Source/Editor/EditorRenderGraphViewer.cpp"
    rhi/FrameContextTests.cpp
    rhi/ParallelRecordingTests.cpp
    rhi/SchedulingBenchmarkTests.cpp
    rhi/FinalBlitTests.cpp
    rhi/RTXDIPostProcessTests.cpp
    rhi/DisplayOutputTests.cpp
    rhi/EditorDisplayTests.cpp
    ../Source/Editor/EditorDisplayRenderer.cpp
    rhi/GPUSceneSubsystemTests.cpp
    rhi/MeshletLODTests.cpp
    rhi/StreamMeshletLODTests.cpp
    rhi/WaveWorkDistributionTests.cpp
    rhi/TessellationTests.cpp
    rhi/TessellationSplitTests.cpp
    rhi/StreamMeshletLODSceneTests.cpp
    rhi/MiniZorahStreamAssetTests.cpp
    rhi/MiniZorahRoamingTests.cpp
    rhi/MiniZorahQualityTests.cpp
    rhi/HistoryResourcesTests.cpp
    rhi/ImportedVirtualLightTests.cpp
    rhi/LightGridDebugPassTests.cpp
    rhi/PhotometricLightingTests.cpp
    rhi/RealtimePipelineTests.cpp
    rhi/EnvironmentPrefilterTests.cpp
    rhi/HybridRasterTests.cpp
    rhi/StreamClusterCandidateTests.cpp
    rhi/StreamClusterClassificationTests.cpp
    rhi/StreamGroupRasterTests.cpp
    rhi/StreamIndexedMeshTests.cpp
    rhi/VisibilityDebugStabilityTests.cpp
    rhi/MiniZorahDebugStabilityTests.cpp
    rhi/GPUDrivenCullingTests.cpp
    rhi/HZBSPDTests.cpp
    rhi/RenderViewTests.cpp
    rhi/ShaderToHumanTests.cpp
    rhi/ShaderHotReloadTests.cpp
    rhi/ShaderModuleTests.cpp
    rhi/SphericalHarmonicsTests.cpp
    rhi/ReGIRVirtualLightTests.cpp
    rhi/RenderGraphTests.cpp
    rhi/RenderGraphAccessPlanTests.cpp
    rhi/RenderGraphTextureAliasPlanTests.cpp
    rhi/RenderPassTransientTests.cpp
    rhi/TextureAliasingTests.cpp
    rhi/RenderGraphTextureAliasingTests.cpp
    rhi/BufferAliasingTests.cpp
    rhi/RenderGraphBufferAliasingTests.cpp
    rhi/RenderGraphAccelerationStructureTests.cpp
    rhi/SceneAccelerationStructurePassTests.cpp
    rhi/RenderGraphComputeStageTests.cpp
    rhi/SceneBindingTests.cpp
    rhi/SceneUploadTests.cpp
    rhi/RadianceCacheTests.cpp
    rhi/RenderingTests.cpp
    rhi/ResourceTests.cpp
    rhi/Ktx2TextureTests.cpp
    rhi/MemoryBudgetTests.cpp
    rhi/RayTracingAccelerationStructureTests.cpp
    rhi/UnifiedTopLevelTests.cpp
        rhi/CLASCompactionTests.cpp
    rhi/RayTracingPositionFetchTests.cpp
    rhi/OpacityMicromapTests.cpp
    rhi/StreamerTests.cpp
    rhi/StreamBLASAllocationTests.cpp
    rhi/GPUDecompressionTests.cpp
    rhi/TestOutput.cpp
    rhi/ValidationTests.cpp
    rhi/stb/Stb.cpp
)

target_include_directories(MetallicRHITests PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
    "${CMAKE_SOURCE_DIR}/Source"
    "${CMAKE_CURRENT_SOURCE_DIR}/rhi"
)

target_link_libraries(MetallicRHITests PRIVATE
    libzstd_static
    imgui
    metallic::sdl3
    MetallicRuntimeRender
    GTest::gtest
)

target_compile_definitions(MetallicRHITests PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

execute_process(COMMAND git rev-parse --short HEAD WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE _testbench_revision OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
target_compile_definitions(MetallicRHITests PRIVATE
    METALLIC_TESTBENCH_REVISION="${_testbench_revision}"
    METALLIC_TESTBENCH_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
)
if(WIN32)
    set(METALLIC_TESTBENCH_LAYER_PATH "" CACHE PATH "Optional explicit Vulkan layer directory for isolated testbench processes")
    set(_testbench_layer_arguments)
    if(METALLIC_TESTBENCH_LAYER_PATH)
        list(APPEND _testbench_layer_arguments --tb-layer-path "${METALLIC_TESTBENCH_LAYER_PATH}")
    endif()
    add_test(NAME MetallicTestbench.Harness COMMAND MetallicRHITests --tb-self-test)
    add_test(NAME MetallicTestbench.Contract COMMAND MetallicRHITests --tb-run --tb-suite contract --tb-require-all)
    add_test(NAME MetallicTestbench.Core COMMAND MetallicRHITests --tb-run --tb-suite core --tb-require-all ${_testbench_layer_arguments})
    add_test(NAME MetallicTestbench.Binding COMMAND MetallicRHITests --tb-run --tb-suite binding ${_testbench_layer_arguments})
    add_test(NAME MetallicTestbench.Sync COMMAND MetallicRHITests --tb-run --tb-suite sync --tb-validation sync ${_testbench_layer_arguments})
    add_test(NAME MetallicTestbench.SyncActivation COMMAND ${CMAKE_COMMAND}
        "-DEXECUTABLE=$<TARGET_FILE:MetallicRHITests>" "-DSOURCE=${CMAKE_SOURCE_DIR}"
        "-DLAYER_PATH=${METALLIC_TESTBENCH_LAYER_PATH}" -P "${CMAKE_CURRENT_SOURCE_DIR}/rhi/harness/VerifySyncProbe.cmake")
    set(METALLIC_TESTBENCH_REQUIRED_EXTENSIONS "" CACHE STRING "GoogleTest filter of extensions required on this reference machine")
    if(METALLIC_TESTBENCH_REQUIRED_EXTENSIONS)
        add_test(NAME MetallicTestbench.ExtensionsRequired COMMAND MetallicRHITests --tb-run --tb-suite extensions
            --tb-filter "${METALLIC_TESTBENCH_REQUIRED_EXTENSIONS}" --tb-require-all ${_testbench_layer_arguments})
        set_tests_properties(MetallicTestbench.ExtensionsRequired PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            LABELS "testbench;extensions-required" TIMEOUT 600 RESOURCE_LOCK MetallicGPU)
    endif()
    add_test(NAME MetallicTestbench.Extensions COMMAND MetallicRHITests --tb-run --tb-suite extensions ${_testbench_layer_arguments})
    add_test(NAME MetallicTestbench.Async COMMAND MetallicRHITests --tb-run --tb-suite async ${_testbench_layer_arguments})
    set_tests_properties(MetallicTestbench.Harness MetallicTestbench.Contract MetallicTestbench.Core MetallicTestbench.Async MetallicTestbench.Binding MetallicTestbench.Sync MetallicTestbench.SyncActivation MetallicTestbench.Extensions
        PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" LABELS "testbench" TIMEOUT 180)
    set_tests_properties(MetallicTestbench.Extensions PROPERTIES TIMEOUT 600)
    set_tests_properties(MetallicTestbench.Core MetallicTestbench.Async MetallicTestbench.Binding MetallicTestbench.Sync MetallicTestbench.SyncActivation MetallicTestbench.Extensions PROPERTIES RESOURCE_LOCK MetallicGPU)
endif()

if(WIN32)
    target_link_libraries(MetallicRHITests PRIVATE dxgi shell32)
    target_compile_definitions(MetallicRHITests PRIVATE NOMINMAX)
    target_compile_options(MetallicRHITests PRIVATE /EHsc)

    set(METALLIC_TEST_SLANG_ROOT "${CMAKE_SOURCE_DIR}/External/slang")
    if(DEFINED SLANG_ROOT)
        set(METALLIC_TEST_SLANG_ROOT "${SLANG_ROOT}")
    endif()

    file(GLOB METALLIC_TEST_SLANG_RUNTIME_DLLS "${METALLIC_TEST_SLANG_ROOT}/bin/*.dll")
    foreach(METALLIC_TEST_SLANG_RUNTIME_DLL IN LISTS METALLIC_TEST_SLANG_RUNTIME_DLLS)
        add_custom_command(TARGET MetallicRHITests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${METALLIC_TEST_SLANG_RUNTIME_DLL}"
                "$<TARGET_FILE_DIR:MetallicRHITests>"
        )
    endforeach()

    metallic_copy_streamline_runtime(MetallicRHITests)
    metallic_copy_dlss_nr_runtime(MetallicRHITests)
    metallic_copy_aftermath_runtime(MetallicRHITests)
    metallic_copy_nrc_runtime(MetallicRHITests)
endif()
metallic_copy_openusd_runtime(MetallicRHITests)
metallic_copy_spdlog_runtime(MetallicRHITests)

if(WIN32)
    set(METALLIC_TEST_SDL_RUNTIME_TARGET "")
    if(TARGET SDL3-shared)
        set(METALLIC_TEST_SDL_RUNTIME_TARGET SDL3-shared)
    elseif(TARGET SDL3::SDL3-shared)
        set(METALLIC_TEST_SDL_RUNTIME_TARGET SDL3::SDL3-shared)
    endif()

    if(METALLIC_TEST_SDL_RUNTIME_TARGET)
        # A POST_BUILD copy is skipped when MSBuild considers a test executable
        # up to date, leaving CTest unable to start after a cleaned DLL. This
        # utility target is deliberately always out of date and is shared by
        # every test executable, so building any focused test target restores
        # SDL3.dll in the common configuration output directory.
        add_custom_target(MetallicTestSDLRuntime
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "$<TARGET_FILE_DIR:MetallicTaskTests>"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:${METALLIC_TEST_SDL_RUNTIME_TARGET}>"
                "$<TARGET_FILE_DIR:MetallicTaskTests>"
            DEPENDS ${METALLIC_TEST_SDL_RUNTIME_TARGET}
            COMMENT "Deploying SDL3 runtime for Metallic tests"
            VERBATIM
        )
        add_dependencies(MetallicTaskTests MetallicTestSDLRuntime)
        add_dependencies(MetallicSceneTests MetallicTestSDLRuntime)
        add_dependencies(MetallicRHITests MetallicTestSDLRuntime)
        if(TARGET MetallicNRDTests)
            add_dependencies(MetallicNRDTests MetallicTestSDLRuntime)
        endif()
    endif()
endif()

add_test(
    NAME MetallicRHITests
    COMMAND MetallicRHITests
        --rhi-no-validation
        --output-dir "$<TARGET_FILE_DIR:MetallicRHITests>/rhi-test-output"
        --gtest_color=no
)
set_tests_properties(MetallicRHITests PROPERTIES
    LABELS "rhi"
    SKIP_RETURN_CODE 77
)

if(METALLIC_HAS_DLSS_NR)
    add_test(NAME MetallicDLSSNRRuntimeTests COMMAND MetallicRHITests
        --rhi-streamline --filter dlss_nr_runtime --rhi-no-validation
        --output-dir "$<TARGET_FILE_DIR:MetallicRHITests>/dlss-nr-output")
    set_tests_properties(MetallicDLSSNRRuntimeTests PROPERTIES LABELS "rhi;dlss-nr" SKIP_RETURN_CODE 77 TIMEOUT 120)
endif()

# Pure feature negotiation and property snapshot tests; no Vulkan loader, GPU, or optional SDK required.
add_executable(MetallicVulkanDeviceFeaturesTests
    rhi/VulkanDeviceFeaturesTests.cpp
    rhi/VulkanDevicePropertiesTests.cpp
    "${CMAKE_SOURCE_DIR}/Source/Runtime/Render/GAPI/Vulkan/VulkanDeviceProperties.cpp")
target_include_directories(MetallicVulkanDeviceFeaturesTests PRIVATE "${CMAKE_SOURCE_DIR}/Source")
target_link_libraries(MetallicVulkanDeviceFeaturesTests PRIVATE metallic::volk GTest::gtest_main)
if(MSVC)
    target_compile_options(MetallicVulkanDeviceFeaturesTests PRIVATE /EHsc)
endif()
add_test(NAME MetallicVulkanDeviceFeaturesTests COMMAND MetallicVulkanDeviceFeaturesTests --gtest_color=no)
set_tests_properties(MetallicVulkanDeviceFeaturesTests PROPERTIES LABELS "rhi;device" TIMEOUT 30)

# Small host-side tests for callback ownership and loss accounting; no Vulkan device.
add_executable(MetallicShaderPrintfTests
    rhi/ShaderPrintfTests.cpp
    "${CMAKE_SOURCE_DIR}/Source/Runtime/Render/GAPI/Vulkan/VulkanShaderPrintf.cpp")
target_include_directories(MetallicShaderPrintfTests PRIVATE "${CMAKE_SOURCE_DIR}/Source")
target_link_libraries(MetallicShaderPrintfTests PRIVATE metallic::volk GTest::gtest_main)
if(MSVC)
    target_compile_options(MetallicShaderPrintfTests PRIVATE /EHsc)
endif()
add_test(NAME MetallicShaderPrintfTests COMMAND MetallicShaderPrintfTests --gtest_color=no)
set_tests_properties(MetallicShaderPrintfTests PROPERTIES LABELS "rhi;shader-printf" TIMEOUT 30)

find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    add_test(NAME MetallicShaderPrintfEvidenceTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/rhi/TestShaderPrintfEvidence.py")
    set_tests_properties(MetallicShaderPrintfEvidenceTests PROPERTIES LABELS "tools;shader-printf" TIMEOUT 30)
    add_test(NAME MetallicShaderDebuggerP3Tests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/rhi/TestShaderDebugger.py")
    set_tests_properties(MetallicShaderDebuggerP3Tests PROPERTIES LABELS "tools;shader-trace" TIMEOUT 30)

    add_test(NAME MetallicShaderTraceP2EvidenceTests
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/rhi/TestShaderTraceP2Evidence.py")
    set_tests_properties(MetallicShaderTraceP2EvidenceTests PROPERTIES LABELS "tools;shader-trace" TIMEOUT 30)
endif()

# Testbench and existing GPU/editor jobs must not contend under ctest -j.
get_property(_metallic_registered_tests DIRECTORY PROPERTY TESTS)
foreach(_test IN LISTS _metallic_registered_tests)
    get_property(_labels TEST "${_test}" PROPERTY LABELS)
    if(_labels MATCHES "(^|;)(rhi|editor)(;|$)")
        set_property(TEST "${_test}" APPEND PROPERTY RESOURCE_LOCK MetallicGPU)
    endif()
endforeach()

if(WIN32)
    # M4 bounded, legal command sequences; no optional SDKs or Python dependency.
    add_test(NAME MetallicTestbench.Property COMMAND MetallicRHITests --tb-run --tb-suite property
        --tb-validation sync --tb-repeat 4 --tb-seed 1592594996 --tb-require-all ${_testbench_layer_arguments})
    add_test(NAME MetallicTestbench.Shrinker COMMAND ${CMAKE_COMMAND}
        "-DEXECUTABLE=$<TARGET_FILE:MetallicRHITests>" "-DSOURCE=${CMAKE_SOURCE_DIR}"
        "-DLAYER_PATH=${METALLIC_TESTBENCH_LAYER_PATH}" -P "${CMAKE_CURRENT_SOURCE_DIR}/rhi/harness/VerifySequenceShrinker.cmake")
    set_tests_properties(MetallicTestbench.Property MetallicTestbench.Shrinker PROPERTIES
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}" LABELS "testbench" RESOURCE_LOCK MetallicGPU TIMEOUT 240)
    if(METALLIC_RHI_DIAGNOSTICS)
        add_test(NAME MetallicTestbench.Trace COMMAND ${CMAKE_COMMAND}
            "-DEXECUTABLE=$<TARGET_FILE:MetallicRHITests>" "-DSOURCE=${CMAKE_SOURCE_DIR}"
            "-DLAYER_PATH=${METALLIC_TESTBENCH_LAYER_PATH}" -P "${CMAKE_CURRENT_SOURCE_DIR}/rhi/harness/VerifyTrace.cmake")
        set_tests_properties(MetallicTestbench.Trace PROPERTIES WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            LABELS "testbench" RESOURCE_LOCK MetallicGPU TIMEOUT 300)
    endif()
endif()
