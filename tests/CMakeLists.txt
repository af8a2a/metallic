include(GoogleTest)

if(METALLIC_ENABLE_NRD)
    add_executable(MetallicNrdTests rhi/NrdTests.cpp)
    target_link_libraries(MetallicNrdTests PRIVATE MetallicRuntimeRender MathLib GTest::gtest_main)
    target_include_directories(MetallicNrdTests PRIVATE "${CMAKE_SOURCE_DIR}")
    target_compile_definitions(MetallicNrdTests PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    if(WIN32)
        target_compile_definitions(MetallicNrdTests PRIVATE NOMINMAX)
        set(_nrd_test_slang_root "${CMAKE_SOURCE_DIR}/External/slang")
        if(DEFINED SLANG_ROOT)
            set(_nrd_test_slang_root "${SLANG_ROOT}")
        endif()
        file(GLOB _nrd_test_slang_dlls "${_nrd_test_slang_root}/bin/*.dll")
        foreach(_dll IN LISTS _nrd_test_slang_dlls)
            add_custom_command(TARGET MetallicNrdTests POST_BUILD COMMAND ${CMAKE_COMMAND} -E
                copy_if_different "${_dll}" "$<TARGET_FILE_DIR:MetallicNrdTests>")
        endforeach()
        metallic_copy_spdlog_runtime(MetallicNrdTests)
        metallic_copy_streamline_runtime(MetallicNrdTests)
    endif()
    add_test(NAME MetallicNrdTests COMMAND MetallicNrdTests --gtest_color=no)
    set_tests_properties(MetallicNrdTests PROPERTIES LABELS "rhi;nrd" TIMEOUT 300)
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
    add_test(NAME MetallicDlssNrConfiguration
        COMMAND ${CMAKE_COMMAND}
            "-DSOURCE_DIRECTORY=${CMAKE_SOURCE_DIR}"
            "-DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/dlss-nr-configuration"
            "-DTEST_GENERATOR=${CMAKE_GENERATOR}"
            "-DTEST_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/DlssNrConfigurationTests.cmake")
    set_tests_properties(MetallicDlssNrConfiguration PROPERTIES LABELS "build;dlss-nr" TIMEOUT 120)
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
        add_test(NAME MetallicEditorDlssCameraSmoke
            COMMAND ${CMAKE_COMMAND}
                "-DEDITOR_EXECUTABLE=$<TARGET_FILE:Metallic>"
                "-DTEST_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}/editor-dlss-camera"
                "-DSOURCE_DIRECTORY=${CMAKE_SOURCE_DIR}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/editor/RunDlssCameraSmoke.cmake"
        )
        set_tests_properties(MetallicEditorDlssCameraSmoke PROPERTIES
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

add_executable(MetallicDebugTests debug/DebugCoreTests.cpp)
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

add_executable(MetallicRhiTests
    rhi/DebugControlTests.cpp
    rhi/Main.cpp
    rhi/RhiTest.h
    rhi/BindlessBufferTests.cpp
    rhi/DescriptorHeapCodePatternTests.cpp
    rhi/DlssMotionVectorTests.cpp
    rhi/DlssNrTests.cpp
    rhi/AutoExposureTests.cpp
    rhi/OpenPbrLookDevTests.cpp
    rhi/SliderDebugTests.cpp
    rhi/VisibilityBufferDeferredTests.cpp
    rhi/MaterialBinningTests.cpp
    rhi/DeferredTransmissionTests.cpp
    rhi/ClusterLightGridTests.cpp
    rhi/CommandSubmissionTests.cpp
    rhi/FrameContextTests.cpp
    rhi/FinalBlitTests.cpp
    rhi/GPUSceneSubsystemTests.cpp
    rhi/HistoryResourcesTests.cpp
    rhi/ImportedVirtualLightTests.cpp
    rhi/LightGridDebugPassTests.cpp
    rhi/PhotometricLightingTests.cpp
    rhi/ShaderToHumanTests.cpp
    rhi/ShaderHotReloadTests.cpp
    rhi/SphericalHarmonicsTests.cpp
    rhi/ReGIRVirtualLightTests.cpp
    rhi/RenderGraphTests.cpp
    rhi/SceneBindingTests.cpp
    rhi/RadianceCacheTests.cpp
    rhi/RenderingTests.cpp
    rhi/ResourceTests.cpp
    rhi/RayTracingAccelerationStructureTests.cpp
    rhi/RayTracingPositionFetchTests.cpp
    rhi/StreamerTests.cpp
    rhi/TestOutput.cpp
    rhi/ValidationTests.cpp
    rhi/stb/Stb.cpp
)

target_include_directories(MetallicRhiTests PRIVATE
    "${CMAKE_SOURCE_DIR}/Source"
    "${CMAKE_CURRENT_SOURCE_DIR}/rhi"
)

target_link_libraries(MetallicRhiTests PRIVATE
    metallic::sdl3
    MetallicRuntimeRender
    GTest::gtest
)

target_compile_definitions(MetallicRhiTests PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

if(WIN32)
    target_compile_definitions(MetallicRhiTests PRIVATE NOMINMAX)
    target_compile_options(MetallicRhiTests PRIVATE /EHsc)

    set(METALLIC_TEST_SLANG_ROOT "${CMAKE_SOURCE_DIR}/External/slang")
    if(DEFINED SLANG_ROOT)
        set(METALLIC_TEST_SLANG_ROOT "${SLANG_ROOT}")
    endif()

    file(GLOB METALLIC_TEST_SLANG_RUNTIME_DLLS "${METALLIC_TEST_SLANG_ROOT}/bin/*.dll")
    foreach(METALLIC_TEST_SLANG_RUNTIME_DLL IN LISTS METALLIC_TEST_SLANG_RUNTIME_DLLS)
        add_custom_command(TARGET MetallicRhiTests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${METALLIC_TEST_SLANG_RUNTIME_DLL}"
                "$<TARGET_FILE_DIR:MetallicRhiTests>"
        )
    endforeach()

    metallic_copy_streamline_runtime(MetallicRhiTests)
    metallic_copy_dlss_nr_runtime(MetallicRhiTests)
    metallic_copy_aftermath_runtime(MetallicRhiTests)
    metallic_copy_nrc_runtime(MetallicRhiTests)
endif()
metallic_copy_openusd_runtime(MetallicRhiTests)
metallic_copy_spdlog_runtime(MetallicRhiTests)

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
        add_custom_target(MetallicTestSdlRuntime
            COMMAND ${CMAKE_COMMAND} -E make_directory
                "$<TARGET_FILE_DIR:MetallicTaskTests>"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:${METALLIC_TEST_SDL_RUNTIME_TARGET}>"
                "$<TARGET_FILE_DIR:MetallicTaskTests>"
            DEPENDS ${METALLIC_TEST_SDL_RUNTIME_TARGET}
            COMMENT "Deploying SDL3 runtime for Metallic tests"
            VERBATIM
        )
        add_dependencies(MetallicTaskTests MetallicTestSdlRuntime)
        add_dependencies(MetallicSceneTests MetallicTestSdlRuntime)
        add_dependencies(MetallicRhiTests MetallicTestSdlRuntime)
    endif()
endif()

add_test(
    NAME MetallicRhiTests
    COMMAND MetallicRhiTests
        --rhi-no-validation
        --output-dir "$<TARGET_FILE_DIR:MetallicRhiTests>/rhi-test-output"
        --gtest_color=no
)
set_tests_properties(MetallicRhiTests PROPERTIES
    LABELS "rhi"
    SKIP_RETURN_CODE 77
)

if(METALLIC_HAS_DLSS_NR)
    add_test(NAME MetallicDlssNrRuntimeTests COMMAND MetallicRhiTests
        --rhi-streamline --filter dlss_nr_runtime --rhi-no-validation
        --output-dir "$<TARGET_FILE_DIR:MetallicRhiTests>/dlss-nr-output")
    set_tests_properties(MetallicDlssNrRuntimeTests PROPERTIES LABELS "rhi;dlss-nr" SKIP_RETURN_CODE 77 TIMEOUT 120)
endif()
