include(GoogleTest)

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
    rhi/Main.cpp
    rhi/RhiTest.h
    rhi/BindlessBufferTests.cpp
    rhi/CommandSubmissionTests.cpp
    rhi/GPUSceneSubsystemTests.cpp
    rhi/HistoryResourcesTests.cpp
    rhi/RenderGraphTests.cpp
    rhi/RenderingTests.cpp
    rhi/ResourceTests.cpp
    rhi/RayTracingAccelerationStructureTests.cpp
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
