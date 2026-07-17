include(GoogleTest)

add_executable(MetallicTaskTests
    task/task_graph_tests.cpp
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
    rhi/HistoryResourcesTests.cpp
    rhi/RenderGraphTests.cpp
    rhi/RenderingTests.cpp
    rhi/ResourceTests.cpp
    rhi/SceneRtxTests.cpp
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

    if(TARGET SDL3-shared)
        add_custom_command(TARGET MetallicRhiTests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SDL3-shared>"
                "$<TARGET_FILE_DIR:MetallicRhiTests>"
        )
    elseif(TARGET SDL3::SDL3-shared)
        add_custom_command(TARGET MetallicRhiTests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SDL3::SDL3-shared>"
                "$<TARGET_FILE_DIR:MetallicRhiTests>"
        )
    endif()

    metallic_copy_streamline_runtime(MetallicRhiTests)
    metallic_copy_aftermath_runtime(MetallicRhiTests)
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
