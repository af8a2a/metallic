add_executable(MetallicRhiTests
    rhi/main.cpp
    rhi/rhi_test.h
    rhi/command_submission_tests.cpp
    rhi/rendering_tests.cpp
    rhi/resource_tests.cpp
    rhi/validation_tests.cpp
)

target_include_directories(MetallicRhiTests PRIVATE
    "${CMAKE_SOURCE_DIR}/Source"
    "${CMAKE_CURRENT_SOURCE_DIR}/rhi"
)

target_link_libraries(MetallicRhiTests PRIVATE
    MetallicRuntimeRender
)

target_compile_definitions(MetallicRhiTests PRIVATE
    PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

if(WIN32)
    target_compile_definitions(MetallicRhiTests PRIVATE NOMINMAX)

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
endif()

add_test(NAME MetallicRhiTests COMMAND MetallicRhiTests --rhi-no-validation)
set_tests_properties(MetallicRhiTests PROPERTIES
    LABELS "rhi"
    SKIP_RETURN_CODE 77
)
