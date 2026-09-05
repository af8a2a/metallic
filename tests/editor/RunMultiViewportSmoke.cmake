if(NOT DEFINED EDITOR_EXECUTABLE OR NOT DEFINED TEST_DIRECTORY)
    message(FATAL_ERROR "EDITOR_EXECUTABLE and TEST_DIRECTORY are required")
endif()

file(MAKE_DIRECTORY "${TEST_DIRECTORY}")
set(ENV{METALLIC_SMOKE_TEST_VIEWPORTS} 1)
set(ENV{METALLIC_DEBUG_VALIDATION} 1)
if(FINAL_BLIT)
    set(ENV{METALLIC_SMOKE_TEST_FINAL_BLIT} 1)
endif()
execute_process(
    COMMAND "${EDITOR_EXECUTABLE}" --smoke-test --debug-control
    WORKING_DIRECTORY "${TEST_DIRECTORY}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE errors
    TIMEOUT 75
)
set(log "${output}\n${errors}")
file(WRITE "${TEST_DIRECTORY}/editor.log" "${log}")
if(NOT result STREQUAL "0" OR log MATCHES "Vulkan validation:|\\[error\\]")
    message(FATAL_ERROR "Multi-viewport smoke test failed (${result}):\n${log}")
endif()
if(NOT log MATCHES "\\[Smoke Viewports\\] Passed")
    message(FATAL_ERROR "Multi-viewport smoke test did not complete:\n${log}")
endif()
if(FINAL_BLIT AND NOT log MATCHES "\\[Smoke FinalBlit\\] Passed")
    message(FATAL_ERROR "FinalBlit smoke test did not complete:\n${log}")
endif()
message(STATUS "Editor smoke test passed without Vulkan validation messages")
