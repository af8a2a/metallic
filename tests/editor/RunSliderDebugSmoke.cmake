if(NOT DEFINED LOOKDEV_EXECUTABLE OR NOT DEFINED TEST_DIRECTORY)
    message(FATAL_ERROR "LOOKDEV_EXECUTABLE and TEST_DIRECTORY are required")
endif()
file(MAKE_DIRECTORY "${TEST_DIRECTORY}")
set(ENV{METALLIC_SMOKE_TEST_SLIDER} 1)
set(ENV{METALLIC_DEBUG_VALIDATION} 1)
execute_process(
    COMMAND "${LOOKDEV_EXECUTABLE}" --sample lookdev-shading-compare --smoke-test --debug-control
    WORKING_DIRECTORY "${TEST_DIRECTORY}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors TIMEOUT 75
)
set(log "${output}\n${errors}")
file(WRITE "${TEST_DIRECTORY}/editor.log" "${log}")
if(NOT result STREQUAL "0" OR log MATCHES "Vulkan validation:|\\[error\\]" OR
    NOT log MATCHES "\\[Smoke Slider\\] Passed")
    message(FATAL_ERROR "LookDev Slider smoke test failed (${result}):\n${log}")
endif()
message(STATUS "LookDev Slider smoke test passed without Vulkan validation messages")
