if(NOT DEFINED LOOKDEV_EXECUTABLE OR NOT DEFINED TEST_DIRECTORY)
    message(FATAL_ERROR "LOOKDEV_EXECUTABLE and TEST_DIRECTORY are required")
endif()
file(MAKE_DIRECTORY "${TEST_DIRECTORY}")
set(ENV{METALLIC_SMOKE_TEST_SCENE_SWITCH} 1)
set(ENV{METALLIC_DEBUG_VALIDATION} 1)
set(ENV{METALLIC_VK_INTERNAL_PIPELINE_CACHE} disabled)
execute_process(
    COMMAND "${LOOKDEV_EXECUTABLE}" --sample lookdev-abeautiful-game --smoke-test --debug-control
    WORKING_DIRECTORY "${TEST_DIRECTORY}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors TIMEOUT 150
)
set(log "${output}\n${errors}")
file(WRITE "${TEST_DIRECTORY}/editor.log" "${log}")
if(NOT result STREQUAL "0" OR log MATCHES "Vulkan validation:|\\[error\\]" OR
    NOT log MATCHES "\\[Smoke Scene Switch\\] Passed")
    message(FATAL_ERROR "LookDev scene switch smoke failed (${result}):\n${log}")
endif()
message(STATUS "LookDev scene switch smoke passed without Vulkan validation messages")
