if(NOT DEFINED LOOKDEV_EXECUTABLE OR NOT DEFINED TEST_DIRECTORY)
    message(FATAL_ERROR "LOOKDEV_EXECUTABLE and TEST_DIRECTORY are required")
endif()
file(MAKE_DIRECTORY "${TEST_DIRECTORY}")
set(ENV{METALLIC_SMOKE_TEST_PAINTER_SWITCH} 1)
set(ENV{METALLIC_SMOKE_TEST_HIDDEN} 1)
set(ENV{METALLIC_DEBUG_VALIDATION} 1)
execute_process(
    COMMAND "${LOOKDEV_EXECUTABLE}" --sample painter-M01_NeutralDielectric-uniform
        --smoke-test --debug-control
    WORKING_DIRECTORY "${TEST_DIRECTORY}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors TIMEOUT 600
)
set(log "${output}\n${errors}")
file(WRITE "${TEST_DIRECTORY}/editor.log" "${log}")
# The loader can report stale optional overlay/ICD manifests before selecting a
# working driver. Keep those diagnostics in editor.log; reject device validation.
string(REGEX REPLACE "[^\n]*Vulkan validation: loader_get_json:[^\n]*(\n|$)" "" validationLog "${log}")
if(NOT result STREQUAL "0" OR validationLog MATCHES "Vulkan validation:|\\[error\\]" OR
    NOT log MATCHES "\\[Smoke Painter Switch\\] Passed")
    message(FATAL_ERROR "Painter scene switch failed (${result}); see ${TEST_DIRECTORY}/editor.log")
endif()
if(NOT log MATCHES "Shader warmup: [0-9]+ requests, [0-9]+ existing cache hits, 0 failures" OR
    log MATCHES "\\[Slang\\] Begin compile")
    message(FATAL_ERROR "Painter runtime shader request was not prewarmed; see ${TEST_DIRECTORY}/editor.log")
endif()
message(STATUS "Painter scene switching passed")
