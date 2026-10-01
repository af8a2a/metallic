if(NOT DEFINED EDITOR_EXECUTABLE OR NOT DEFINED TEST_DIRECTORY OR NOT DEFINED CASE)
    message(FATAL_ERROR "EDITOR_EXECUTABLE, TEST_DIRECTORY and CASE are required")
endif()

file(MAKE_DIRECTORY "${TEST_DIRECTORY}")
set(ENV{METALLIC_SMOKE_TEST_HIDDEN} 1)
set(ENV{METALLIC_DEBUG_VALIDATION} 0)
set(ENV{METALLIC_NSIGHT_GRAPHICS_CAPTURE} 0) # The explicit alias must win.
unset(ENV{METALLIC_FULL_ROAM_OUTPUT})
unset(ENV{METALLIC_NSIGHT_GPU_TRACE_METRICS})
unset(ENV{METALLIC_SMOKE_TEST_NSIGHT_CAPTURE})
set(arguments --nsight-gputrace --smoke-test)

if(CASE STREQUAL "disabled")
    # A retained legacy CMake default and metrics settings must not opt the app
    # into Nsight. Deliberately use an invalid path to detect accidental startup.
    unset(ENV{METALLIC_NSIGHT_GRAPHICS_CAPTURE})
    set(ENV{METALLIC_NSIGHT_GPU_TRACE_METRICS} "${TEST_DIRECTORY}/MissingMetrics.json")
    set(arguments --smoke-test)
    set(timeout 180)
elseif(CASE STREQUAL "export")
    set(ENV{METALLIC_SMOKE_TEST_NSIGHT_CAPTURE} 1)
    set(timeout 180)
elseif(CASE STREQUAL "collection")
    set(arguments --nsight-capture --smoke-test)
    set(ENV{METALLIC_SMOKE_TEST_NSIGHT_CAPTURE} 1)
    # Collection selects its own metrics and must ignore live-trace overrides.
    set(ENV{METALLIC_NSIGHT_GPU_TRACE_METRICS} "${TEST_DIRECTORY}/MissingMetrics.json")
    set(timeout 600)
elseif(CASE STREQUAL "invalid-metrics")
    # Deliberately reject the host configuration. Before the startup guard this
    # can leave the target blocked inside Nsight on its first queue submission.
    set(config "${TEST_DIRECTORY}/InvalidMetrics.json")
    file(WRITE "${config}" "[{\"architecture\":\"Turing\",\"metric-set-name\":\"Metallic invalid metric set\"}]")
    set(ENV{METALLIC_NSIGHT_GPU_TRACE_METRICS} "${config}")
    set(timeout 75)
else()
    message(FATAL_ERROR "Unknown CASE=${CASE}")
endif()

execute_process(
    COMMAND "${EDITOR_EXECUTABLE}" ${arguments}
    WORKING_DIRECTORY "${TEST_DIRECTORY}"
    RESULT_VARIABLE result
    OUTPUT_FILE "${TEST_DIRECTORY}/stdout.log"
    ERROR_FILE "${TEST_DIRECTORY}/stderr.log"
    TIMEOUT ${timeout}
)
file(READ "${TEST_DIRECTORY}/stdout.log" output)
file(READ "${TEST_DIRECTORY}/stderr.log" errors)
file(WRITE "${TEST_DIRECTORY}/LookDev.log" "${output}\n${errors}")
if(CASE STREQUAL "disabled")
    if(NOT result STREQUAL "0" OR NOT output MATCHES "nsightCapture=false" OR
       NOT output MATCHES "Presented editor frame" OR output MATCHES "Nsight GPU Trace startup handshake")
        message(FATAL_ERROR "Ordinary startup must render without Nsight (${result}); see ${TEST_DIRECTORY}/LookDev.log")
    endif()
elseif(CASE STREQUAL "collection")
    if(NOT result STREQUAL "0" OR NOT output MATCHES "Replay GPU Trace completed:" OR
       NOT output MATCHES "\\[Smoke Nsight\\] Capture 3.*\\.ngfx-capture" OR
       output MATCHES "Nsight GPU Trace startup handshake")
        message(FATAL_ERROR "Capture/replay collection failed (${result}); see ${TEST_DIRECTORY}/LookDev.log")
    endif()
elseif(CASE STREQUAL "export")
    if(NOT result STREQUAL "0" OR NOT output MATCHES "End Nsight GPU Trace startup handshake" OR
       NOT output MATCHES "\\[Smoke Nsight\\] Capture 3.*\\.ngfx-gputrace")
        message(FATAL_ERROR "GPU Trace startup/export failed (${result}); see ${TEST_DIRECTORY}/LookDev.log")
    endif()
elseif(NOT result STREQUAL "1" OR NOT "${output}\n${errors}" MATCHES "[Nn]sight startup")
    message(FATAL_ERROR "Invalid metrics must exit 1 with a startup diagnostic, not hang (${result}); see ${TEST_DIRECTORY}/LookDev.log")
endif()
message(STATUS "Nsight GPU Trace ${CASE} regression passed")
