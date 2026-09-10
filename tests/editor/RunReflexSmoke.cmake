if(NOT DEFINED EDITOR_EXECUTABLE OR NOT DEFINED TEST_DIRECTORY OR NOT DEFINED REFLEX_MODE OR
   NOT DEFINED SOURCE_DIRECTORY)
    message(FATAL_ERROR "EDITOR_EXECUTABLE, TEST_DIRECTORY, REFLEX_MODE and SOURCE_DIRECTORY are required")
endif()

file(MAKE_DIRECTORY "${TEST_DIRECTORY}")
configure_file("${SOURCE_DIRECTORY}/Asset/StandfordBunny/scene.gltf" "${TEST_DIRECTORY}/scene.gltf" COPYONLY)
configure_file("${SOURCE_DIRECTORY}/Asset/StandfordBunny/scene.bin" "${TEST_DIRECTORY}/scene.bin" COPYONLY)
set(ENV{METALLIC_SMOKE_TEST_REFLEX} 1)
set(ENV{METALLIC_SMOKE_TEST_FRAMES} 128)
set(ENV{METALLIC_REFLEX_MODE} "${REFLEX_MODE}")
set(ENV{METALLIC_DEBUG_VALIDATION} 1)
execute_process(
    COMMAND "${EDITOR_EXECUTABLE}" --smoke-test --debug-control --scene "${TEST_DIRECTORY}/scene.gltf"
    WORKING_DIRECTORY "${TEST_DIRECTORY}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors TIMEOUT 160)
set(log "${output}\n${errors}")
file(WRITE "${TEST_DIRECTORY}/editor.log" "${log}")
if(log MATCHES "Vulkan validation:|\\[error\\]|\\[Reflex\\].*failed")
    message(FATAL_ERROR "Reflex ${REFLEX_MODE} smoke reported errors:\n${log}")
endif()
if(result STREQUAL "77")
    message(STATUS "Reflex unavailable; skipping")
elseif(NOT result STREQUAL "0" OR NOT log MATCHES "\\[Smoke Reflex\\] available=true, report=true")
    message(FATAL_ERROR "Reflex ${REFLEX_MODE} smoke failed (${result}):\n${log}")
else()
    message(STATUS "Reflex ${REFLEX_MODE}: driver returned complete frame timings without validation errors")
endif()
