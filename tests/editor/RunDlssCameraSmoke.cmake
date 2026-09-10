if(NOT DEFINED EDITOR_EXECUTABLE OR NOT DEFINED TEST_DIRECTORY OR NOT DEFINED SOURCE_DIRECTORY)
    message(FATAL_ERROR "EDITOR_EXECUTABLE, TEST_DIRECTORY and SOURCE_DIRECTORY are required")
endif()

file(MAKE_DIRECTORY "${TEST_DIRECTORY}")
configure_file("${SOURCE_DIRECTORY}/Asset/StandfordBunny/scene.gltf" "${TEST_DIRECTORY}/scene.gltf" COPYONLY)
configure_file("${SOURCE_DIRECTORY}/Asset/StandfordBunny/scene.bin" "${TEST_DIRECTORY}/scene.bin" COPYONLY)
set(ENV{METALLIC_SMOKE_TEST_SAMPLE} pathtracing-sample-dlss-rr)
set(ENV{METALLIC_SMOKE_TEST_DLSS_CAMERA} 1)
set(ENV{METALLIC_DEBUG_VALIDATION} 1)
execute_process(
    COMMAND "${EDITOR_EXECUTABLE}" --smoke-test
        --scene "${TEST_DIRECTORY}/scene.gltf"
    WORKING_DIRECTORY "${TEST_DIRECTORY}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE errors
    TIMEOUT 75
)
set(log "${output}\n${errors}")
file(WRITE "${TEST_DIRECTORY}/editor.log" "${log}")
if(NOT result STREQUAL "0" OR log MATCHES "Vulkan validation:|\\[error\\]")
    message(FATAL_ERROR "DLSS camera smoke test failed (${result}):\n${log}")
endif()
if(NOT log MATCHES "\\[Smoke DLSS Camera\\] Passed")
    message(FATAL_ERROR "DLSS camera smoke test did not complete:\n${log}")
endif()
message(STATUS "DLSS camera smoke test passed without Vulkan validation messages")
