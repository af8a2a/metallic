if(NOT DEFINED LOOKDEV_EXECUTABLE OR NOT DEFINED TEST_DIRECTORY OR NOT DEFINED SOURCE_DIRECTORY)
    message(FATAL_ERROR "LOOKDEV_EXECUTABLE, TEST_DIRECTORY and SOURCE_DIRECTORY are required")
endif()

file(MAKE_DIRECTORY "${TEST_DIRECTORY}/scene")
set(scene_directory "${TEST_DIRECTORY}/scene")
set(source_scene_directory "${SOURCE_DIRECTORY}/Asset/LookDev/OpenPbrDefault")
# Save/reload exercises an isolated document; source-controlled LookDev assets
# and the user's material sidecars must never be modified by this smoke test.
file(READ "${source_scene_directory}/OpenPbrDefault.gltf" scene_json)
# An unused second material exercises handoff between two simultaneous panels
# without adding geometry or changing the shaderball rendered by either path.
string(JSON second_material GET "${scene_json}" materials 0)
string(JSON second_material SET "${second_material}" name "\"Inspector handoff material\"")
string(JSON scene_json SET "${scene_json}" materials 1 "${second_material}")
file(WRITE "${scene_directory}/OpenPbrDefault.gltf" "${scene_json}\n")
configure_file("${source_scene_directory}/Shaderball.bin" "${scene_directory}/Shaderball.bin" COPYONLY)
file(REMOVE "${scene_directory}/OpenPbrDefault.metallic_scene.json")

set(ENV{METALLIC_SMOKE_TEST_MATERIAL_INSPECTOR} 1)
set(ENV{METALLIC_DEBUG_VALIDATION} 1)
execute_process(
    COMMAND "${LOOKDEV_EXECUTABLE}" --sample lookdev-vbuffer
        --scene "${scene_directory}/OpenPbrDefault.gltf" --smoke-test --debug-control
    WORKING_DIRECTORY "${TEST_DIRECTORY}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors TIMEOUT 75
)
set(log "${output}\n${errors}")
file(WRITE "${TEST_DIRECTORY}/editor.log" "${log}")
if(NOT result STREQUAL "0" OR log MATCHES "Vulkan validation:|\\[error\\]" OR
    NOT log MATCHES "\\[Smoke Material Inspector\\] Passed")
    message(FATAL_ERROR "LookDev material inspector smoke failed (${result}):\n${log}")
endif()
message(STATUS "LookDev material inspector smoke passed without Vulkan validation messages")
