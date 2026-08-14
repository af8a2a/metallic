find_package(Python3 QUIET COMPONENTS Interpreter)

if(Python3_Interpreter_FOUND)
    set(METALLIC_TERRAIN_P0_SOURCE
        "${CMAKE_SOURCE_DIR}/Asset/simple_terrain_height.exr")
    set(METALLIC_TERRAIN_P0_DIRECTORY
        "${CMAKE_SOURCE_DIR}/Asset/MeshletCache/TerrainP0")
    set(METALLIC_TERRAIN_P0_GLTF
        "${METALLIC_TERRAIN_P0_DIRECTORY}/simple_terrain_height.gltf")
    set(METALLIC_TERRAIN_P0_BUFFER
        "${METALLIC_TERRAIN_P0_DIRECTORY}/simple_terrain_height.bin")
    set(METALLIC_TERRAIN_P0_STREAMASSET
        "${METALLIC_TERRAIN_P0_GLTF}.meshstream.bin")

    add_custom_command(
        OUTPUT
            "${METALLIC_TERRAIN_P0_GLTF}"
            "${METALLIC_TERRAIN_P0_BUFFER}"
        COMMAND ${CMAKE_COMMAND} -E make_directory
            "${METALLIC_TERRAIN_P0_DIRECTORY}"
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/BuildTerrainMesh.py"
            "${METALLIC_TERRAIN_P0_SOURCE}"
            --output "${METALLIC_TERRAIN_P0_GLTF}"
            --binary-output "${METALLIC_TERRAIN_P0_BUFFER}"
            --horizontal-size 256
            --height-scale 16
            --skirt-depth 8
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/BuildTerrainMesh.py"
            "${METALLIC_TERRAIN_P0_SOURCE}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Converting the Houdini EXR height field to Terrain P0 glTF"
        VERBATIM
    )

    add_custom_command(
        OUTPUT "${METALLIC_TERRAIN_P0_STREAMASSET}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SDL3::SDL3>"
            "$<TARGET_FILE_DIR:Metallic>"
        COMMAND "$<TARGET_FILE:Metallic>"
            --build-meshstream "${METALLIC_TERRAIN_P0_GLTF}"
            --output "${METALLIC_TERRAIN_P0_STREAMASSET}"
        DEPENDS
            Metallic
            "${METALLIC_TERRAIN_P0_GLTF}"
            "${METALLIC_TERRAIN_P0_BUFFER}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Building the Terrain P0 MeshletStreamAsset"
        VERBATIM
    )

    add_custom_target(MetallicTerrainP0
        DEPENDS "${METALLIC_TERRAIN_P0_STREAMASSET}"
    )

    add_custom_target(MetallicTerrainP0Smoke
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:SDL3::SDL3>"
            "$<TARGET_FILE_DIR:MetallicGPUDrivenSample>"
        COMMAND "$<TARGET_FILE:MetallicGPUDrivenSample>"
            --terrain-p0
            --smoke-test
        DEPENDS
            MetallicGPUDrivenSample
            MetallicTerrainP0
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Rendering the Terrain P0 StreamAsset smoke frame"
        VERBATIM
    )
else()
    message(STATUS "Python3 was not found; Terrain P0 build targets are unavailable")
endif()
