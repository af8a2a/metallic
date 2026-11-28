if(WIN32)
    set(METALLIC_TRACY_VIEWER "" CACHE FILEPATH
        "Tracy Viewer executable (empty uses build-tracy-viewer in the source tree)")
    add_executable(MetallicTracy TracyLauncher.cpp)
    target_compile_definitions(MetallicTracy PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
    if(MSVC)
        target_compile_options(MetallicTracy PRIVATE /utf-8)
    endif()
    # Generate per configuration without making the small launcher depend on
    # rebuilding the editor or Tracy's independently built Release viewer.
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/tracy/$<CONFIG>/TracyLauncherConfig.h"
        CONTENT "#pragma once
inline constexpr wchar_t kRepositoryRoot[] = LR\"metallic(${PROJECT_SOURCE_DIR})metallic\";
inline constexpr wchar_t kMetallicExecutable[] = LR\"metallic($<TARGET_FILE:Metallic>)metallic\";
inline constexpr wchar_t kViewerOverride[] = LR\"metallic(${METALLIC_TRACY_VIEWER})metallic\";
")
    target_include_directories(MetallicTracy PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/tracy/$<CONFIG>")
    set_target_properties(MetallicTracy PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif()

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
    set(METALLIC_TERRAIN_HEIGHT_ARGS --horizontal-size 256 --height-scale 16 --skirt-depth 8)

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
            ${METALLIC_TERRAIN_HEIGHT_ARGS}
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
        COMMAND ${CMAKE_COMMAND} -E env
            METALLIC_SMOKE_TEST_HIDDEN=1
            METALLIC_SMOKE_TEST_SAMPLE=gpu-driven-terrain-p0
            "$<TARGET_FILE:Metallic>" --smoke-test
        DEPENDS MetallicTerrainP0
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Rendering the Terrain P0 StreamAsset smoke frame"
        VERBATIM
    )

    add_custom_target(MetallicTerrainP1Smoke
        COMMAND ${CMAKE_COMMAND} -E env
            METALLIC_SMOKE_TEST_HIDDEN=1
            METALLIC_SMOKE_TEST_SAMPLE=gpu-driven-terrain-p1-unified
            "$<TARGET_FILE:Metallic>" --smoke-test
        DEPENDS MetallicTerrainP0
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Rendering the Terrain P1 unified GPUScene/StreamAsset smoke frame"
        VERBATIM
    )

    function(add_metallic_terrain_block_height output tiles_per_axis)
        cmake_parse_arguments(PARSE_ARGV 2 terrain "" "HEIGHT_FIELD" "")
        set(terrain_args)
        if(terrain_HEIGHT_FIELD)
            set(terrain_args --height-field-output "${terrain_HEIGHT_FIELD}")
        endif()
        add_custom_command(
            OUTPUT "${output}.gltf" "${output}.bin" ${terrain_HEIGHT_FIELD}
            COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/BuildTerrainMesh.py"
                "${METALLIC_TERRAIN_P0_SOURCE}" --output "${output}.gltf"
                ${METALLIC_TERRAIN_HEIGHT_ARGS} --tiles-per-axis ${tiles_per_axis} ${terrain_args}
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/BuildTerrainMesh.py" "${METALLIC_TERRAIN_P0_SOURCE}"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            COMMENT "Converting the Houdini EXR height field to ${tiles_per_axis}x${tiles_per_axis} terrain blocks"
            VERBATIM
        )
    endfunction()

    function(add_metallic_terrain_vsm_sample bake_target smoke_target sample_id terrain_source output)
        cmake_parse_arguments(PARSE_ARGV 5 terrain "" "WEIGHT_MAP;HEIGHT_FIELD" "")
        set(lighting "${CMAKE_SOURCE_DIR}/Asset/Terrain/TerrainVSM.metallic_scene.json")
        set(graph "${CMAKE_SOURCE_DIR}/Pipelines/Samples/gpu_driven_terrain_vsm.metallic_graph.json")
        set(stream_asset "${output}.gltf.meshstream.bin")
        set(layer_args)
        set(layer_outputs)
        set(layer_dependencies)
        if(terrain_WEIGHT_MAP)
            set(layer_args --weight-map "${terrain_WEIGHT_MAP}")
            set(layer_outputs "${output}.material" "${output}.materialdef" "${output}_weight.png"
                "${output}_grass_color.png" "${output}_grass_normal.png"
                "${output}_rock_color.png" "${output}_rock_normal.png")
            set(layer_dependencies "${terrain_WEIGHT_MAP}" "${CMAKE_CURRENT_SOURCE_DIR}/BuildTerrainLayerTextures.py")
        endif()
        if(terrain_HEIGHT_FIELD)
            list(APPEND layer_args --height-field "${terrain_HEIGHT_FIELD}")
            list(APPEND layer_dependencies "${terrain_HEIGHT_FIELD}")
        endif()
        add_custom_command(
            OUTPUT "${output}.gltf" "${output}.bin" "${output}.metallic_scene.json" ${layer_outputs}
            COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/BuildTerrainVSMSample.py"
                "${terrain_source}.gltf" "${lighting}" "${graph}" "${output}.gltf" ${layer_args}
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/BuildTerrainVSMSample.py"
                "${CMAKE_CURRENT_SOURCE_DIR}/BuildTerrainMesh.py"
                "${terrain_source}.gltf" "${lighting}" "${graph}" ${layer_dependencies}
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            COMMENT "Assembling ${sample_id} with opaque mesh props"
            VERBATIM
        )
        add_custom_command(
            OUTPUT "${stream_asset}"
            COMMAND "$<TARGET_FILE:Metallic>" --build-meshstream "${output}.gltf" --output "${stream_asset}"
            DEPENDS Metallic "${output}.gltf" "${output}.bin" "${terrain_source}.bin"
                "${output}.metallic_scene.json" ${layer_outputs}
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            COMMENT "Building the ${sample_id} MeshletStreamAsset"
            VERBATIM
        )
        add_custom_target(${bake_target} DEPENDS "${stream_asset}" "${output}.metallic_scene.json")
        add_custom_target(${smoke_target}
            COMMAND ${CMAKE_COMMAND} -E env METALLIC_SMOKE_TEST_HIDDEN=1
                "METALLIC_SMOKE_TEST_SAMPLE=${sample_id}" "$<TARGET_FILE:Metallic>" --smoke-test
            DEPENDS ${bake_target}
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            COMMENT "Rendering the ${sample_id} smoke frame"
            VERBATIM
        )
    endfunction()

    add_metallic_terrain_vsm_sample(MetallicTerrainVSMSample MetallicTerrainVSMSmoke
        gpu-driven-terrain-vsm "${METALLIC_TERRAIN_P0_DIRECTORY}/simple_terrain_height"
        "${CMAKE_SOURCE_DIR}/Asset/MeshletCache/TerrainVSM/terrain_vsm")

    set(METALLIC_TERRAIN_BLOCKS_DIRECTORY "${CMAKE_SOURCE_DIR}/Asset/MeshletCache/TerrainBlocks")
    set(METALLIC_TERRAIN_BLOCKS_HEIGHT "${METALLIC_TERRAIN_BLOCKS_DIRECTORY}/terrain_blocks_height")
    add_metallic_terrain_block_height("${METALLIC_TERRAIN_BLOCKS_HEIGHT}" 2)
    add_metallic_terrain_vsm_sample(MetallicTerrainBlocksVSMSample MetallicTerrainBlocksVSMSmoke
        gpu-driven-terrain-blocks-vsm "${METALLIC_TERRAIN_BLOCKS_HEIGHT}"
        "${METALLIC_TERRAIN_BLOCKS_DIRECTORY}/terrain_blocks")

    set(METALLIC_TERRAIN_BLOCKS_4X4_DIRECTORY "${CMAKE_SOURCE_DIR}/Asset/MeshletCache/TerrainBlocks4x4")
    set(METALLIC_TERRAIN_BLOCKS_4X4_HEIGHT "${METALLIC_TERRAIN_BLOCKS_4X4_DIRECTORY}/terrain_blocks_4x4_height")
    add_metallic_terrain_block_height("${METALLIC_TERRAIN_BLOCKS_4X4_HEIGHT}" 4)
    add_metallic_terrain_vsm_sample(MetallicTerrainBlocks4x4VSMSample MetallicTerrainBlocks4x4VSMSmoke
        gpu-driven-terrain-blocks-4x4-vsm "${METALLIC_TERRAIN_BLOCKS_4X4_HEIGHT}"
        "${METALLIC_TERRAIN_BLOCKS_4X4_DIRECTORY}/terrain_blocks_4x4")

    set(METALLIC_TERRAIN_BLOCKS_8X8_DIRECTORY "${CMAKE_SOURCE_DIR}/Asset/MeshletCache/TerrainBlocks8x8")
    set(METALLIC_TERRAIN_BLOCKS_8X8_HEIGHT "${METALLIC_TERRAIN_BLOCKS_8X8_DIRECTORY}/terrain_blocks_8x8_height")
    set(METALLIC_TERRAIN_BLOCKS_8X8_QUERY "${METALLIC_TERRAIN_BLOCKS_8X8_HEIGHT}.heightfield.bin")
    add_metallic_terrain_block_height("${METALLIC_TERRAIN_BLOCKS_8X8_HEIGHT}" 8
        HEIGHT_FIELD "${METALLIC_TERRAIN_BLOCKS_8X8_QUERY}")
    add_metallic_terrain_vsm_sample(MetallicTerrainBlocks8x8VSMSample MetallicTerrainBlocks8x8VSMSmoke
        gpu-driven-terrain-blocks-8x8-vsm "${METALLIC_TERRAIN_BLOCKS_8X8_HEIGHT}"
        "${METALLIC_TERRAIN_BLOCKS_8X8_DIRECTORY}/terrain_blocks_8x8")

    set(METALLIC_TERRAIN_BLOCKS_16X16_DIRECTORY "${CMAKE_SOURCE_DIR}/Asset/MeshletCache/TerrainBlocks16x16")
    set(METALLIC_TERRAIN_BLOCKS_16X16_HEIGHT "${METALLIC_TERRAIN_BLOCKS_16X16_DIRECTORY}/terrain_blocks_16x16_height")
    set(METALLIC_TERRAIN_BLOCKS_16X16_QUERY "${METALLIC_TERRAIN_BLOCKS_16X16_HEIGHT}.heightfield.bin")
    add_metallic_terrain_block_height("${METALLIC_TERRAIN_BLOCKS_16X16_HEIGHT}" 16
        HEIGHT_FIELD "${METALLIC_TERRAIN_BLOCKS_16X16_QUERY}")
    add_metallic_terrain_vsm_sample(MetallicTerrainBlocks16x16VSMSample MetallicTerrainBlocks16x16VSMSmoke
        gpu-driven-terrain-blocks-16x16-vsm "${METALLIC_TERRAIN_BLOCKS_16X16_HEIGHT}"
        "${METALLIC_TERRAIN_BLOCKS_16X16_DIRECTORY}/terrain_blocks_16x16")

    add_metallic_terrain_vsm_sample(MetallicTerrainLayers16x16VSMSample MetallicTerrainLayers16x16VSMSmoke
        gpu-driven-terrain-layers-16x16-vsm "${METALLIC_TERRAIN_BLOCKS_16X16_HEIGHT}"
        "${CMAKE_SOURCE_DIR}/Asset/MeshletCache/TerrainLayers16x16/terrain_layers_16x16"
        WEIGHT_MAP "${CMAKE_SOURCE_DIR}/Asset/Terrain/TerrainRockWeight.exr")

    add_metallic_terrain_vsm_sample(MetallicTerrainPlacement16x16VSMSample MetallicTerrainPlacement16x16VSMSmoke
        gpu-driven-terrain-placement-16x16-vsm "${METALLIC_TERRAIN_BLOCKS_16X16_HEIGHT}"
        "${CMAKE_SOURCE_DIR}/Asset/MeshletCache/TerrainPlacement16x16/terrain_placement_16x16"
        WEIGHT_MAP "${CMAKE_SOURCE_DIR}/Asset/Terrain/TerrainRockWeight.exr"
        HEIGHT_FIELD "${METALLIC_TERRAIN_BLOCKS_16X16_QUERY}")

    set(terrain_regions_dir "${CMAKE_SOURCE_DIR}/Asset/MeshletCache/TerrainRegions")
    set(terrain_regions_scene "${terrain_regions_dir}/terrain_regions")
    set(terrain_regions_lighting "${CMAKE_SOURCE_DIR}/Asset/Terrain/TerrainVSM.metallic_scene.json")
    add_custom_command(
        OUTPUT "${terrain_regions_scene}.gltf" "${terrain_regions_scene}.metallic_scene.json"
            "${terrain_regions_dir}/terrain_region_west.gltf" "${terrain_regions_dir}/terrain_region_west.bin"
            "${terrain_regions_dir}/terrain_region_east.gltf" "${terrain_regions_dir}/terrain_region_east.bin"
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/BuildTerrainRegions.py"
            "${METALLIC_TERRAIN_BLOCKS_8X8_HEIGHT}.gltf" --output-directory "${terrain_regions_dir}"
            --lighting "${terrain_regions_lighting}" --height-field "${METALLIC_TERRAIN_BLOCKS_8X8_QUERY}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/BuildTerrainRegions.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/BuildTerrainVSMSample.py" "${CMAKE_CURRENT_SOURCE_DIR}/BuildTerrainMesh.py"
            "${METALLIC_TERRAIN_BLOCKS_8X8_HEIGHT}.gltf" "${METALLIC_TERRAIN_BLOCKS_8X8_HEIGHT}.bin"
            "${terrain_regions_lighting}" "${METALLIC_TERRAIN_BLOCKS_8X8_QUERY}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Splitting the 8x8 terrain into two adjacent regions"
        VERBATIM
    )
    set(terrain_region_cooks)
    foreach(region IN ITEMS west east)
        set(region_source "${terrain_regions_dir}/terrain_region_${region}")
        set(region_cook "${region_source}.gltf.meshstream.bin")
        add_custom_command(
            OUTPUT "${region_cook}"
            COMMAND "$<TARGET_FILE:MetallicMeshletCook>" --source "${region_source}.gltf"
                --output "${region_cook}" --compression none
            DEPENDS MetallicMeshletCook "${region_source}.gltf" "${region_source}.bin"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            COMMENT "Cooking the ${region} terrain region"
            VERBATIM
        )
        list(APPEND terrain_region_cooks "${region_cook}")
    endforeach()
    set(terrain_regions_cook "${terrain_regions_scene}.gltf.meshstream.bin")
    set(terrain_regions_receipt "${terrain_regions_scene}.meshstream.regions.json")
    add_custom_command(
        OUTPUT "${terrain_regions_cook}" "${terrain_regions_receipt}"
        COMMAND "$<TARGET_FILE:MetallicMeshletCook>" --source "${terrain_regions_scene}.gltf"
            --output "${terrain_regions_cook}" --compression none --validate-payloads
        DEPENDS MetallicMeshletCook "${terrain_regions_scene}.gltf"
            "${terrain_regions_scene}.metallic_scene.json"
            ${terrain_region_cooks}
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Packing the independent terrain region MeshletStreamAssets"
        VERBATIM
    )
    add_custom_target(MetallicTerrainRegionsSample DEPENDS "${terrain_regions_cook}" "${terrain_regions_receipt}")
    add_custom_target(MetallicPCGWorld DEPENDS LookDev MetallicTerrainRegionsSample)
    add_custom_target(MetallicTerrainRegionsSmoke
        COMMAND ${CMAKE_COMMAND} -E env METALLIC_SMOKE_TEST_HIDDEN=1
            METALLIC_SMOKE_TEST_TERRAIN_ROAM=1
            METALLIC_SMOKE_TEST_SAMPLE=gpu-driven-terrain-regions-vsm "$<TARGET_FILE:Metallic>" --smoke-test
        DEPENDS MetallicTerrainRegionsSample
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        COMMENT "Replaying the Terrain Regions viewport route over the west slope"
        VERBATIM
    )
else()
    message(STATUS "Python3 was not found; terrain sample build targets are unavailable")
endif()
