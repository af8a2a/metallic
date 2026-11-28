# Shared terrain cooking rules used by the built-in samples and the isolated Ninja regression.
get_filename_component(METALLIC_TERRAIN_REPOSITORY_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(METALLIC_TERRAIN_TOOLS_DIR "${METALLIC_TERRAIN_REPOSITORY_ROOT}/Tools")
set(METALLIC_TERRAIN_P0_SOURCE "${METALLIC_TERRAIN_REPOSITORY_ROOT}/Asset/simple_terrain_height.exr")
set(METALLIC_TERRAIN_HEIGHT_ARGS --horizontal-size 256 --height-scale 16 --skirt-depth 8)

function(add_metallic_terrain_block_height output tiles_per_axis)
    cmake_parse_arguments(PARSE_ARGV 2 terrain "" "HEIGHT_FIELD" "")
    set(terrain_args)
    if(terrain_HEIGHT_FIELD)
        set(terrain_args --height-field-output "${terrain_HEIGHT_FIELD}")
    endif()
    add_custom_command(
        OUTPUT "${output}.gltf" "${output}.bin" ${terrain_HEIGHT_FIELD}
        COMMAND "${Python3_EXECUTABLE}" -B "${METALLIC_TERRAIN_TOOLS_DIR}/BuildTerrainMesh.py"
            "${METALLIC_TERRAIN_P0_SOURCE}" --output "${output}.gltf"
            ${METALLIC_TERRAIN_HEIGHT_ARGS} --tiles-per-axis ${tiles_per_axis} ${terrain_args}
        DEPENDS "${METALLIC_TERRAIN_TOOLS_DIR}/BuildTerrainMesh.py" "${METALLIC_TERRAIN_P0_SOURCE}"
        WORKING_DIRECTORY "${METALLIC_TERRAIN_REPOSITORY_ROOT}"
        COMMENT "Converting the Houdini EXR height field to ${tiles_per_axis}x${tiles_per_axis} terrain blocks"
        VERBATIM
    )
endfunction()

function(add_metallic_terrain_vsm_sample bake_target smoke_target sample_id terrain_source output)
    cmake_parse_arguments(PARSE_ARGV 5 terrain "" "WEIGHT_MAP;LAYER_ASSET;HEIGHT_FIELD;MATERIAL_TARGET" "LAYER_TEXTURES")
    set(lighting "${METALLIC_TERRAIN_REPOSITORY_ROOT}/Asset/Terrain/TerrainVSM.metallic_scene.json")
    set(graph "${METALLIC_TERRAIN_REPOSITORY_ROOT}/Pipelines/Samples/gpu_driven_terrain_vsm.metallic_graph.json")
    set(stream_asset "${output}.gltf.meshstream.bin")
    set(scene_args)
    set(scene_dependencies)
    set(layer_args)
    set(layer_outputs)
    set(layer_dependencies)
    if(terrain_WEIGHT_MAP)
        list(APPEND scene_args --terrain-material legacy)
        set(layer_args --weight-map "${terrain_WEIGHT_MAP}")
        set(layer_outputs "${output}_weight.png"
            "${output}_grass_color.png" "${output}_grass_normal.png"
            "${output}_rock_color.png" "${output}_rock_normal.png")
        set(layer_dependencies "${terrain_WEIGHT_MAP}")
    elseif(terrain_LAYER_ASSET)
        list(APPEND scene_args --terrain-material multilayer)
        set(layer_args --layer-asset "${terrain_LAYER_ASSET}")
        set(layer_dependencies "${terrain_LAYER_ASSET}" ${terrain_LAYER_TEXTURES}
            "${METALLIC_TERRAIN_TOOLS_DIR}/ImportTerrainLayers.py"
            "${METALLIC_TERRAIN_TOOLS_DIR}/TerrainLayerAsset.py")
    endif()
    if(terrain_HEIGHT_FIELD)
        list(APPEND scene_args --height-field "${terrain_HEIGHT_FIELD}")
        list(APPEND scene_dependencies "${terrain_HEIGHT_FIELD}")
    endif()
    add_custom_command(
        OUTPUT "${output}.gltf" "${output}.bin" "${output}.metallic_scene.json"
        COMMAND "${Python3_EXECUTABLE}" -B "${METALLIC_TERRAIN_TOOLS_DIR}/BuildTerrainVSMSample.py"
            "${terrain_source}.gltf" "${lighting}" "${graph}" "${output}.gltf" ${scene_args}
        DEPENDS "${METALLIC_TERRAIN_TOOLS_DIR}/BuildTerrainVSMSample.py"
            "${METALLIC_TERRAIN_TOOLS_DIR}/BuildTerrainVSMMaterial.py"
            "${METALLIC_TERRAIN_TOOLS_DIR}/BuildTerrainMesh.py"
            "${terrain_source}.gltf" "${lighting}" "${graph}" ${scene_dependencies}
        WORKING_DIRECTORY "${METALLIC_TERRAIN_REPOSITORY_ROOT}"
        COMMENT "Assembling ${sample_id} with opaque mesh props"
        VERBATIM
    )
    if(layer_args)
        list(APPEND layer_outputs "${output}.material" "${output}.materialdef" "${output}.terrain-layers.json")
        add_custom_command(
            OUTPUT ${layer_outputs}
            COMMAND "${Python3_EXECUTABLE}" -B "${METALLIC_TERRAIN_TOOLS_DIR}/BuildTerrainVSMMaterial.py"
                "${output}.gltf" ${layer_args}
            DEPENDS "${output}.gltf" ${layer_dependencies}
                "${METALLIC_TERRAIN_TOOLS_DIR}/BuildTerrainVSMMaterial.py"
                "${METALLIC_TERRAIN_TOOLS_DIR}/BuildTerrainLayerTextures.py"
                "${METALLIC_TERRAIN_TOOLS_DIR}/BuildTerrainMesh.py"
            WORKING_DIRECTORY "${METALLIC_TERRAIN_REPOSITORY_ROOT}"
            COMMENT "Importing ${sample_id} terrain materials"
            VERBATIM
        )
        if(terrain_MATERIAL_TARGET)
            add_custom_target(${terrain_MATERIAL_TARGET} DEPENDS ${layer_outputs})
        endif()
    endif()
    add_custom_command(
        OUTPUT "${stream_asset}"
        COMMAND "$<TARGET_FILE:Metallic>" --build-meshstream "${output}.gltf" --output "${stream_asset}"
        DEPENDS Metallic "${output}.gltf" "${output}.bin" "${terrain_source}.bin"
        WORKING_DIRECTORY "${METALLIC_TERRAIN_REPOSITORY_ROOT}"
        COMMENT "Building the ${sample_id} MeshletStreamAsset"
        VERBATIM
    )
    add_custom_target(${bake_target} DEPENDS "${stream_asset}" "${output}.metallic_scene.json" ${layer_outputs})
    add_custom_target(${smoke_target}
        COMMAND ${CMAKE_COMMAND} -E env METALLIC_SMOKE_TEST_HIDDEN=1
            "METALLIC_SMOKE_TEST_SAMPLE=${sample_id}" "$<TARGET_FILE:Metallic>" --smoke-test
        DEPENDS ${bake_target}
        WORKING_DIRECTORY "${METALLIC_TERRAIN_REPOSITORY_ROOT}"
        COMMENT "Rendering the ${sample_id} smoke frame"
        VERBATIM
    )
endfunction()

function(add_metallic_terrain_layer_example asset out_textures)
    cmake_parse_arguments(PARSE_ARGV 2 example "" "REGION_GRID" "")
    get_filename_component(directory "${asset}" DIRECTORY)
    set(textures "${directory}/weights.exr")
    foreach(layer IN ITEMS rock grass gravel snow)
        foreach(role IN ITEMS baseColor normal roughness)
            list(APPEND textures "${directory}/${layer}_${role}.png")
        endforeach()
    endforeach()
    if(example_REGION_GRID)
        set(region_args --region-grid "${example_REGION_GRID}")
    endif()
    add_custom_command(
        OUTPUT "${asset}" ${textures}
        COMMAND "${Python3_EXECUTABLE}" -B "${METALLIC_TERRAIN_TOOLS_DIR}/ExportTerrainMultilayerExample.py"
            "${METALLIC_TERRAIN_REPOSITORY_ROOT}/Asset/Terrain/TerrainRockWeight.exr" --output "${asset}" ${region_args}
        DEPENDS "${METALLIC_TERRAIN_REPOSITORY_ROOT}/Asset/Terrain/TerrainRockWeight.exr"
            "${METALLIC_TERRAIN_TOOLS_DIR}/ExportTerrainMultilayerExample.py"
            "${METALLIC_TERRAIN_TOOLS_DIR}/BuildTerrainLayerTextures.py" "${METALLIC_TERRAIN_TOOLS_DIR}/BuildTerrainMesh.py"
            "${METALLIC_TERRAIN_TOOLS_DIR}/ImportTerrainLayers.py" "${METALLIC_TERRAIN_TOOLS_DIR}/TerrainLayerAsset.py"
            ${example_REGION_GRID}
        COMMENT "Generating the four terrain sample layers and RGBA weights"
        VERBATIM
    )
    set(${out_textures} ${textures} PARENT_SCOPE)
endfunction()

function(add_metallic_terrain_multilayer_vsm_sample terrain_source terrain_multilayer_dir)
    set(terrain_multilayer_asset "${terrain_multilayer_dir}/source/terrain.layers.json")
    add_metallic_terrain_layer_example("${terrain_multilayer_asset}" terrain_multilayer_textures)
    add_metallic_terrain_vsm_sample(MetallicTerrainMultilayer16x16VSMSample MetallicTerrainMultilayer16x16VSMSmoke
        gpu-driven-terrain-multilayer-16x16-vsm "${terrain_source}"
        "${terrain_multilayer_dir}/terrain_multilayer_16x16"
        MATERIAL_TARGET MetallicTerrainMultilayer16x16VSMMaterials
        LAYER_ASSET "${terrain_multilayer_asset}" LAYER_TEXTURES ${terrain_multilayer_textures})
endfunction()
