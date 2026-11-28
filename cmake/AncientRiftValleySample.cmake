# Shared by the editor sample and the isolated material reimport regression.
function(add_metallic_ancient_rift_valley_sample manifest output_directory cook_target)
    cmake_parse_arguments(PARSE_ARGV 3 valley "FULL_WORLD" "" "")
    get_filename_component(repo "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
    get_filename_component(authoring_directory "${manifest}" DIRECTORY)
    set(tools "${repo}/Tools")
    set(lighting "${repo}/Asset/Terrain/TerrainVSM.metallic_scene.json")
    if(valley_FULL_WORLD)
        set(dimensions 4x4)
        set(first_region 0)
        set(last_region 3)
        set(region_args --regions-per-axis 4)
    else()
        set(dimensions 2x2)
        set(first_region 1)
        set(last_region 2)
        set(material_args --planar-layer-projection)
    endif()
    set(target "MetallicAncientRiftValley${dimensions}")
    set(sample "${output_directory}/ancient_rift_valley_${dimensions}")
    set(region_ids)
    foreach(z RANGE ${first_region} ${last_region})
        foreach(x RANGE ${first_region} ${last_region})
            list(APPEND region_ids "x${x}_z${z}")
        endforeach()
    endforeach()

    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${manifest}")
    file(READ "${manifest}" authoring)
    string(JSON height GET "${authoring}" height path)
    set(height "${authoring_directory}/${height}")
    set(material_inputs "${manifest}")
    string(JSON weight_count LENGTH "${authoring}" weightMaps)
    math(EXPR last_weight "${weight_count} - 1")
    foreach(index RANGE ${last_weight})
        string(JSON path GET "${authoring}" weightMaps ${index} path)
        list(APPEND material_inputs "${authoring_directory}/${path}")
    endforeach()
    string(JSON layer_count LENGTH "${authoring}" layers)
    math(EXPR last_layer "${layer_count} - 1")
    foreach(index RANGE ${last_layer})
        foreach(role IN ITEMS baseColor normal roughness)
            string(JSON path GET "${authoring}" layers ${index} textures ${role} path)
            list(APPEND material_inputs "${authoring_directory}/${path}")
        endforeach()
    endforeach()

    foreach(region IN LISTS region_ids)
        set(source "${output_directory}/terrain_region_${region}")
        list(APPEND region_geometry "${source}.gltf" "${source}.bin")
        set(stream "${source}.gltf.meshstream.bin")
        list(APPEND region_cooks "${stream}")
        add_custom_command(
            OUTPUT "${stream}"
            COMMAND "$<TARGET_FILE:${cook_target}>" --source "${source}.gltf" --output "${stream}" --compression none
            DEPENDS ${cook_target} "${source}.gltf" "${source}.bin"
            WORKING_DIRECTORY "${repo}"
            COMMENT "Cooking Ancient Rift Valley region ${region}"
            VERBATIM
        )
    endforeach()
    add_custom_command(
        OUTPUT "${sample}.gltf" "${sample}.metallic_scene.json" "${sample}.heightfield.bin" ${region_geometry}
        COMMAND "${Python3_EXECUTABLE}" -B "${tools}/BuildAncientRiftValleySample.py"
            "${height}" --lighting "${lighting}" --output "${sample}.gltf" ${region_args}
        DEPENDS "${height}" "${lighting}" "${tools}/BuildAncientRiftValleySample.py"
            "${tools}/BuildTerrainMesh.py" "${tools}/BuildTerrainRegions.py"
            "${tools}/BuildTerrainVSMMaterial.py" "${tools}/BuildTerrainVSMSample.py"
        WORKING_DIRECTORY "${repo}"
        COMMENT "Building world-space Ancient Rift Valley ${dimensions} terrain geometry"
        VERBATIM
    )
    add_custom_command(
        OUTPUT "${sample}.gltf.meshstream.bin" "${sample}.meshstream.regions.json"
        COMMAND "$<TARGET_FILE:${cook_target}>" --source "${sample}.gltf" --output "${sample}.gltf.meshstream.bin"
            --compression none --validate-payloads
        DEPENDS ${cook_target} "${sample}.gltf" ${region_cooks}
        WORKING_DIRECTORY "${repo}"
        COMMENT "Packing the Ancient Rift Valley ${dimensions} regions"
        VERBATIM
    )

    set(material_outputs "${sample}.material" "${sample}.materialdef" "${sample}.terrain-layers.json")
    add_custom_command(
        OUTPUT ${material_outputs}
        COMMAND "${Python3_EXECUTABLE}" -B "${tools}/BuildTerrainVSMMaterial.py"
            "${sample}.gltf" --layer-asset "${manifest}" ${material_args}
        DEPENDS "${sample}.gltf" ${material_inputs} "${tools}/BuildTerrainVSMMaterial.py"
            "${tools}/ImportTerrainLayers.py" "${tools}/TerrainLayerAsset.py"
            "${tools}/BuildTerrainLayerTextures.py" "${tools}/BuildTerrainMesh.py"
        WORKING_DIRECTORY "${repo}"
        COMMENT "Importing Ancient Rift Valley terrain materials"
        VERBATIM
    )
    add_custom_target("${target}Materials" DEPENDS ${material_outputs})
    add_custom_target("${target}Sample"
        DEPENDS "${sample}.gltf.meshstream.bin" "${sample}.meshstream.regions.json" "${sample}.metallic_scene.json")
    add_dependencies("${target}Sample" "${target}Materials")
endfunction()
