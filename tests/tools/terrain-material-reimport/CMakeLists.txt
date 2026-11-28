cmake_minimum_required(VERSION 3.20)
project(TerrainMaterialReimportRegression LANGUAGES NONE)

add_executable(Metallic IMPORTED)
set_target_properties(Metallic PROPERTIES IMPORTED_LOCATION "${METALLIC_TEST_COOKER}")

include("${CMAKE_CURRENT_LIST_DIR}/../../../cmake/TerrainVSMSamples.cmake")
set(height "${CMAKE_CURRENT_BINARY_DIR}/TerrainBlocks16x16/terrain_blocks_16x16_height")
add_metallic_terrain_block_height("${height}" 16)
add_metallic_terrain_multilayer_vsm_sample("${height}" "${CMAKE_CURRENT_BINARY_DIR}/TerrainMultilayer16x16")
