/*
 * tinygltf single implementation unit for MetallicRuntimeScene.
 *
 * SPDX-License-Identifier: MIT
 */

#define TINYGLTF_IMPLEMENTATION
// External images are read by SceneLoader's decode tasks, not during glTF import.
// Keep embedded images and external geometry buffers available to the importer.
#define TINYGLTF_NO_EXTERNAL_IMAGE
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION

#include "json.hpp"
#include "stb_image.h"
#include "tiny_gltf.h"
