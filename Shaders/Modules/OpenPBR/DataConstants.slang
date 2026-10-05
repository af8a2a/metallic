#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_data_constants.h. Modified by Metallic, 2026.
namespace Metallic.OpenPBR
{
/***************************************************************************
 * Copyright 2026 Adobe
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ***************************************************************************/

// LUT dimensions and LUT ID constants.
//
// Metallic supports texture mode only. CPU texture payloads are defined in
// Runtime/Render/Material/OpenPBRLutData.h from Adobe's offline-integrated data.
// No table arrays or LUT integration code are compiled into this Slang module.

// =======================
// Lookup Table Dimensions
// =======================

// ================
// LUT ID Constants
// ================

// Integer IDs for all 8 OpenPBR precomputed lookup textures.
// Used as the 'id' argument to OPENPBR_SAMPLE_2D_TEXTURE / OPENPBR_SAMPLE_3D_TEXTURE
// (texture mode).
// Renderers storing OpenPBR textures consecutively can use these IDs directly as slot offsets.
//
// IDs 0-6: energy compensation tables (used by openpbr_look_up_* functions)
// ID  7  : LTC sheen table (sampled directly by the fuzz lobe)
// clang-format off

// clang-format on
} // namespace Metallic.OpenPBR
