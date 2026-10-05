#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_bsdf_lobe_type.h. Modified by Metallic, 2026.
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

public static const uint OpenPBRBSDFLobeTypeNone = 0;
public static const uint OpenPBRBSDFLobeTypeReflection = uint(1) << 0;
public static const uint OpenPBRBSDFLobeTypeTransmission = uint(1) << 1;
public static const uint OpenPBRBSDFLobeTypeDiffuse = uint(1) << 2;
public static const uint OpenPBRBSDFLobeTypeGlossy = uint(1) << 3;
public static const uint OpenPBRBSDFLobeTypeSpecular = uint(1) << 4;
public static const uint OpenPBRBSDFLobeTypeVolume = uint(1) << 5;

// Swap reflection and transmission in the OpenPBR_BsdfLobeType bitmask.

public uint openPBRSwapReflectTransFlags(uint bsdf_lobe_type)
{
    const bool has_r = bool(bsdf_lobe_type & OpenPBRBSDFLobeTypeReflection);
    const bool has_t = bool(bsdf_lobe_type & OpenPBRBSDFLobeTypeTransmission);
    bsdf_lobe_type &= ~(OpenPBRBSDFLobeTypeReflection | OpenPBRBSDFLobeTypeTransmission);
    if (has_r)
        bsdf_lobe_type |= OpenPBRBSDFLobeTypeTransmission;
    if (has_t)
        bsdf_lobe_type |= OpenPBRBSDFLobeTypeReflection;
    return bsdf_lobe_type;
}
} // namespace Metallic.OpenPBR
