#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_generic_flipped_lobe.h. Modified by Metallic, 2026.
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

// Native thin-wall wrappers flip the child lobe hemisphere.
// The original implementation instantiated these with preprocessor macros.
//
// TODO: Note that when the reflected view direction is nearly parallel to the surface, it can
//       sometimes end up in the opposite hemisphere as the flipped normal due to rounding error
//       in the reflection. This is extremely rare, but if it causes problems in practice, we
//       could extract the view-direction-reflecting logic into a separate function and add
//       a fallback mechanism to handle this case, similar to the one in the
//       OpenPBR_ThinWallDiffuseTransmissionLobe's openpbr_initialize_lobe function.

// Get lobe type: query inner lobe type, then swap flags.

public uint openPBRGetLobeType<TContext : IOpenPBRContext>(TContext context, const OpenPBRThinWallDiffuseTransmissionLobe lobe)
{
    const uint inner_type = openPBRGetLobeType(context, lobe.flipped_lobe);
    return openPBRSwapReflectTransFlags(inner_type);
}

// Evaluate lobe value (BSDF*cosine): reflect view across original normal, then delegate to inner lobe.

public OpenPBRDiffuseSpecular
openPBRCalculateLobeValue<TContext : IOpenPBRContext>(TContext context,
                                                      const OpenPBRThinWallDiffuseTransmissionLobe lobe,
                                                      const float3 view_direction, const float3 light_direction)
{
    const float3 reflected_view = reflect(
        view_direction, lobe.flipped_lobe.normal_ff); // reflects view across surface (view points away from surface)
    return openPBRCalculateLobeValue(context, lobe.flipped_lobe, reflected_view, light_direction);
}

// Evaluate lobe-sampling PDF: same reflect-then-delegate.

public float openPBRCalculateLobePdf<TContext : IOpenPBRContext>(TContext context,
                                                          const OpenPBRThinWallDiffuseTransmissionLobe lobe,
                                                          const float3 view_direction, const float3 light_direction)
{
    const float3 reflected_view = reflect(
        view_direction, lobe.flipped_lobe.normal_ff); // reflects view across surface (view points away from surface)
    return openPBRCalculateLobePdf(context, lobe.flipped_lobe, reflected_view, light_direction);
}

// Sample lobe: reflect view, delegate, then swap reflection/transmission bits.

public bool openPBRSampleLobe<TContext : IOpenPBRContext>(TContext context, const OpenPBRThinWallDiffuseTransmissionLobe lobe,
                                                   const float3 rand, const float3 view_direction,
                                                   out float3 light_direction, out OpenPBRDiffuseSpecular weight,
                                                   out float pdf, out uint sampled_type)
{
    const float3 reflected_view = reflect(
        view_direction, lobe.flipped_lobe.normal_ff); // reflects view across surface (view points away from surface)
    if (!openPBRSampleLobe(context, lobe.flipped_lobe, rand, reflected_view, light_direction, weight, pdf,
                           sampled_type))
        return false;
    sampled_type = openPBRSwapReflectTransFlags(sampled_type);
    return true;
}

// Estimate lobe contribution: reflect view, delegate.

public float openPBREstimateLobeContribution<TContext : IOpenPBRContext>(TContext context,
                                                                  const OpenPBRThinWallDiffuseTransmissionLobe lobe,
                                                                  const float3 view_direction,
                                                                  const float3 path_throughput)
{
    const float3 reflected_view = reflect(
        view_direction, lobe.flipped_lobe.normal_ff); // reflects view across surface (view points away from surface)
    return openPBREstimateLobeContribution(context, lobe.flipped_lobe, reflected_view, path_throughput);
}
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

// Native thin-wall wrappers flip the child lobe hemisphere.
// The original implementation instantiated these with preprocessor macros.
//
// TODO: Note that when the reflected view direction is nearly parallel to the surface, it can
//       sometimes end up in the opposite hemisphere as the flipped normal due to rounding error
//       in the reflection. This is extremely rare, but if it causes problems in practice, we
//       could extract the view-direction-reflecting logic into a separate function and add
//       a fallback mechanism to handle this case, similar to the one in the
//       OpenPBR_ThinWallDiffuseTransmissionLobe's openpbr_initialize_lobe function.

// Get lobe type: query inner lobe type, then swap flags.

public uint openPBRGetLobeType<TContext : IOpenPBRContext>(TContext context,
                                                    const OpenPBRThinWallSpecularTransmissionLobe lobe)
{
    const uint inner_type = openPBRGetLobeType(context, lobe.flipped_lobe);
    return openPBRSwapReflectTransFlags(inner_type);
}

// Evaluate lobe value (BSDF*cosine): reflect view across original normal, then delegate to inner lobe.

public OpenPBRDiffuseSpecular
openPBRCalculateLobeValue<TContext : IOpenPBRContext>(TContext context,
                                                      const OpenPBRThinWallSpecularTransmissionLobe lobe,
                                                      const float3 view_direction, const float3 light_direction)
{
    const float3 reflected_view = reflect(
        view_direction, lobe.flipped_lobe.normal_ff); // reflects view across surface (view points away from surface)
    return openPBRCalculateLobeValue(context, lobe.flipped_lobe, reflected_view, light_direction);
}

// Evaluate lobe-sampling PDF: same reflect-then-delegate.

public float openPBRCalculateLobePdf<TContext : IOpenPBRContext>(TContext context,
                                                          const OpenPBRThinWallSpecularTransmissionLobe lobe,
                                                          const float3 view_direction, const float3 light_direction)
{
    const float3 reflected_view = reflect(
        view_direction, lobe.flipped_lobe.normal_ff); // reflects view across surface (view points away from surface)
    return openPBRCalculateLobePdf(context, lobe.flipped_lobe, reflected_view, light_direction);
}

// Sample lobe: reflect view, delegate, then swap reflection/transmission bits.

public bool openPBRSampleLobe<TContext : IOpenPBRContext>(TContext context, const OpenPBRThinWallSpecularTransmissionLobe lobe,
                                                   const float3 rand, const float3 view_direction,
                                                   out float3 light_direction, out OpenPBRDiffuseSpecular weight,
                                                   out float pdf, out uint sampled_type)
{
    const float3 reflected_view = reflect(
        view_direction, lobe.flipped_lobe.normal_ff); // reflects view across surface (view points away from surface)
    if (!openPBRSampleLobe(context, lobe.flipped_lobe, rand, reflected_view, light_direction, weight, pdf,
                           sampled_type))
        return false;
    sampled_type = openPBRSwapReflectTransFlags(sampled_type);
    return true;
}

// Estimate lobe contribution: reflect view, delegate.

public float openPBREstimateLobeContribution<TContext : IOpenPBRContext>(TContext context,
                                                                  const OpenPBRThinWallSpecularTransmissionLobe lobe,
                                                                  const float3 view_direction,
                                                                  const float3 path_throughput)
{
    const float3 reflected_view = reflect(
        view_direction, lobe.flipped_lobe.normal_ff); // reflects view across surface (view points away from surface)
    return openPBREstimateLobeContribution(context, lobe.flipped_lobe, reflected_view, path_throughput);
}
} // namespace Metallic.OpenPBR
