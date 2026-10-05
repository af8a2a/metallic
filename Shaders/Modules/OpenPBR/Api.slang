#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_api.h. Modified by Metallic, 2026.
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

// Public API for OpenPBR BSDF
// These wrapper functions provide the stable public interface, calling internal functions.
// More advanced workflows can call additional internal functions directly if needed - see README.md and BSDF.slang for details.

// Main initialization entrypoint: prepares volume, BSDF lobes, and emission from resolved inputs.

public OpenPBRPreparedBSDF openPBRPrepare<TContext : IOpenPBRContext>(TContext context,
                                                               const OpenPBRResolvedInputs resolved_inputs,
                                                               const float3 path_throughput,
                                                               const float3 rgb_wavelengths_nm,
                                                               const float exterior_ior, const float3 view_direction)
{
    return openPBRPrepareImpl(context, resolved_inputs, path_throughput, rgb_wavelengths_nm, exterior_ior,
                              view_direction);
}

// Evaluates the BSDF for a given light direction (includes cosine term).

public OpenPBRDiffuseSpecular openPBREval<TContext : IOpenPBRContext>(TContext context, const OpenPBRPreparedBSDF prepared,
                                                               const float3 light_direction)
{
    return openPBREvalImpl(context, prepared, light_direction);
}

// Importance-samples a light direction and returns weight, PDF, and sampled lobe type.

public void openPBRSample<TContext : IOpenPBRContext>(TContext context, const OpenPBRPreparedBSDF prepared, const float3 rand,
                                               out float3 light_direction, out OpenPBRDiffuseSpecular weight,
                                               out float pdf, out uint sampled_type)
{
    openPBRSampleImpl(context, prepared, rand, light_direction, weight, pdf, sampled_type);
}

// Returns the PDF for sampling the given light direction.

public float openPBRPdf<TContext : IOpenPBRContext>(TContext context, const OpenPBRPreparedBSDF prepared,
                                             const float3 light_direction)
{
    return openPBRPdfImpl(context, prepared, light_direction);
}
} // namespace Metallic.OpenPBR
