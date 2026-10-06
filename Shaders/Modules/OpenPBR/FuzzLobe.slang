#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_fuzz_lobe.h. Modified by Metallic, 2026.
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

// This file implements the OpenPBR fuzz lobe, which is based on the paper:
// "Practical Multiple-Scattering Sheen Using Linearly Transformed Cosines".

///////////////////////////////
// DISNEY SHEEN LOBE SUMMARY //
///////////////////////////////

// This is an implementation of this paper:
//
//     "Practical Multiple-Scattering Sheen Using Linearly Transformed Cosines"
//     by Tizian Zeltner, Brent Burley, and Matt Jen-Yuan Chiang (2022).
//
// For more details about linearly transformed cosine distributions (LTCs), see this paper:
//
//     "Real-Time Polygonal-Light Shading with Linearly Transformed Cosines"
//     by Heitz et al. (2016).
//
// The LTCs used by this BRDF were fitted to a volumetric sheen layer of fiber-like
// particles with a SGGX microflake phase function. For details about SGGX, see this paper:
//
//     "The SGGX Microflake Distribution"
//     by Heitz et al. (2015).
//
// The reference volume was non-absorptive (with single-scattering albedo = 1)
// and had unit density and thickness. These choices were somewhat arbitrary but
// reproduced the overall sheen look from explicit fibers that the authors were
// going for. The ASM sheen lobe made similar choices.
//
// The reference implementation by Tizian Zeltner, Brent Burley, and Matt Jen-Yuan Chiang
// is available at https://github.com/tizian/ltc-sheen and is licensed under the
// Apache License, Version 2.0. The core Disney-sheen-specific functions below are
// derived from that reference: they are prefixed with "disney_sheen_" and use the
// structure and variable names from the reference implementation to make it easier
// to compare them. Direction vectors and calculations in those functions are in the
// z-up local space.

////////////////////////////
// DISNEY SHEEN LOBE DATA //
////////////////////////////

public struct OpenPBRFuzzLobe
{
    public OpenPBRCoatingLobe coating_lobe; // the lobe underneath the sheen lobe

    // Note that there is some confusing overloaded terminology in the literature related to the roughness.
    //
    // The "alpha" in GGX is sometimes called the "roughness" (in the Disney LTC sheen paper for example),
    // but that's different than the "roughness" in most artist-friendly material models.
    // For example, the Disney Principled BRDF paper defines the "roughness" as the
    // square root of "alpha" (see attached excerpt from the Disney Principled BRDF paper).
    // The SGGX paper calls "sigma" the "roughness". But the Disney LTC sheen paper's "roughness"
    // is denoted "alpha" and defined as the square root of "sigma".
    //
    // In other words, the common microfacet-model perceptual roughness is the square root
    // of the GGX roughness alpha, which matches the SGGX roughness sigma but is different
    // from the LTC-sheen roughness alpha, which is the square root of sigma.

    public float alpha; // artist-specified roughness parameter;
                 // the square root of the SGGX fiber cross section sigma

    public float3 tint; // multiplied by the sheen result to make even black sheen possible
    public float presence; // the proportion of the microsurface covered by the sheen

    public OpenPBRBasis basis; // a basis aligned with the normal and oriented according to the view direction
    public float3 view_dir_local; // the view direction in the z-up local space of the basis
    public float view_reflected; // overall proportion of light reflected by the sheen lobe for the given view direction
};

//////////////////////////////////////////////////////
// LOW-LEVEL MATH HELPER FUNCTIONS FOR DISNEY SHEEN //
//////////////////////////////////////////////////////

// Two helper functions for evaluating / sampling from LTCs that are defined
// based on a standard coordinate system aligned with "phi = 0".

public float openPBRDisneySheenPhi(const float3 v)
{
    float p = atan2(v.y, v.x);
    if (p < 0.0f)
        p += OpenPBRTwoPi;
    return p;
}

public float3 openPBRDisneySheenRotateVector(const float3 v, const float3 axis, const float angle)
{
    const float s = sin(angle);
    const float c = cos(angle);
    return v * c + axis * dot(v, axis) * (1.0f - c) + s * cross(axis, v);
}

//////////////////////////////////////////////////////
// LOOKUP-TABLE-RELATED UTILITIES FOR DISNEY SHEEN  //
//////////////////////////////////////////////////////

// Evaluate the LTC distribution in its default coordinate system.

public float openPBRDisneySheenEvalLtc(const float3 wi_local, const float3 ltc_coeffs)
{
    // The (inverse) transform matrix "M^{-1}" is given by:
    //
    //              [[a_inv 0    b_inv]
    //     M^{-1} =  [0    a_inv 0    ]
    //               [0    0    1     ]]
    //
    // with "a_inv = ltc_coeffs[0]", "b_inv = ltc_coeffs[1]" fetched from the
    // table. The transformed direction "wi_original_local" is therefore:
    //
    //                                            [[a_inv * wi_local.x + b_inv * wi_local.z]
    //     wi_original_local = M^{-1} * wi_local = [a_inv * wi_local.y                     ]
    //                                             [wi_local.z                             ]]
    //
    // which is subsequently normalized. The determinant of the matrix is:
    //
    //     |M^{-1}| = a_inv * a_inv
    //
    // which is used to compute the Jacobian determinant of the complete
    // mapping including the normalization.
    //
    // See the original paper [Heitz et al. 2016] for details about the LTC itself.

    const float a_inv = ltc_coeffs[0];
    const float b_inv = ltc_coeffs[1];
    float3 wi_original_local = float3(a_inv * wi_local.x + b_inv * wi_local.z, a_inv * wi_local.y, wi_local.z);
    const float len = length(wi_original_local);
    if (len == 0.0f)
        return 0.0f;
    wi_original_local /= len;

    const float det = a_inv * a_inv;
    const float jacobian = det / (len * len * len);

    return openPBRComputePdfForSampleUnitHemisphereCosine(wi_original_local) * jacobian;
}

// Sample from the LTC distribution in its default coordinate system.

public float3 openPBRDisneySheenSampleLtc(const float3 ltc_coeffs, const float2 rand)
{
    // The (inverse) transform matrix "M^{-1}" is given by:
    //
    //              [[a_inv 0    b_inv]
    //     M^{-1} =  [0    a_inv 0    ]
    //               [0    0    1     ]]
    //
    // with "a_inv = ltc_coeffs[0]", "b_inv = ltc_coeffs[1]" fetched from the
    // table. The non-inverted matrix "M" is therefore:
    //
    //         [[1/a_inv 0      -b_inv/a_inv]
    //     M =  [0      1/a_inv  0          ]
    //          [0      0       1           ]]
    //
    // and the transformed direction wi_local is:
    //
    //                                       [[wi_original_local.x/a_inv - wi_original_local.z*b_inv/a_inv]
    //     wi_local = M * wi_original_local = [wi_original_local.y/a_inv                                  ]
    //                                        [wi_original_local.z                                        ]]
    //
    // which is subsequently normalized.
    //
    // See the original paper [Heitz et al. 2016] for details about the LTC itself.

    const float3 wi_original_local = openPBRSampleUnitHemisphereCosine(rand);

    const float a_inv = ltc_coeffs[0];
    const float b_inv = ltc_coeffs[1];
    const float a = 1.0f / a_inv;
    const float3 wi_local =
        float3(wi_original_local.x * a - wi_original_local.z * b_inv * a, wi_original_local.y * a, wi_original_local.z);
    return normalize(wi_local);
}

// Fetch the LTC coefficients by bilinearly interpolating entries in a 32x32 lookup table.

public float3 openPBRDisneySheenFetchCoeffs<TContext : IOpenPBRContext>(TContext context, const OpenPBRFuzzLobe lobe,
                                                                 const float3 direction_local)
{
    // LTC coefficients are stored in a table parameterized by these variables:
    //
    //     - Elevation angle, parameterized as "cos(theta)".
    //     - Sheen roughness "alpha", which is related to the SGGX fiber cross
    //           section "sigma" as "alpha = sqrt(sigma)".
    //
    // Two versions of the LTC fit are available. We use the "Volume" version, which
    // is a direct fit of the volumetric sheen model (not the "Approx" version).

    // Since the original data is contained in a table (see https://github.com/tizian/ltc-sheen/blob/master/pbrt-v3/src/materials/sheenltc.cpp),
    // we compensate the difference between bilinear interpolation in that table and in a regular texture by:
    //     - keeping the original sampling indices.
    //     - offsetting the UVs by 0.5 (in pixel space).
    const float2 uv_max = float2(float(32 - 1) / float(32), float(32 - 1) / float(32));
    const float2 uv_step = float2(0.5f / float(32), 0.5f / float(32));
    const float2 uv = float2(openPBRGetCosThetaLocal(direction_local), lobe.alpha) * uv_max + uv_step;
    return (context.sample2D(7, uv)).xyz;
}

/////////////////////////////////////
// CORE BSDF MATH FOR DISNEY SHEEN //
/////////////////////////////////////

// Evaluate the BRDF.

public float3 openPBRDisneySheenF<TContext : IOpenPBRContext>(TContext context, const OpenPBRFuzzLobe lobe,
                                                       const float3 wo_local, const float3 wi_local)
{
    const float cos_theta_o = openPBRGetCosThetaLocal(wo_local);
    const float cos_theta_i = openPBRGetCosThetaLocal(wi_local);
    if (cos_theta_o <= 0.0f || cos_theta_i <= 0.0f)
        return float3(0.0f);

    // Rotate coordinate frame to align with incident direction wo_local.
    const float phi_std = openPBRDisneySheenPhi(wo_local);
    const float3 wi_std_local = openPBRDisneySheenRotateVector(wi_local, float3(0.0f, 0.0f, 1.0f), -phi_std);

    // Evaluate LTC distribution in aligned coordinates.
    const float3 ltc_coeffs = openPBRDisneySheenFetchCoeffs(context, lobe, wo_local);
    float3 value = float3(openPBRDisneySheenEvalLtc(wi_std_local, ltc_coeffs));

    // Also consider the presence (surface coverage), overall reflectance ("R" in the paper),
    // and artist-specified sheen tint ("Csheen" in the paper).
    //
    // Ideally color would be introduced by adding absorption to the microflake volume itself
    // during the simulation, however that would complicate the LTC fitting --
    // the simpler solution is just to multiply the final result by the given color.
    const float r = ltc_coeffs[2];
    value *= lobe.presence * r * lobe.tint;

    // The cosine foreshortening factor included in the LTC and Eclair's convention
    // is to include this factor in value returned from the BSDF.
    // If the cosine foreshortening factor were handled in the integrator instead of the BSDF,
    // we would need to cancel it out here like this:
    //
    // return cos_theta_i > 0.0f ? value / cos_theta_i : vec3(0.0f);

    return value;
}

// Sample proportionally to the BRDF.
// The LTC representation allows perfect importance sampling.
// Returns whether a valid sample was successfully generated.

public bool openPBRDisneySheenSampleF<TContext : IOpenPBRContext>(TContext context, const OpenPBRFuzzLobe lobe,
                                                           const float2 rand, const float3 wo_local,
                                                           out float3 wi_local)
{
    const float cos_theta_o = openPBRGetCosThetaLocal(wo_local);
    if (cos_theta_o <= 0.0f) {
        wi_local = float3(0.0f);
        return false;
    }

    // Sample from the LTC distribution in aligned coordinates.
    const float3 wi_std_local =
        openPBRDisneySheenSampleLtc(openPBRDisneySheenFetchCoeffs(context, lobe, wo_local), rand);

    // Rotate coordinate frame based on incident direction wo_local.
    const float phi_std = openPBRDisneySheenPhi(wo_local);
    wi_local = openPBRDisneySheenRotateVector(wi_std_local, float3(0.0f, 0.0f, 1.0f), +phi_std);

    if (!openPBRAreInSameHemisphereLocal(wo_local, wi_local))
        return false;

    return true;
}

// Sampling density associated with the "disney_sheen_sample_f" function above. Used for MIS.

public float openPBRDisneySheenPdf<TContext : IOpenPBRContext>(TContext context, const OpenPBRFuzzLobe lobe,
                                                        const float3 wo_local, const float3 wi_local)
{
    const float cos_theta_o = openPBRGetCosThetaLocal(wo_local);
    const float cos_theta_i = openPBRGetCosThetaLocal(wi_local);
    if (cos_theta_o <= 0.0f || cos_theta_i <= 0.0f)
        return 0.0f;

    // Rotate coordinate frame to align with incident direction wo_local.
    const float phi_std = openPBRDisneySheenPhi(wo_local);
    const float3 wi_std_local = openPBRDisneySheenRotateVector(wi_local, float3(0.0f, 0.0f, 1.0f), -phi_std);

    // Evaluate LTC distribution in aligned coordinates.
    return openPBRDisneySheenEvalLtc(wi_std_local, openPBRDisneySheenFetchCoeffs(context, lobe, wo_local));
}

///////////////////////////////////////////////////////////
// GENERAL SHEEN LOBE FRAMEWORK ADAPTED FOR DISNEY SHEEN //
///////////////////////////////////////////////////////////

public float openPBRProportionReflected<TContext : IOpenPBRContext>(TContext context, const OpenPBRFuzzLobe lobe,
                                                             const float3 direction_local)
{
    if (lobe.presence == 0.0f) { return 0.0f; }

    // The implementation of this function uses the conventions of the core Disney sheen functions.
    //
    // Note that the proportion reflected is always less than one, which also means that
    // the base layer is always visible to some extent. This makes sense because the sheen
    // is based on a volume of finite density and thickness.

    const float cos_theta = openPBRGetCosThetaLocal(direction_local);
    if (cos_theta <= 0.0f)
        return 0.0f;

    // The third component of each of the items in the LTC LUT is the overall reflectance ("R" in the paper).
    const float3 ltc_coeffs = openPBRDisneySheenFetchCoeffs(context, lobe, direction_local);
    const float r = ltc_coeffs[2];
    return lobe.presence * r;
}

public float openPBRBaseLayerScaleIncoming(const OpenPBRFuzzLobe lobe) { return 1.0f - lobe.view_reflected; }

public float openPBRBaseLayerScaleOutgoing<TContext : IOpenPBRContext>(TContext context, const OpenPBRFuzzLobe lobe,
                                                                const float3 light_direction_local)
{
    const float light_reflected = openPBRProportionReflected(context, lobe, light_direction_local);
    return 1.0f - light_reflected;
}

public float openPBRBaseLayerScaleComplete<TContext : IOpenPBRContext>(TContext context, const OpenPBRFuzzLobe lobe,
                                                                const float3 light_direction_local)
{
    const float incoming = openPBRBaseLayerScaleIncoming(lobe);
    const float outgoing = openPBRBaseLayerScaleOutgoing(context, lobe, light_direction_local);
    return incoming * outgoing;
}

public float openPBRSheenProbability<TContext : IOpenPBRContext>(TContext context, const OpenPBRFuzzLobe lobe,
                                                          const float3 view_direction)
{
    // Here we calculate the expected contribution of the sheen lobe relative to the overall lobe
    // and we use that to set the probability of sampling the sheen lobe.

    // TODO: Take the real path throughput into account (for both standalone lobes),
    //       either by saving it in the lobe struct or by passing it in.
    const float3 PlaceholderPathThroughput = float3(1.0f);

    // lobe.view_reflected is already weighted by lobe.presence; see openpbr_proportion_reflected().
    const float sheen_contribution = lobe.view_reflected * openPBRMax3(lobe.tint);

    const float unscaled_base_lobe_contribution =
        openPBREstimateLobeContribution(context, lobe.coating_lobe, view_direction, PlaceholderPathThroughput);
    const float scaled_base_lobe_contribution = unscaled_base_lobe_contribution * openPBRBaseLayerScaleIncoming(lobe);

    return openPBRSafeDivide(sheen_contribution, sheen_contribution + scaled_base_lobe_contribution, 0.5f);
}

// Assumes that child lobe has already been initialized.

public void openPBRInitializeLobe<TContext : IOpenPBRContext>(TContext context, inout OpenPBRFuzzLobe lobe,
                                                       const float3 normal_ff, const float3 view_direction,
                                                       const float roughness, const float3 tint, const float presence)
{
    lobe.alpha = roughness;

    lobe.tint = tint;
    lobe.presence = presence;

    lobe.basis = openPBRMakeBasis(normal_ff);
    lobe.view_dir_local = openPBRWorldToLocal(lobe.basis, view_direction);
    lobe.view_reflected = openPBRProportionReflected(context, lobe, lobe.view_dir_local);
}

public uint openPBRGetLobeType<TContext : IOpenPBRContext>(TContext context, const OpenPBRFuzzLobe lobe)
{
    return OpenPBRBSDFLobeTypeGlossy | OpenPBRBSDFLobeTypeReflection | openPBRGetLobeType(context, lobe.coating_lobe);
}

public OpenPBRDiffuseSpecular openPBRCalculateLobeValue<TContext : IOpenPBRContext>(TContext context,
                                                                             const OpenPBRFuzzLobe lobe,
                                                                             const float3 view_direction,
                                                                             const float3 light_direction)
{
    const OpenPBRDiffuseSpecular base_value =
        openPBRCalculateLobeValue(context, lobe.coating_lobe, view_direction, light_direction);

    if (lobe.presence == 0.0f) { return base_value; }

    const float3 light_dir_local = openPBRWorldToLocal(lobe.basis, light_direction);
    const float3 sheen_value = openPBRDisneySheenF(context, lobe, lobe.view_dir_local, light_dir_local);

    return openPBRAddDiffuseSpecular(
        openPBRScaleDiffuseSpecular(base_value, openPBRBaseLayerScaleComplete(context, lobe, light_dir_local)),
        openPBRMakeDiffuseSpecularFromSpecular(sheen_value));
}

public float openPBRCalculateLobePdf<TContext : IOpenPBRContext>(TContext context, const OpenPBRFuzzLobe lobe,
                                                          const float3 view_direction, const float3 light_direction)
{
    const float base_pdf = openPBRCalculateLobePdf(context, lobe.coating_lobe, view_direction, light_direction);

    const float sheen_prob = openPBRSheenProbability(context, lobe, view_direction);
    if (sheen_prob == 0.0f) { return base_pdf; }

    const float3 light_dir_local = openPBRWorldToLocal(lobe.basis, light_direction);
    const float sheen_pdf = openPBRDisneySheenPdf(context, lobe, lobe.view_dir_local, light_dir_local);


    return lerp(base_pdf, sheen_pdf, sheen_prob);
}

// Given a view direction, sample a light direction,
// evaluate the throughput and pdf for this direction,
// and return whether a sample was successfully generated.
// Output arguments are only set when the function returns true.

public bool openPBRSampleLobe<TContext : IOpenPBRContext>(TContext context, const OpenPBRFuzzLobe lobe, const float3 rand,
                                                   const float3 view_direction, out float3 light_direction,
                                                   out OpenPBRDiffuseSpecular weight, out float pdf,
                                                   out uint sampled_type)
{
    const float sheen_prob = openPBRSheenProbability(context, lobe, view_direction);
    // Preserve Adobe's 0/0 -> 0.5 mixture for a completely black closure.
    // Presence alone is insufficient to bypass that distribution.
    if (lobe.presence == 0.0f && sheen_prob == 0.0f) {
        bool valid = openPBRSampleLobe(context, lobe.coating_lobe, rand, view_direction,
            light_direction, weight, pdf, sampled_type);
        if (valid && !bool(sampled_type & OpenPBRBSDFLobeTypeSpecular)) {
            // Preserve the reference weight -> projected value -> weight rounding.
            weight = openPBRScaleDiffuseSpecular(openPBRScaleDiffuseSpecular(weight, pdf), 1.0f / pdf);
        }
        return valid;
    }
    const float base_prob = 1.0f - sheen_prob;

    float rand_x = rand.x;
    if (rand_x < sheen_prob) {
        // Sample the sheen lobe itself.

        // If we were to need 3 random numbers instead of 2, we could remap the first one like this:
        //     rand_x /= sheen_prob;
        //     clamp_remapped_random_number(rand_x);

        float3 light_dir_local;
        const bool success =
            openPBRDisneySheenSampleF(context, lobe, float2(rand.y, rand.z), lobe.view_dir_local, light_dir_local);
        if (!success) {
            openPBRClearLobeSamplingOutput(light_direction, weight, pdf, sampled_type);
            return false;
        }

        light_direction = openPBRLocalToWorld(lobe.basis, light_dir_local);

        pdf = openPBRDisneySheenPdf(context, lobe, lobe.view_dir_local, light_dir_local) * sheen_prob;
        pdf += openPBRCalculateLobePdf(context, lobe.coating_lobe, view_direction, light_direction) *
               base_prob; // The returned pdf takes both lobes into account.

        weight = openPBRScaleDiffuseSpecular(openPBRCalculateLobeValue(context, lobe, view_direction, light_direction),
                                             1.0f / pdf); // The returned weight takes both lobes into account.

        sampled_type = OpenPBRBSDFLobeTypeGlossy | OpenPBRBSDFLobeTypeReflection;
    } else {
        // Sample the base lobe below the sheen lobe.

        const float reciprocal_base_prob = 1.0f / base_prob;

        rand_x = (rand_x - sheen_prob) * reciprocal_base_prob;
        openPBRClampRemappedRandomNumber(rand_x);

        const bool success = openPBRSampleLobe(context, lobe.coating_lobe, float3(rand_x, rand.y, rand.z),
                                               view_direction, light_direction, weight, pdf, sampled_type);
        if (!success)
            return false;

        const float3 light_dir_local = openPBRWorldToLocal(lobe.basis, light_direction);

        if (!bool(sampled_type & OpenPBRBSDFLobeTypeSpecular)) // non-delta
        {
            OpenPBRDiffuseSpecular bsdf_cos = openPBRScaleDiffuseSpecular(weight, pdf);
            bsdf_cos =
                openPBRScaleDiffuseSpecular(bsdf_cos, openPBRBaseLayerScaleComplete(context, lobe, light_dir_local));
            const float3 sheen = openPBRDisneySheenF(context, lobe, lobe.view_dir_local, light_dir_local);
            bsdf_cos = openPBRAddDiffuseSpecular(bsdf_cos, openPBRMakeDiffuseSpecularFromSpecular(sheen));

            pdf *= base_prob;
            pdf += openPBRDisneySheenPdf(context, lobe, lobe.view_dir_local, light_dir_local) *
                   sheen_prob; // The returned pdf takes both lobes into account.

            weight =
                openPBRScaleDiffuseSpecular(bsdf_cos, 1.0f / pdf); // The returned weight takes both lobes into account.
        } else                                                     // delta
        {
            weight = openPBRScaleDiffuseSpecular(weight, openPBRBaseLayerScaleComplete(context, lobe, light_dir_local) *
                                                             reciprocal_base_prob);
        }
    }

    return true;
}

public float openPBREstimateLobeContribution<TContext : IOpenPBRContext>(TContext context, const OpenPBRFuzzLobe lobe,
                                                                  const float3 view_direction,
                                                                  const float3 path_throughput)
{
    // Note that presence isn't explicitly included below because lobe.view_reflected is already
    // weighted by lobe.presence; see openpbr_proportion_reflected().
    return openPBRMaxComponentOfThroughputWeightedColor(path_throughput, lobe.view_reflected * lobe.tint) +
           openPBREstimateLobeContribution(context, lobe.coating_lobe, view_direction, path_throughput) *
               openPBRBaseLayerScaleIncoming(lobe);
}
} // namespace Metallic.OpenPBR
