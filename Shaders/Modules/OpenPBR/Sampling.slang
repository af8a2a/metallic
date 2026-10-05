#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_sampling.h. Modified by Metallic, 2026.
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

// Sample a sphere uniformly and return a unit vector

public float3 openPBRSampleUnitSphereUniform(const float2 s)
{
    const float phi = OpenPBRTwoPi * s[0];
    const float z = 2.0f * s[1] - 1.0f;            // cos(theta)
    const float r = sqrt(1.0f - openPBRSquare(z)); // sin(theta)
    return float3(cos(phi) * r, sin(phi) * r, z);
}

// Sample the upper z hemisphere with a cosine distribution

public float3 openPBRSampleUnitHemisphereCosine(const float2 s)
{
    const float phi = OpenPBRTwoPi * s[0];
    const float z = sqrt(s[1]);        // cos(theta)
    const float r = sqrt(1.0f - s[1]); // sin(theta)
    return float3(cos(phi) * r, sin(phi) * r, z);
}

// Returns the PDF corresponding to the cosine-weighted hemisphere sampling function above.

public float openPBRComputePdfForSampleUnitHemisphereCosine(const float3 direction_local)
{
    return max(0.0f, openPBRGetCosThetaLocal(direction_local) * OpenPBRRcpPi);
}
} // namespace Metallic.OpenPBR
