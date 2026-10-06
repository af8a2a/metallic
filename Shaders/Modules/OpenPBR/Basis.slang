#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_basis.h. Modified by Metallic, 2026.
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

// This class stores an orthonormal basis with a z-up local space.
// It can be right-handed (handedness +1) or left-handed (handedness -1).

public struct OpenPBRBasis
{
    public float3 t, b, n;
};

public OpenPBRBasis openPBRMakeIdentityBasis()
{
    OpenPBRBasis basis;

    basis.t = float3(1.0f, 0.0f, 0.0f);
    basis.b = float3(0.0f, 1.0f, 0.0f);
    basis.n = float3(0.0f, 0.0f, 1.0f);

    return basis;
}

// The input normal needs to be unit length.

public OpenPBRBasis openPBRMakeBasis(const float3 normal)
{

    OpenPBRBasis basis;
    basis.n = normal;

    // The code below uses the algorithm from "Building an Orthonormal Basis, Revisited"
    // (http://jcgt.org/published/0006/01/01/).

    const float nz_sign = basis.n.z < 0.0f ? -1.0f : 1.0f;
    const float a = -1.0f / (nz_sign + basis.n.z);
    const float b = basis.n.x * basis.n.y * a;

    basis.t = float3(b, nz_sign + basis.n.y * basis.n.y * a, -basis.n.y);
    basis.b = float3(1.0f + nz_sign * basis.n.x * basis.n.x * a, nz_sign * b, -nz_sign * basis.n.x);

    return basis;
}

// The input normal needs to be unit length.

public OpenPBRBasis openPBRMakeBasis(const float3 normal, const float3 tangent, const float handedness)
{

    OpenPBRBasis basis;

    basis.n = normal;
    basis.t = openPBRFastNormalize(tangent - dot(tangent, basis.n) * basis.n);
    basis.b = cross(basis.n, basis.t) * handedness;

    return basis;
}

// The input normal needs to be unit length.

public OpenPBRBasis openPBRMakeBasis(const float3 normal, const float3 tangent, const float3 bitangent)
{
    // Orthonormalize the input vectors using the modified Gram-Schmidt process
    // (which is more numerically stable than the classical Gram-Schmidt process).
    // For details, see https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process#Numerical_stability.

    const float3 new_tangent =
        openPBRFastNormalize(tangent - dot(tangent, normal) * normal); // make tangent perpendicular to normal
    const float3 intermediate_bitangent =
        bitangent - dot(bitangent, normal) * normal; // make bitangent perpendicular to normal
    const float3 new_bitangent =
        openPBRFastNormalize(intermediate_bitangent - dot(intermediate_bitangent, new_tangent) *
                                                          new_tangent); // make bitangent perpendicular to tangent also

    OpenPBRBasis basis;

    basis.t = new_tangent;
    basis.b = new_bitangent;
    basis.n = normal; // would need to be explicitly normalized if function accepted unnormalized normals

    return basis;
}

public void openPBRInvertBasis(inout OpenPBRBasis basis)
{
    basis.t *= -1.0f;
    basis.b *= -1.0f;
    basis.n *= -1.0f;
}

public float openPBRGetBasisHandedness(const float3 normal, const float3 tangent, const float3 bitangent)
{
    return dot(cross(tangent, bitangent), normal) >= 0.0f ? 1.0f : -1.0f;
}

public float openPBRGetBasisHandedness(const OpenPBRBasis basis)
{
    return openPBRGetBasisHandedness(basis.n, basis.t, basis.b);
}

// Rotates counterclockwise around the normal.

public void openPBRRotateBasisAroundNormal(inout OpenPBRBasis basis, float2 cos_sin, float handedness)
{
    const float3 rotated_tangent = basis.t * cos_sin.x + handedness * basis.b * cos_sin.y;
    const float3 rotated_bitangent = basis.b * cos_sin.x - handedness * basis.t * cos_sin.y;
    basis.t = rotated_tangent;
    basis.b = rotated_bitangent;
}

public void openPBRRotateBasisAroundNormal(inout OpenPBRBasis basis, float2 cos_sin)
{
    openPBRRotateBasisAroundNormal(basis, cos_sin, openPBRGetBasisHandedness(basis));
}

public void openPBRRotateBasisAroundNormal(inout OpenPBRBasis basis, float radians)
{
    const float sine = sin(radians);
    const float cosine = cos(radians);
    openPBRRotateBasisAroundNormal(basis, float2(cosine, sine));
}

public float3 openPBRWorldToLocal(const OpenPBRBasis basis, const float3 direction)
{
    return float3(dot(direction, basis.t), dot(direction, basis.b), dot(direction, basis.n));
}

public float3 openPBRLocalToWorld(const OpenPBRBasis basis, const float3 direction)
{
    return direction.x * basis.t + direction.y * basis.b + direction.z * basis.n;
}
} // namespace Metallic.OpenPBR
