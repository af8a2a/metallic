#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_complex.h. Modified by Metallic, 2026.
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

// Complex arithmetic functions

// A complex number is represented by a vec2
// The vec2's first component represents the real component
// The vec2's second component represents the imaginary component
// The constructor complex(x, y) can then be used to create a complex number

// Complex negate

public float2 openPBRComplexNegate(const float2 z) { return -z; }

// Function to create a complex exponential e^(i * theta)

public float2 openPBRComplexExpI(const float theta) { return float2(cos(theta), sin(theta)); }

// Complex addition: z1 + z2

public float2 openPBRComplexAdd(const float2 z1, const float2 z2) { return z1 + z2; }

// Complex subtraction: z1 - z2

public float2 openPBRComplexSubtract(const float2 z1, const float2 z2) { return z1 - z2; }

// Complex multiplication with scalar: scalar * z

public float2 openPBRComplexScalarMultiply(const float scalar, const float2 z) { return float2(scalar * z.x, scalar * z.y); }

// Complex multiplication: z1 * z2

public float2 openPBRComplexMultiply(const float2 z1, const float2 z2)
{
    return float2(z1.x * z2.x - z1.y * z2.y, z1.x * z2.y + z1.y * z2.x);
}

// Complex division: z1 / z2

public float2 openPBRComplexDivide(const float2 z1, const float2 z2, const float min_denom_mag, const float2 fallback)
{
    const float denom = dot(z2, z2); // z2.x * z2.x + z2.y * z2.y
    if (denom < min_denom_mag * min_denom_mag) {
        // Return the fallback to avoid division by zero
        return fallback;
    }
    // Precompute reciprocal denominators to minimize number of divisions
    const float rcp_denom = 1.0f / denom;
    return float2((z1.x * z2.x + z1.y * z2.y) * rcp_denom, (z1.y * z2.x - z1.x * z2.y) * rcp_denom);
}

// Complex square root function

public float2 openPBRComplexSqrt(const float2 z)
{
    const float r = length(z);

    const float a = sqrt(max((r + z.x) * 0.5f, 0.0f));
    const float b = openPBRSignNonzero(z.y) * sqrt(max((r - z.x) * 0.5f, 0.0f));

    return float2(a, b);
}

// Complex magnitude squared: |z|^2

public float openPBRComplexMagnitudeSquared(const float2 z)
{
    return dot(z, z); // z.x * z.x + z.y * z.y
}
} // namespace Metallic.OpenPBR
