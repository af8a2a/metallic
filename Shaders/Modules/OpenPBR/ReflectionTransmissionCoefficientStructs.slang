#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_reflection_transmission_coefficient_structs.h. Modified by Metallic, 2026.
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

// The reflection/transmission structs below are used by the reflection and transmission coefficient classes
// to return multiple coefficients and probabilities together. This allows the classes to avoid redundant
// calculations that would occur if computing and returning the reflection and transmission coefficients separately.

public struct OpenPBRAllCoefficients
{
    public float3 reflection_coefficient;
    public float3 transmission_coefficient;
};

public struct OpenPBRAllCoefficientsAndProbabilities
{
    public float3 reflection_coefficient;
    public float3 transmission_coefficient;
    public float reflection_probability;
    public float transmission_probability;
};
} // namespace Metallic.OpenPBR
