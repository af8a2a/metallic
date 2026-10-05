#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_dispersion_utils.h. Modified by Metallic, 2026.
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

public static const float OpenPBRFraunhoferLineCnm = 656.3f;
public static const float OpenPBRFraunhoferLineDnm = 589.3f; // average of sodium lines D1 and D2
public static const float OpenPBRFraunhoferLinednm = 587.6f; // helium line D3
public static const float OpenPBRFraunhoferLineFnm = 486.1f;

public static const float OpenPBRDispersionWavelengthLongnm = OpenPBRFraunhoferLineCnm;
public static const float OpenPBRDispersionWavelengthMediumnm =
    OpenPBRFraunhoferLinednm; // Abbe originally used D, but today d is more common.
public static const float OpenPBRDispersionWavelengthShortnm = OpenPBRFraunhoferLineFnm;

// Converts OpenPBR "dispersion" parameter to Abbe number V_d, the reciprocal of the relative mean dispersion:
// V_d = (n_d - 1) / (n_F - n_C) (Abbe originally used D, but today d is more common). The mapping is
// V_d = 20 / "dispersion". This maps all of the realistic glass Abbe numbers, which range from about
// 20 to 95, into [0, 1], where 0 corresponds to no dispersion (infinite Abbe number) and 1 corresponds
// to the highest amount of realistic dispersion (Abbe number 20). By being based on the reciprocal of the
// Abbe number, the parameterization ensures that visible color spread increases approximately linearly
// with the parameter (the reciprocal of the Abbe number is proportional to the change in the index across
// the spectrum and the angle of refraction varies approximately linearly with the index at these scales),
// and the inverse mapping from Abbe number takes the same simple form (the parameter can easily be
// derived from a measured Abbe number V_d by simply taking 20 / V_d). The mapping from Abbe number to
// wavelength-dependent indexes is based on the basic form of Cauchy's equation (which is about the best
// we can do without additional information such as the partial dispersions).

public float openPBRAbbeNumberVD(const float dispersion_param) { return 20.0f / dispersion_param; }

// A and B are derived from n_d and V_d in such a way that the resultant
// indexes have an Abbe number of exactly V_d and n_lambda(A, B, d) = n_d.

public float openPBRCauchyEquationB(const float n_d, const float V_d)
{
    return (n_d - 1.0f) / (V_d * (openPBRSquare(1.0f / OpenPBRDispersionWavelengthShortnm) -
                                  openPBRSquare(1.0f / OpenPBRDispersionWavelengthLongnm)));
}

public float openPBRCauchyEquationA(const float n_d, const float B)
{
    return n_d - B * openPBRSquare(1.0f / OpenPBRDispersionWavelengthMediumnm);
}

public float openPBRCauchyEquation(const float A, const float B, const float lambda_nm)
{
    const float n_lambda = A + B * openPBRSquare(1.0f / lambda_nm);
    return n_lambda;
}

// Applies dispersion to an IOR based on the value of the OpenPBR "dispersion" parameter.
// Values of "dispersion" in [0, 1] map to plausible Abbe numbers in [95+, 20].
// Valid values of the "dispersion" parameter are those in [0, ~20+].
// Dispersion 0 corresponds to no dispersion at all and should be handled separately by the caller.
// IOR 1 cannot produce dispersion and should also be handled separately by the caller.

public float openPBRDispersionAdjustedIor(const float original_ior, const float dispersion_param, const float wavelength_nm)
{

    const bool original_ior_was_less_than_one = original_ior < 1.0f;
    const float ior_above_one = original_ior_was_less_than_one ? 1.0f / original_ior : original_ior;

    const float V_d = openPBRAbbeNumberVD(dispersion_param);
    const float B = openPBRCauchyEquationB(ior_above_one, V_d);
    const float A = openPBRCauchyEquationA(ior_above_one, B);

    const float adjusted_ior = openPBRCauchyEquation(A, B, wavelength_nm);

    return original_ior_was_less_than_one ? 1.0f / adjusted_ior : adjusted_ior;
}
} // namespace Metallic.OpenPBR
