#language slang 2026
implementing OpenPBR;

// Native Slang port of Adobe openpbr_constants.h. Modified by Metallic, 2026.
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

public static const float OpenPBRFloatMin = 1.175494351e-38f;
public static const float OpenPBRLargestFloatBelowOne = 0.999999940f; // 1 - 2^-24

public static const float OpenPBRPi = 3.14159265f;
public static const float OpenPBRTwoPi = 6.28318531f; // 2 * Pi
public static const float OpenPBRHalfPi = 1.57079633f; // Pi / 2
public static const float OpenPBRPiOverFour = 0.785398163f; // Pi / 4
public static const float OpenPBRRcpPi = 0.318309886f; // 1 / Pi
public static const float OpenPBRRcpTwoPi = 0.159154943f; // 1 / (2 * Pi) = 0.5 / Pi
public static const float OpenPBRRcpFourPi = 0.0795774715f; // 1 / (4 * Pi) = 0.25 / Pi

public static const int OpenPBRNumRGBChannels = 3;
public static const float OpenPBRVacuumIor = 1.0f;
public static const float3 OpenPBRBaseRGBWavelengthsNm = float3(620.0f, 540.0f, 450.0f);
} // namespace Metallic.OpenPBR
