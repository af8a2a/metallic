#pragma once

#include "Runtime/Render/GAPI/RHI.h"
#include <cstdint>
#include <string_view>

namespace metallic::render::openpbr {
// Offline-integrated Adobe data (Apache-2.0, see External/openpbr-bsdf/LICENSE).
// These embedded texture payloads need no generator, asset search path, or runtime
// integration/expansion. R16_UNORM decodes the original uint16 / 65535 encoding.
using OpenPBRLutScalar = uint16_t;

struct OpenPBRVec3 {
    float x;
    float y;
    float z;
    float w = 1.0f;
};

constexpr OpenPBRVec3 vec3(float x, float y, float z)
{
    return OpenPBRVec3{x, y, z};
}

inline constexpr OpenPBRLutScalar kOpenPBRIdealDielectricEnergyComplement[] = {
#include "../../../../External/openpbr-bsdf/impl/data/openpbr_ideal_dielectric_energy_complement_data.h"
};

inline constexpr OpenPBRLutScalar kOpenPBRIdealDielectricAverageEnergyComplement[] = {
#include "../../../../External/openpbr-bsdf/impl/data/openpbr_ideal_dielectric_avg_energy_complement_data.h"
};

inline constexpr OpenPBRLutScalar kOpenPBRIdealDielectricReflectionRatio[] = {
#include "../../../../External/openpbr-bsdf/impl/data/openpbr_ideal_dielectric_reflection_ratio_data.h"
};

inline constexpr OpenPBRLutScalar kOpenPBROpaqueDielectricEnergyComplement[] = {
#include "../../../../External/openpbr-bsdf/impl/data/openpbr_opaque_dielectric_energy_complement_data.h"
};

inline constexpr OpenPBRLutScalar kOpenPBROpaqueDielectricAverageEnergyComplement[] = {
#include "../../../../External/openpbr-bsdf/impl/data/openpbr_opaque_dielectric_avg_energy_complement_data.h"
};

inline constexpr OpenPBRLutScalar kOpenPBRIdealMetalEnergyComplement[] = {
#include "../../../../External/openpbr-bsdf/impl/data/openpbr_ideal_metal_energy_complement_data.h"
};

inline constexpr OpenPBRLutScalar kOpenPBRIdealMetalAverageEnergyComplement[] = {
#include "../../../../External/openpbr-bsdf/impl/data/openpbr_ideal_metal_avg_energy_complement_data.h"
};

inline constexpr OpenPBRVec3 kOpenPBRLtc[] = {
#include "../../../../External/openpbr-bsdf/impl/data/openpbr_ltc_data.h"
};

struct LutPayload
{
    const void* pixels;
    uint64_t byteSize;
    Format format;
    uint32_t width, height, depth;
    std::string_view label;
};

// Ordered by Adobe LUT ID, not by texture dimensionality. LTC stays full float
// because its signed transform coefficients are not normalized energy values.
inline constexpr LutPayload kLutPayloads[] = {
    {kOpenPBRIdealDielectricEnergyComplement, sizeof(kOpenPBRIdealDielectricEnergyComplement), Format::R16Unorm, 32, 32, 32, "OpenPBR IdealDielectricEnergyComplement LUT"},
    {kOpenPBRIdealDielectricAverageEnergyComplement, sizeof(kOpenPBRIdealDielectricAverageEnergyComplement), Format::R16Unorm, 32, 32, 1, "OpenPBR IdealDielectricAverageEnergyComplement LUT"},
    {kOpenPBRIdealDielectricReflectionRatio, sizeof(kOpenPBRIdealDielectricReflectionRatio), Format::R16Unorm, 32, 32, 1, "OpenPBR IdealDielectricReflectionRatio LUT"},
    {kOpenPBROpaqueDielectricEnergyComplement, sizeof(kOpenPBROpaqueDielectricEnergyComplement), Format::R16Unorm, 32, 32, 32, "OpenPBR OpaqueDielectricEnergyComplement LUT"},
    {kOpenPBROpaqueDielectricAverageEnergyComplement, sizeof(kOpenPBROpaqueDielectricAverageEnergyComplement), Format::R16Unorm, 32, 32, 1, "OpenPBR OpaqueDielectricAverageEnergyComplement LUT"},
    {kOpenPBRIdealMetalEnergyComplement, sizeof(kOpenPBRIdealMetalEnergyComplement), Format::R16Unorm, 32, 32, 1, "OpenPBR IdealMetalEnergyComplement LUT"},
    {kOpenPBRIdealMetalAverageEnergyComplement, sizeof(kOpenPBRIdealMetalAverageEnergyComplement), Format::R16Unorm, 32, 1, 1, "OpenPBR IdealMetalAverageEnergyComplement LUT"},
    {kOpenPBRLtc, sizeof(kOpenPBRLtc), Format::RGBA32Sfloat, 32, 32, 1, "OpenPBR Ltc LUT"},
};
static_assert(sizeof(OpenPBRVec3) == 4 * sizeof(float));
static_assert(std::size(kOpenPBRIdealDielectricEnergyComplement) == 32768);
static_assert(std::size(kOpenPBRIdealDielectricAverageEnergyComplement) == 1024);
static_assert(std::size(kOpenPBRIdealDielectricReflectionRatio) == 1024);
static_assert(std::size(kOpenPBROpaqueDielectricEnergyComplement) == 32768);
static_assert(std::size(kOpenPBROpaqueDielectricAverageEnergyComplement) == 1024);
static_assert(std::size(kOpenPBRIdealMetalEnergyComplement) == 1024);
static_assert(std::size(kOpenPBRIdealMetalAverageEnergyComplement) == 32);
static_assert(std::size(kOpenPBRLtc) == 1024);
} // namespace metallic::render::openpbr
