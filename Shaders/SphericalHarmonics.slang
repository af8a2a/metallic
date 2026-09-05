#ifndef METALLIC_SPHERICAL_HARMONICS
#define METALLIC_SPHERICAL_HARMONICS

// Real SH through l=2: three bands, nine RGB coefficients, Y-up lat-long maps.
float shBasis(uint index, float3 d)
{
    switch (index) {
    case 0u: return 0.2820947918;
    case 1u: return 0.4886025119 * d.y;
    case 2u: return 0.4886025119 * d.z;
    case 3u: return 0.4886025119 * d.x;
    case 4u: return 1.0925484306 * d.x * d.y;
    case 5u: return 1.0925484306 * d.y * d.z;
    case 6u: return 0.3153915653 * (3.0 * d.z * d.z - 1.0);
    case 7u: return 1.0925484306 * d.x * d.z;
    default: return 0.5462742153 * (d.x * d.x - d.y * d.y);
    }
}

float shCosineConvolution(uint index)
{
    const float pi = 3.14159265358979323846;
    return index == 0u ? pi : (index < 4u ? 2.0 * pi / 3.0 : pi / 4.0);
}

#endif
