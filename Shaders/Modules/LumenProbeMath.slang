module LumenProbeMath;

namespace Metallic.Lumen
{
public static const float kLumenProbePi = 3.14159265358979323846f;

// UE 5.7.4 RandomPCG.ush:12 and MonteCarlo.ush:65.
public uint3 lumenRand3DPCG16(int3 p)
{
    uint3 v = uint3(p);
    v = v * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v >> 16u;
}

public float2 lumenHammersley16(uint index, uint numSamples, uint2 random)
{
    float e1 = frac(float(index) / numSamples + float(random.x) * (1.0f / 65536.0f));
    float e2 = float((reversebits(index) >> 16u) ^ random.y) * (1.0f / 65536.0f);
    return float2(e1, e2);
}

// UE 5.7.4 MonteCarlo.ush:159 (Clarberg equal-area sphere mapping).
public float3 lumenEquiAreaSphericalMapping(float2 uv)
{
    uv = 2.0f * uv - 1.0f;
    float d = 1.0f - (abs(uv.x) + abs(uv.y));
    float r = 1.0f - abs(d);
    float phi = r == 0.0f ? 0.0f : (kLumenProbePi / 4.0f) * ((abs(uv.y) - abs(uv.x)) / r + 1.0f);
    float f = r * sqrt(2.0f - r * r);
    return float3(f * sign(uv.x) * abs(cos(phi)), f * sign(uv.y) * abs(sin(phi)), sign(d) * (1.0f - r * r));
}

// UE 5.7.4 MonteCarlo.ush:178.
public float2 lumenInverseEquiAreaSphericalMapping(float3 direction)
{
    direction = normalize(direction);
    float3 absDir = abs(direction);
    float r = sqrt(1.0f - absDir.z);
    const float epsilon = 5.42101086243e-20f;
    float x = min(absDir.x, absDir.y) / (max(absDir.x, absDir.y) + epsilon);

    const float t1 = 0.406758566246788489601959989e-5f;
    const float t2 = 0.636226545274016134946890922156f;
    const float t3 = 0.61572017898280213493197203466e-2f;
    const float t4 = -0.247333733281268944196501420480f;
    const float t5 = 0.881770664775316294736387951347e-1f;
    const float t6 = 0.419038818029165735901852432784e-1f;
    const float t7 = -0.251390972343483509333252996350e-1f;
    float phi = t6 + t7 * x;
    phi = t5 + phi * x;
    phi = t4 + phi * x;
    phi = t3 + phi * x;
    phi = t2 + phi * x;
    phi = t1 + phi * x;

    phi = absDir.x < absDir.y ? 1.0f - phi : phi;
    float2 uv = float2(r - phi * r, phi * r);
    uv = direction.z < 0.0f ? 1.0f - uv.yx : uv;
    uv = asfloat(asuint(uv) ^ (asuint(direction.xy) & 0x80000000u));
    return uv * 0.5f + 0.5f;
}

// UE 5.7.4 LumenScreenSpaceBentNormal.ush:19, with short range AO disabled.
public float3 lumenDistantIlluminationRescale(float3 diffuseColor)
{
    const float ao = min(1.0f, 0.999f);
    const float f0 = ao * (1.0f + (1.0f - ao) / (2.0f * sqrt(sqrt(1.0f - ao))));
    const float a = 27.576937094210385f;
    const float b = 3.3364392003423804f;
    const float f1 = a * ao * pow(1.0f - ao, 1.5f) * exp(-b * sqrt(sqrt(ao)));
    const float tau = 1.0f - f1 / max(1.0f - f0, 0.00001f);
    diffuseColor = min(diffuseColor, 0.5f);
    return f0 + diffuseColor / max(1.0f - diffuseColor * tau, 0.00001f) * f1;
}
} // namespace Metallic.Lumen
