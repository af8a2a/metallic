#language slang 2026
implementing Material;
import MaterialProgram;

namespace Metallic.Material
{
public struct CoverageContext
{
    public float2 uv;
    public float baseAlpha;
    public float cutoff;
    public float alphaMode;
};

public struct CoverageResult
{
    public float opacity;
    public bool accepted;
};

public interface ICoverageEvaluator
{
    CoverageResult evaluateCoverage(CoverageContext context, MaterialInstanceRef instance);
};

public MaterialInstanceRef coverageInstanceRef(uint materialIndex)
{
    MaterialInstanceRef result;
    result.parameterOffset = materialIndex * 80u;
    result.resourceOffset = result.parameterOffset;
    return result;
}

public CoverageResult coverageFromOpacity(CoverageContext context, float opacity)
{
    CoverageResult result;
    result.opacity = saturate(opacity);
    result.accepted = context.alphaMode < 0.5 || (context.alphaMode > 1.5
        ? result.opacity > 0.0 : result.opacity >= saturate(context.cutoff));
    return result;
}

public struct LegacyCoverageEvaluator : ICoverageEvaluator
{
    public CoverageResult evaluateCoverage(CoverageContext context, MaterialInstanceRef instance)
    {
        return coverageFromOpacity(context, context.baseAlpha);
    }
};

// Independent Coverage backend. No Surface/Closure evaluation, normals, lighting
// resources, derivatives or pass-local clocks enter this program.
public struct CoverageProgram : ICoverageEvaluator
{
    public ByteAddressBuffer inputs;

    public bool usesBaseAlpha(MaterialInstanceRef instance)
    {
        uint4 header = inputs.Load4(instance.parameterOffset);
        return header.z == 0u || (header.w & 1u) != 0u;
    }

    public CoverageResult evaluateCoverage(CoverageContext context, MaterialInstanceRef instance)
    {
        if (context.alphaMode < 0.5) { return coverageFromOpacity(context, 1.0); }
        uint4 header = inputs.Load4(instance.parameterOffset);
        if (header.z == 0u) { return coverageFromOpacity(context, context.baseAlpha); }
        float4 values[64];
        for (uint i = 0; i < min(header.z, 64u); ++i) {
            uint address = header.y + i * 32u;
            uint4 instruction = inputs.Load4(address);
            uint op = instruction.x;
            float4 value = 0.0;
            if (op == 0u) { value = asfloat(inputs.Load4(address + 16u)); }
            else if (op == 1u) { value = asfloat(inputs.Load4(instance.parameterOffset + 16u + instruction.y * 16u)); }
            else if (op == 2u) { value = float4(context.uv, 0.0, 0.0); }
            else if (op == 3u) { value = context.baseAlpha; }
            else {
                float4 a = values[instruction.y];
                if (op == 4u) { value = a + values[instruction.z]; }
                else if (op == 5u) { value = a * values[instruction.z]; }
                else if (op == 6u) { value = dot(a, values[instruction.z]); }
                else if (op == 7u) { value = lerp(a, values[instruction.z], saturate(values[instruction.w])); }
                else if (op == 8u) { value = sin(a); }
                else if (op == 9u) { value = frac(a); }
                else if (op == 10u) { value = abs(a); }
                else if (op == 11u) { value = saturate(a); }
            }
            values[i] = clamp(value, -1e6, 1e6);
        }
        return coverageFromOpacity(context, values[min(header.z, 64u) - 1u].x);
    }
};
} // namespace Metallic.Material
