#language slang 2026
implementing Material;
import MaterialProgram;
#if METALLIC_CUSTOM_COVERAGE
import MaterialCoverageGenerated;
#endif

namespace Metallic.Material
{
public struct MaterialCoverageParameters
{
    public float baseAlpha;
    public float cutoff;
    public float alphaMode;
    public bool hasProgram;
    public MaterialTextureInfo baseColorTexture;
};

public MaterialCoverageParameters sceneMaterialCoverageParameters(SceneMaterialData data)
{
    MaterialCoverageParameters result;
    result.baseAlpha = data.baseColor.a;
    result.cutoff = data.params.z;
    result.alphaMode = data.textureParams.w;
    result.hasProgram = data.attenuationColor.w > 0.5;
    result.baseColorTexture = data.baseColorTexture;
    return result;
}

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
#if METALLIC_CUSTOM_COVERAGE
        return coverageFromOpacity(context, Metallic.MaterialCoverage.evaluateGeneratedCoverage(
            inputs, header.y, instance.parameterOffset, context.uv, context.baseAlpha));
#else
        return coverageFromOpacity(context, context.baseAlpha);
#endif
    }
};
} // namespace Metallic.Material
