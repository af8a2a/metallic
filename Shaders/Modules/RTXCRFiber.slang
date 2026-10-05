#language slang 2026
module RTXCRFiber;
import RTXCRHair;
import FiberMaterial;
import MaterialProgram;

namespace Metallic.Material
{


public interface IRTXCRHairParameterProvider
{
    [mutating] public Metallic.RTXCR.HairMaterialData load(MaterialInstanceRef instance);
};

public struct RTXCRChiangPreparedClosure : IPreparedFiberClosure
{
    public Metallic.RTXCR.HairMaterialInteraction interaction;
    public float3x3 frame;
    public float3 woLocal;
    public bool valid;

    public float3 evalProjected(float3 wiWS)
    {
        return valid ? Metallic.RTXCR.hairChiangBSDFEval(interaction, mul(frame, wiWS), woLocal) : float3(0);
    }

    public FiberSample sampleWeighted(float4 random)
    {
        FiberSample result = (FiberSample)0;
        if (!valid) { return result; }
        float2 u[2] = { random.xy, random.zw };
        float3 wi, numerator;
        float density;
        uint eventType;
        if (!Metallic.RTXCR.sampleChiangBSDF(interaction, woLocal, u, wi, density, numerator, eventType)) { return result; }
        result.wiWS = mul(transpose(frame), wi);
        result.pdf = density;
        result.weight = max(numerator, float3(0)) / max(density, 0.000001);
        result.eventType = eventType;
        return result;
    }
};

public struct RTXCRChiangClosure : IFiberClosure
{
    public typealias Prepared = RTXCRChiangPreparedClosure;
    public Metallic.RTXCR.HairMaterialData parameters;
    public float3 normalWS;
    public float3 tangentWS;

    public Prepared prepare(float3 woWS, TransportMode mode)
    {
        Metallic.RTXCR.HairInteractionSurface surface;
        surface.shadingNormal = normalWS;
        surface.tangent = tangentWS;
        surface.incidentRayDirection = woWS;
        Prepared result;
        result.frame = float3x3(tangentWS, cross(normalWS, tangentWS), normalWS);
        result.woLocal = mul(result.frame, woWS);
        // Match the original camera-path implementation. Importance transport
        // is explicitly unsupported, rather than silently using Surface rules.
        float3 transverseView = woWS - dot(woWS, tangentWS) * tangentWS;
        // Exactly axial views have no azimuthal frame. Reject this measure-zero
        // case before upstream normalize(0); never propagate NaNs to lighting.
        result.valid = mode == TransportMode.Radiance && dot(transverseView, transverseView) > 1e-12;
        result.interaction = (Metallic.RTXCR.HairMaterialInteraction)0;
        if (result.valid) { result.interaction = Metallic.RTXCR.createHairMaterialInteraction(parameters, surface); }
        return result;
    }
};

public struct RTXCRChiangMaterialProgram<TParameters : IRTXCRHairParameterProvider> : IFiberMaterialProgram
{
    public typealias Closure = RTXCRChiangClosure;
    public TParameters parameters;

    [mutating]
    public FiberMaterialResult<Closure> evaluate(FiberMaterialContext context, MaterialInstanceRef instance)
    {
        FiberMaterialResult<Closure> result;
        result.closure.parameters = parameters.load(instance);
        result.closure.normalWS = context.normalWS;
        result.closure.tangentWS = context.tangentWS;
        result.emission = 0;
        return result;
    }
};
} // namespace Metallic.Material
