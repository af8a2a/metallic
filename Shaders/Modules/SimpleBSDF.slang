#language slang 2026
module SimpleBSDF;

import MaterialProgram;
import VkGltfBSDF;

namespace Metallic.Material
{
// The full nvpro BSDF is a stochastic estimator domain. It deliberately does
// not implement IPreparedSurfaceClosure: that interface requires a deterministic
// unprojected value and the complete sampling PDF. No renderer resources here.
public interface IPreparedSimpleBSDFClosure
{
    // Both directions point away from the surface. One explicit random tuple
    // produces the projected contribution and its selected-lobe PDF together.
    public Metallic.VkGltf.BSDFEvaluation evalProjected(float3 wiWS, float3 random);
    // Preserve native weight, event bits and the -1 Dirac PDF sentinel.
    public Metallic.VkGltf.BSDFSample sampleWeighted(float3 random);
    public float3 shadingNormal();
};

public interface ISimpleBSDFClosure
{
    public associatedtype Prepared : IPreparedSimpleBSDFClosure;
    public Prepared prepare(float3 woWS, TransportMode mode);
};

public struct SimpleBSDFMaterialResult<TClosure : ISimpleBSDFClosure>
{
    public TClosure closure;
    public float3 emission;
    public uint flags;
};

public interface ISimpleBSDFMaterialProgram
{
    public associatedtype Closure : ISimpleBSDFClosure;
    [mutating]
    public SimpleBSDFMaterialResult<Closure> evaluate(
        SurfaceMaterialContext context, MaterialInstanceRef instance);
};

// Like the independent Fiber domain, reuse the concrete Material<TProgram>
// owner without weakening the existing Surface scattering contract.
public extension<TProgram : ISimpleBSDFMaterialProgram> Material<TProgram>
{
    [mutating]
    public SimpleBSDFMaterialResult<TProgram.Closure> evaluate(
        SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        return program.evaluate(context, instance);
    }
};

// Only the material program owns the parameter/texture provider and its RNG.
// resolve() returns fully resolved numeric values, without any resource handles.
public interface ISimpleBSDFSource
{
    [mutating]
    public Metallic.VkGltf.PreparedMaterial resolve(
        SurfaceMaterialContext context, MaterialInstanceRef instance);
};

public struct SimpleBSDFPreparedClosure : IPreparedSimpleBSDFClosure
{
    public Metallic.VkGltf.PreparedMaterial inputs;
    public float3 woWS;
    public bool validRadiance;

    public float3 shadingNormal() { return inputs.N; }

    public Metallic.VkGltf.BSDFEvaluation evalProjected(float3 wiWS, float3 random)
    {
        if (!validRadiance) { return Metallic.VkGltf.BSDFEvaluation(); }
        return Metallic.VkGltf.evaluateBSDF(inputs, woWS, wiWS, random);
    }

    public Metallic.VkGltf.BSDFSample sampleWeighted(float3 random)
    {
        // Zero initializes the native absorb event. Do not reject valid native
        // impulse samples based on pdf <= 0 or reconstruct f from their weight.
        if (!validRadiance) { return Metallic.VkGltf.BSDFSample(); }
        return Metallic.VkGltf.sampleBSDF(inputs, woWS, random);
    }
};

public struct SimpleBSDFClosure : ISimpleBSDFClosure
{
    public typealias Prepared = SimpleBSDFPreparedClosure;
    public Metallic.VkGltf.PreparedMaterial inputs;

    public Prepared prepare(float3 woWS, TransportMode mode)
    {
        Prepared result;
        result.inputs = inputs;
        result.woWS = woWS;
        // The source renderer defines camera/radiance transport only. Preparing
        // must not normalize/remap inputs, evaluate lobes or consume randomness.
        result.validRadiance = mode == TransportMode.Radiance;
        return result;
    }
};

public struct SimpleBSDFMaterialProgram<TSource : ISimpleBSDFSource> : ISimpleBSDFMaterialProgram
{
    public typealias Closure = SimpleBSDFClosure;
    public TSource source;

    [mutating]
    public SimpleBSDFMaterialResult<Closure> evaluate(
        SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        SimpleBSDFMaterialResult<Closure> result;
        result.closure.inputs = source.resolve(context, instance);
        result.emission = result.closure.inputs.emissive;
        result.flags = 0u;
        return result;
    }
};

// These helpers specialize on concrete closure types at compile time. There is
// no existential dispatch, hidden RNG or independent pdf() evaluation.
public TClosure.Prepared prepareSimpleBSDF<TClosure : ISimpleBSDFClosure>(
    TClosure closure, float3 woWS, TransportMode mode)
{
    return closure.prepare(woWS, mode);
}

public Metallic.VkGltf.BSDFEvaluation evaluateSimpleBSDF<TPrepared : IPreparedSimpleBSDFClosure>(
    TPrepared prepared, float3 wiWS, float3 random)
{
    return prepared.evalProjected(wiWS, random);
}

public Metallic.VkGltf.BSDFSample sampleSimpleBSDF<TPrepared : IPreparedSimpleBSDFClosure>(
    TPrepared prepared, float3 random)
{
    return prepared.sampleWeighted(random);
}
} // namespace Metallic.Material
