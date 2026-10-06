#ifndef METALLIC_SCENE_SURFACE_CLOSURE
#define METALLIC_SCENE_SURFACE_CLOSURE

// Closed scene family dispatch, compiled away in statically binned primary hits.
// No resources or material evaluation live in these resolved/prepared values.
#if METALLIC_CUSTOM_MATERIALS
import SlabClosure;

struct ScenePreparedSurfaceClosure : IWeightedPreparedSurfaceClosure
{
    SceneOpenPBRPreparedClosure openPBR;
    SlabPreparedClosure slab;
    uint family;
    float3 resolvedEmission;
    float3 shadingNormal() { return family == 0u ? openPBR.shadingNormal() : slab.shadingNormal(); }
    float3 emission() { return family == 0u ? openPBR.emission() : resolvedEmission; }
    override float3 surfaceEmission(float3 value) { return emission(); }
    BSDFEval eval(float3 wi) { return family == 0u ? openPBR.eval(wi) : slab.eval(wi); }
    BSDFSample sample(float3 random) { return family == 0u ? openPBR.sample(random) : slab.sample(random); }
    override float3 evalProjected(float3 wi) { return family == 0u ? openPBR.evalProjected(wi) : slab.evalProjected(wi); }
    override float pdf(float3 wi) { return family == 0u ? openPBR.pdf(wi) : slab.pdf(wi); }
    override BSDFWeightSample sampleWeighted(float3 random)
    {
        return family == 0u ? openPBR.sampleWeighted(random) : slab.sampleWeighted(random);
    }
};

struct SceneSurfaceClosure : ISurfaceClosure
{
    typealias Prepared = ScenePreparedSurfaceClosure;
    // OpenPBR inputs also supply conservative diagnostic/guide summaries for Slab.
    OpenPBRResolvedInputs inputs;
    float occlusion;
    uint family; // 0 OpenPBR, 1 SingleSlab, 2 DualSlab.
    DualSlabClosure slab;
    float3 resolvedEmission;
    override Prepared prepare(float3 wo, TransportMode mode, SurfaceSamplingContext sampling)
    {
        Prepared result = (Prepared)0;
        result.family = family;
        result.resolvedEmission = resolvedEmission;
        if (family == 0u) {
            SceneOpenPBRClosure closure;
            closure.context = SceneOpenPBRContext();
            closure.inputs = inputs;
            closure.occlusion = occlusion;
            result.openPBR = closure.prepare(wo, mode, sampling);
        } else if (family == 1u) {
            SingleSlabClosure closure;
            closure.slab = slab.first;
            closure.normal = slab.normal;
            result.slab = closure.prepare(wo, mode);
        } else {
            result.slab = slab.prepare(wo, mode);
        }
        return result;
    }
    Prepared prepare(float3 wo, TransportMode mode)
    {
        SurfaceSamplingContext sampling;
        sampling.throughput = float3(1);
        sampling.wavelengths = OpenPBRBaseRGBWavelengthsNm;
        sampling.exteriorIor = OpenPBRVacuumIor;
        return prepare(wo, mode, sampling);
    }
};
#else
typealias SceneSurfaceClosure = SceneOpenPBRClosure;
#endif
#endif
