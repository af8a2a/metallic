#ifndef METALLIC_OPENPBR_CLOSURE
#define METALLIC_OPENPBR_CLOSURE

// Include after OpenPBRModule.hlsli and the program's LUT provider.
// No material resources, hit records, texture provider or renderer are needed.
import MaterialProgram;
using Metallic.Material;

typealias OpenPBRSamplingContext = SurfaceSamplingContext;

struct OpenPBRPreparedClosure : IWeightedPreparedSurfaceClosure
{
    OpenPBR_PreparedBsdf bsdf;
    float3 normalWS;
    float transmissionEta;
    bool validTransport;

    float3 shadingNormal() { return normalWS; }
    override float3 surfaceEmission(float3 resolvedEmission) { return emission(); }

    float3 emission() { return validTransport ? bsdf.emission : float3(0); }

    override float3 evalProjected(float3 wiWS)
    {
        if (!validTransport) { return float3(0); }
        return openpbr_get_sum_of_diffuse_specular(openpbr_eval(bsdf, wiWS));
    }

    override float pdf(float3 wiWS) { return validTransport ? openpbr_pdf(bsdf, wiWS) : 0.0; }

    BSDFEval eval(float3 wiWS)
    {
        BSDFEval result = (BSDFEval)0;
        float cosine = abs(dot(normalWS, wiWS));
        if (!validTransport || cosine <= 0.0) { return result; }
        result.f = evalProjected(wiWS) / cosine;
        result.pdf = pdf(wiWS);
        // Directional eval is the sum of continuous lobes, not a sampled event.
        if (any(result.f != float3(0))) {
            result.flags = dot(normalWS, wiWS) * dot(normalWS, bsdf.view_direction) < 0.0
                ? kBSDFTransmission : kBSDFReflection;
        }
        return result;
    }

    override BSDFWeightSample sampleWeighted(float3 random)
    {
        BSDFWeightSample result = (BSDFWeightSample)0;
        result.eta = 1.0;
        if (!validTransport) { return result; }
        float3 direction;
        OpenPBR_DiffuseSpecular weight;
        float density;
        OpenPBR_BsdfLobeType eventType;
        openpbr_sample(bsdf, random, direction, weight, density, eventType);
        // Vendor outputs other than pdf are undefined for a failed sample.
        if (density <= 0.0) { return result; }
        result.wiWS = direction;
        result.weight = openpbr_get_sum_of_diffuse_specular(weight);
        result.pdf = density;
        result.flags = 0u;
        if (eventType & OpenPBR_BsdfLobeTypeReflection) { result.flags |= kBSDFReflection; }
        if (eventType & OpenPBR_BsdfLobeTypeTransmission) { result.flags |= kBSDFTransmission; }
        if (eventType & OpenPBR_BsdfLobeTypeDiffuse) { result.flags |= kBSDFDiffuse; }
        if (eventType & OpenPBR_BsdfLobeTypeGlossy) { result.flags |= kBSDFGlossy; }
        if (eventType & OpenPBR_BsdfLobeTypeSpecular) { result.flags |= kBSDFDelta; }
        if (result.flags & kBSDFTransmission) { result.eta = transmissionEta; }
        return result;
    }

    BSDFSample sample(float3 random)
    {
        BSDFWeightSample weighted = sampleWeighted(random);
        BSDFSample result = (BSDFSample)0;
        result.eta = 1.0;
        float cosine = abs(dot(normalWS, weighted.wiWS));
        if (weighted.pdf <= 0.0 || cosine <= 0.0) { return result; }
        result.wiWS = weighted.wiWS;
        result.f = weighted.weight * weighted.pdf / cosine;
        result.pdf = weighted.pdf;
        result.eta = weighted.eta;
        result.flags = weighted.flags;
        return result;
    }
};

struct OpenPBRClosure : ISurfaceClosure
{
    typealias Prepared = OpenPBRPreparedClosure;
    OpenPBR_ResolvedInputs inputs;
    float occlusion;

    override Prepared prepare(float3 woWS, TransportMode transportMode, OpenPBRSamplingContext sampling)
    {
        Prepared result;
        // Preserve the vendor's existing camera-path convention. Its API has
        // no adjoint mode: fail closed for Importance, never pretend support.
        result.validTransport = transportMode == TransportMode.Radiance;
        result.bsdf = openpbr_prepare(inputs, sampling.throughput, sampling.wavelengths, sampling.exteriorIor, woWS);
        result.normalWS = inputs.geometry_basis.n;
        // Use the exact clamped, specular-weight-adjusted refraction IOR from
        // preparation. Dispersion is disabled by this production program.
        result.transmissionEta = inputs.geometry_thin_walled ? 1.0
            : 1.0 / result.bsdf.fuzz_lobe.coating_lobe.base_lobe.specular_lobe.eta_t_over_eta_i.r;
        return result;
    }

    Prepared prepare(float3 woWS, TransportMode transportMode)
    {
        OpenPBRSamplingContext sampling;
        sampling.throughput = float3(1);
        sampling.wavelengths = OpenPBR_BaseRgbWavelengths_nm;
        sampling.exteriorIor = OpenPBR_VacuumIor;
        return prepare(woWS, transportMode, sampling);
    }
};

// Canonical Phase 10 family name; the implementation is the native OpenPBR module.
typealias OpenPBRCompositeClosure = OpenPBRClosure;

#endif
