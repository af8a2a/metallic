#language slang 2026
module OpenPBRClosure;

import OpenPBR;
import MaterialProgram;
using Metallic.OpenPBR;

namespace Metallic.Material
{

public struct OpenPBRPreparedClosure<TContext : IOpenPBRContext> : IWeightedPreparedSurfaceClosure
{
    // Immutable BSDF LUT context only; no material texture/instance provider.
    public TContext context;
    public OpenPBRPreparedBSDF bsdf;
    public float3 normalWS;
    public float transmissionEta;
    public bool validTransport;

    public float3 shadingNormal() { return normalWS; }
    public override float3 surfaceEmission(float3 resolvedEmission) { return emission(); }

    public float3 emission() { return validTransport ? bsdf.emission : float3(0); }

    public override float3 evalProjected(float3 wiWS)
    {
        return openPBRGetSumOfDiffuseSpecular(evalProjectedComponents(wiWS));
    }

    public OpenPBRDiffuseSpecular evalProjectedComponents(float3 wiWS)
    {
        return validTransport ? openPBREval(context, bsdf, wiWS) : openPBRMakeZeroDiffuseSpecular();
    }

    public override float pdf(float3 wiWS) { return validTransport ? openPBRPdf(context, bsdf, wiWS) : 0.0; }

    public BSDFEval eval(float3 wiWS)
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

    public override BSDFWeightSample sampleWeighted(float3 random)
    {
        OpenPBRDiffuseSpecular components;
        return sampleWeightedComponents(random, components);
    }

    // Keep the vendor's mixture PDF and weights. A sampled event flag does not
    // identify all contributing lobes: one direction can carry both components.
    public BSDFWeightSample sampleWeightedComponents(float3 random, out OpenPBRDiffuseSpecular components)
    {
        BSDFWeightSample result = (BSDFWeightSample)0;
        components = openPBRMakeZeroDiffuseSpecular();
        result.eta = 1.0;
        if (!validTransport) { return result; }
        float3 direction;
        OpenPBRDiffuseSpecular weight;
        float density;
        uint eventType;
        openPBRSample(context, bsdf, random, direction, weight, density, eventType);
        // Vendor outputs other than pdf are undefined for a failed sample.
        if (density <= 0.0) { return result; }
        components = weight;
        result.wiWS = direction;
        result.weight = openPBRGetSumOfDiffuseSpecular(weight);
        result.pdf = density;
        result.flags = 0u;
        if (eventType & OpenPBRBSDFLobeTypeReflection) { result.flags |= kBSDFReflection; }
        if (eventType & OpenPBRBSDFLobeTypeTransmission) { result.flags |= kBSDFTransmission; }
        if (eventType & OpenPBRBSDFLobeTypeDiffuse) { result.flags |= kBSDFDiffuse; }
        if (eventType & OpenPBRBSDFLobeTypeGlossy) { result.flags |= kBSDFGlossy; }
        if (eventType & OpenPBRBSDFLobeTypeSpecular) { result.flags |= kBSDFDelta; }
        if (result.flags & kBSDFTransmission) { result.eta = transmissionEta; }
        return result;
    }

    public BSDFSample sample(float3 random)
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

public struct OpenPBRClosure<TContext : IOpenPBRContext> : ISurfaceClosure
{
    public typealias Prepared = OpenPBRPreparedClosure<TContext>;
    public TContext context;
    public OpenPBRResolvedInputs inputs;
    public float occlusion;

    public override Prepared prepare(float3 woWS, TransportMode transportMode, SurfaceSamplingContext sampling)
    {
        Prepared result;
        result.context = context;
        // Preserve the vendor's existing camera-path convention. Its API has
        // no adjoint mode: fail closed for Importance, never pretend support.
        result.validTransport = transportMode == TransportMode.Radiance;
        result.bsdf = openPBRPrepare(context, inputs, sampling.throughput, sampling.wavelengths, sampling.exteriorIor, woWS);
        result.normalWS = inputs.geometry_basis.n;
        // Use the exact clamped, specular-weight-adjusted refraction IOR from
        // preparation. Dispersion is disabled by this production program.
        result.transmissionEta = inputs.geometry_thin_walled ? 1.0
            : 1.0 / result.bsdf.fuzz_lobe.coating_lobe.base_lobe.specular_lobe.eta_t_over_eta_i.r;
        return result;
    }

    public Prepared prepare(float3 woWS, TransportMode transportMode)
    {
        SurfaceSamplingContext sampling;
        sampling.throughput = float3(1);
        sampling.wavelengths = OpenPBRBaseRGBWavelengthsNm;
        sampling.exteriorIor = OpenPBRVacuumIor;
        return prepare(woWS, transportMode, sampling);
    }
};

// Canonical Phase 10 family name; the implementation is the native OpenPBR module.
public typealias OpenPBRCompositeClosure<TContext : IOpenPBRContext> = OpenPBRClosure<TContext>;

} // namespace Metallic.Material
