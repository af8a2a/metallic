#language slang 2026
module OpenPBRClosureProbe;

import Core;
import OpenPBRProbe;
import OpenPBR;
import OpenPBRClosure;
import OpenPBRSurface;
import OpenPBRMaterialParameters;
import ScenePathTypes;
import Material;
import MaterialProgram;
import SurfaceLighting;
using Metallic;
using Metallic.Material;
using Metallic.OpenPBR;
using Metallic.SceneOpenPBR;
using Metallic.Lighting;
using Metallic.Tests;

[[vk::push_constant]] ConstantBuffer<ProbeParams> gProbe;

// Resolve all color-slot presence combinations through a custom provider with
// materialTextureCount deliberately zero, for both transfer functions and the
// debug override. No BSDF preparation is needed to inspect resolved inputs.
[shader("compute")]
[numthreads(64, 1, 1)]
void openPBRColorSlotProbeMain(uint3 id : SV_DispatchThreadID)
{
    uint lane = id.x;
    uint mask = lane & 7u;
    bool linear = (lane & 8u) != 0u;
    bool disabled = (lane & 16u) != 0u;
    bool unlit = (lane & 32u) != 0u;
    ProbeSource source;
    source.probe = gProbe;
    OpenPBRMaterialParameters material = (OpenPBRMaterialParameters)0;
    material.baseColor = float4(0.8, 0.6, 0.4, 0.37);
    material.emissive = float4(2.3, -0.1, 0.7, unlit ? 1 : 0);
    material.specular = float4(0.2, 0.7, 0.9, 0.63);
    material.params = float4(0, 0.35, 0, 0);
    material.glassParams = float4(0, 1.5, 0, 0);
    material.baseColorTexture.textureIndex = (mask & 1u) != 0u ? 3u : 0u;
    material.emissiveTexture.textureIndex = (mask & 2u) != 0u ? 3u : 0u;
    material.specularColorTexture.textureIndex = (mask & 4u) != 0u ? 3u : 0u;
    material.baseColorTexture.transform0.w = linear ? 1.0 : 0.0;
    material.emissiveTexture.transform0.w = linear ? 1.0 : 0.0;
    material.specularColorTexture.transform0.w = linear ? 1.0 : 0.0;
    HitInfo hit = (HitInfo)0;
    hit.normal = hit.geometryNormal = float3(0, 0, 1);
    hit.tangent = float4(1, 0, 0, 1);
    hit.bitangent = float3(0, 1, 0);
    hit.texcoord = float2(0.25, 0.25);
    ScenePathTracePush push = (ScenePathTracePush)0;
    push.debugFlags = disabled ? kOpenPBRDebugDisableMaterialTextures : 0u;
    uint seed = 123u;
    let closure = resolveOpenPBRInputs(hit, material, push, (RayCone)0,
        float3(0, 0, -1), seed, source, (TextureFootprint)0, hit.normal, false);
    uint base = lane * 4u;
    gProbe.output.store(base, float4(closure.inputs.base_color, closure.inputs.geometry_opacity));
    gProbe.output.store(base + 1u, float4(closure.inputs.emission_color * closure.inputs.emission_luminance, closure.inputs.base_weight));
    gProbe.output.store(base + 2u, float4(closure.inputs.specular_color, closure.inputs.specular_weight));
    gProbe.output.store(base + 3u, float4(closure.inputs.geometry_basis.n, float(seed)));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void openPBRClosureProbeMain(uint3 id : SV_DispatchThreadID)
{
    uint lane = id.x;
    uint variant = lane / 128;
    Material<OpenPBRMaterialProgram<ProbeSource>> material = {};
    material.program.source.probe = gProbe;
    material.program.fixedMetalness = -1;
    material.program.hit.tangent = float4(1, 0, 0, 1);
    SurfaceMaterialContext context = (SurfaceMaterialContext)0;
    context.shadingNormalWS = context.geometryNormalWS = float3(0, 0, 1);
    context.tangentWS = float3(1, 0, 0);
    context.bitangentWS = float3(0, 1, 0);
    context.texCoord0 = float2((lane & 1) * 0.5 + 0.25, ((lane >> 1) & 1) * 0.5 + 0.25);
    context.texCoordMask = 1;
    context.frontFace = variant == 6 ? 0 : 1;
    MaterialInstanceRef instance;
    instance.parameterOffset = (lane & 1) * 16;
    instance.resourceOffset = variant;
    let evaluated = evaluateHit(material, context, instance);
    let closure = evaluated.closure;
    float3 wo = normalize(float3(0.2, 0.1, variant == 6 ? -1 : 1));
    SurfaceSamplingContext sampling;
    sampling.throughput = float3(0.2, 0.7, 0.4);
    sampling.wavelengths = OpenPBRBaseRGBWavelengthsNm;
    sampling.exteriorIor = 1.1;
    countStage(gProbe, 3);
    let prepared = closure.prepare(wo, TransportMode.Radiance, sampling);
    let probeContext = material.program.source.context();
    let native = openPBRPrepare(probeContext, closure.inputs, sampling.throughput, sampling.wavelengths, sampling.exteriorIor, wo);
    OpenPBRClosure<StatefulOpenPBRContext> standalone;
    standalone.context.base = probeContext;
    standalone.context.scale = 0.8;
    standalone.inputs = closure.inputs;
    standalone.occlusion = closure.occlusion;
    let standalonePrepared = standalone.prepare(wo, TransportMode.Radiance, sampling);
    let standaloneNative = openPBRPrepare(standalone.context, standalone.inputs,
        sampling.throughput, sampling.wavelengths, sampling.exteriorIor, wo);
    float3 error = float3(0);
    for (uint light = 0; light < gProbe.lightCount; ++light) {
        float3 wi = normalize(float3(float(light) * 0.13 - 0.3, 0.2, (light & 1) ? -1 : 1));
        float3 expected = openPBRGetSumOfDiffuseSpecular(openPBREval(probeContext, native, wi));
        countStage(gProbe, 4);
        float3 projected = evaluateSurfaceProjected(prepared, wi);
        BSDFEval value = prepared.eval(wi);
        error = max(error, abs(projected - expected));
        error = max(error, abs(value.f * abs(dot(closure.inputs.geometry_basis.n, wi)) - expected));
        error = max(error, float3(abs(value.pdf - openPBRPdf(probeContext, native, wi))));
        error = max(error, abs(standalonePrepared.evalProjected(wi) -
            openPBRGetSumOfDiffuseSpecular(openPBREval(standalone.context, standaloneNative, wi))));
        error = max(error, float3(abs(standalonePrepared.pdf(wi) - openPBRPdf(standalone.context, standaloneNative, wi))));
        let components = standalonePrepared.evalProjectedComponents(wi);
        let nativeComponents = openPBREval(standalone.context, standaloneNative, wi);
        error = max(error, abs(components.diffuse - nativeComponents.diffuse));
        error = max(error, abs(components.specular - nativeComponents.specular));
    }
    float3 random = float3((float(lane % 128) + 0.5) / 128, frac(float(lane) * 0.61803398875), frac(float(lane) * 0.41421356237));
    BSDFWeightSample weighted = sampleSurfaceWeight(prepared, random);
    BSDFSample sample = sampleSurfaceContinuation(prepared, random);
    float3 nativeDirection;
    OpenPBRDiffuseSpecular nativeWeight;
    float nativePdf;
    uint nativeType;
    openPBRSample(probeContext, native, random, nativeDirection, nativeWeight, nativePdf, nativeType);
    float3 sampleError = float3(abs(weighted.pdf - nativePdf));
    if (nativePdf > 0) {
        sampleError = max(sampleError, abs(weighted.wiWS - nativeDirection));
        sampleError = max(sampleError, abs(weighted.weight - openPBRGetSumOfDiffuseSpecular(nativeWeight)));
        sampleError = max(sampleError, abs(sample.f * abs(dot(closure.inputs.geometry_basis.n, sample.wiWS)) / sample.pdf - weighted.weight));
        sampleError = max(sampleError, float3(abs(float(weighted.flags) - float(nativeType))));
    } else {
        sampleError = max(sampleError, abs(sample.f) + abs(weighted.weight));
    }
    float3 standaloneDirection;
    OpenPBRDiffuseSpecular standaloneWeight;
    float standalonePdf;
    uint standaloneType;
    openPBRSample(standalone.context, standaloneNative, random, standaloneDirection, standaloneWeight, standalonePdf, standaloneType);
    OpenPBRDiffuseSpecular components;
    let standaloneSample = standalonePrepared.sampleWeightedComponents(random, components);
    sampleError = max(sampleError, float3(abs(standaloneSample.pdf - standalonePdf)));
    if (standalonePdf > 0) {
        sampleError = max(sampleError, abs(components.diffuse - standaloneWeight.diffuse));
        sampleError = max(sampleError, abs(components.specular - standaloneWeight.specular));
        sampleError = max(sampleError, abs(standaloneSample.wiWS - standaloneDirection));
        sampleError = max(sampleError, abs(standaloneSample.weight - openPBRGetSumOfDiffuseSpecular(standaloneWeight)));
        sampleError = max(sampleError, float3(abs(float(standaloneSample.flags) - float(standaloneType))));
    }
    // Reusing the resolved closure for a second view must not load textures.
    let second = closure.prepare(normalize(float3(-0.4, 0.2, 1)), TransportMode.Radiance);
    let importance = closure.prepare(wo, TransportMode.Importance);
    float unsupported = importance.pdf(wo) + length(importance.eval(wo).f) + importance.sample(random).pdf;
    uint base = lane * 8;
    gProbe.output.store(base, float4(error, unsupported));
    gProbe.output.store(base + 1, float4(sampleError, weighted.pdf));
    gProbe.output.store(base + 2, float4(closure.inputs.base_color, closure.inputs.base_metalness));
    gProbe.output.store(base + 3, float4(closure.inputs.geometry_basis.n, weighted.eta));
    gProbe.output.store(base + 4, float4(abs(prepared.emission() - native.emission), float(weighted.flags)));
    gProbe.output.store(base + 5, float4(second.eval(wo).f, evaluated.flags));
    gProbe.output.store(base + 6, float4(evaluated.emission, closure.inputs.transmission_weight));
    gProbe.output.store(base + 7, float4(weighted.weight, sample.pdf));
}
