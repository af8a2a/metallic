import ShaderCore;
import Material;
import MaterialProgram;
import FiberMaterial;
import FiberLighting;
import RTXCRFiber;
import RTXCRHair;
using Metallic;
using Metallic.Material;
using Metallic.Lighting;
#include "../../../Source/Runtime/Render/Core/StrandParameters.h"
import ParameterRoot;

float3 safeNormalize(float3 v, float3 fallback) { return dot(v,v)>1e-16 ? normalize(v) : fallback; }
#include "../../Interop/RTXCRMaterialAdapter.hlsli"

float3 strandPosition(float3 p, float u, float phase, float amplitude)
{
    return p + float3(sin(phase)*amplitude*u*u,0,0);
}
float3 strandProject(float3 p, StrandCamera c, uint2 extent)
{
    float3 d = p-c.eyeNear.xyz;
    float z = dot(d,c.forwardFar.xyz);
    float scale = c.rightTan.w>0 ? max(z,c.eyeNear.w)*c.rightTan.w : -c.rightTan.w;
    float2 ndc = float2(dot(d,c.rightTan.xyz)/c.upAspect.w,-dot(d,c.upAspect.xyz)) / scale;
    return float3((ndc*0.5+0.5)*float2(extent),z);
}
// Analytic capsule footprint; original coordinates survive near/far clipping.
bool strandRaster(StrandSegment s, StrandFrame f, float2 pixel, out StrandVisibilityRecord hit, out float2 hRange)
{
    hit = (StrandVisibilityRecord)0; hRange = 0;
    float opacity = s.shape.y*smoothstep(s.shape.z,s.shape.z+0.03125,f.density);
    if (opacity <= 0) { return false; }
    float3 a = strandPosition(s.a.xyz,s.normalU.w,f.phase,f.amplitude);
    float3 b = strandPosition(s.b.xyz,s.shape.x,f.phase,f.amplitude);
    float za = dot(a-f.camera.eyeNear.xyz,f.camera.forwardFar.xyz), zb = dot(b-f.camera.eyeNear.xyz,f.camera.forwardFar.xyz);
    float lo = 0, hi = 1, dz = zb-za;
    if (abs(dz)<1e-12) {
        if (za<f.camera.eyeNear.w || za>f.camera.forwardFar.w) { return false; }
    } else {
        float t0 = (f.camera.eyeNear.w-za)/dz, t1 = (f.camera.forwardFar.w-za)/dz;
        lo = max(0.0,min(t0,t1)); hi = min(1.0,max(t0,t1));
        if (lo>hi) { return false; }
    }
    float3 pa = strandProject(lerp(a,b,lo),f.camera,uint2(f.width,f.height));
    float3 pb = strandProject(lerp(a,b,hi),f.camera,uint2(f.width,f.height));
    float2 axis = pb.xy-pa.xy;
    float t = saturate(dot(pixel-pa.xy,axis)/max(dot(axis,axis),1e-12));
    float perspectiveT = f.camera.rightTan.w>0 ? (t/pb.z)/max((1-t)/pa.z+t/pb.z,1e-12) : t;
    hit.u = lerp(lo,hi,perspectiveT); hit.depth = lerp(za,zb,hit.u);
    float scale = f.camera.rightTan.w>0 ? hit.depth*f.camera.rightTan.w : -f.camera.rightTan.w;
    float radius = lerp(s.a.w,s.b.w,hit.u)*float(f.height)*0.5/scale;
    float2 offset = pixel-lerp(pa.xy,pb.xy,t);
    hit.coverage = min(saturate(radius+0.5-length(offset)),min(2*radius,1.0))*opacity;
    // Integrate the covered cross-section, rather than sampling h=+/-1 for
    // partially covered edge pixels. The same one-pixel box defines coverage.
    float center = -dot(offset,safeNormalize(float3(-axis.y,axis.x,0),float3(1,0,0)).xy);
    hRange = clamp((center+float2(-0.5,0.5))/max(radius,1e-8),-1.0,1.0);
    return hit.coverage>0;
}
bool strandBefore(StrandVisibilityRecord a, StrandVisibilityRecord b, StrandParameters p)
{
    if (a.depth != b.depth) { return a.depth<b.depth; }
    uint4 ia = p.segments.load(a.segment).identity, ib = p.segments.load(b.segment).identity;
    return ia.x != ib.x ? ia.x<ib.x : ia.y<ib.y;
}
void strandInsert(StrandVisibilityRecord hit, StrandParameters p, uint capacity,
    inout StrandVisibilityRecord layers[8], inout uint count, inout uint total)
{
    ++total;
    if (count<capacity) { layers[count++] = hit; }
    else if (strandBefore(hit,layers[count-1],p)) { layers[count-1] = hit; }
    for (uint j = count-1; j>0 && strandBefore(layers[j],layers[j-1],p); --j) {
        StrandVisibilityRecord tmp = layers[j]; layers[j] = layers[j-1]; layers[j-1] = tmp;
    }
}
[shader("compute")]
[numthreads(8,8,1)]
void strandVisibilityMain(uint3 tid : SV_DispatchThreadID)
{
    StrandParameters p = getParameters<StrandParameters>(); StrandFrame f = p.frame.load(0);
    if (tid.x>=f.width || tid.y>=f.height) { return; }
    StrandVisibilityRecord layers[8]; uint count = 0, total = 0;
    for (uint i=0;i<8;++i) { layers[i]=(StrandVisibilityRecord)0; }
    float opaque = f.camera.forwardFar.w;
    if (p.hasOpaqueDepth != 0) { opaque = resolveUniform(p.opaqueDepth).Load(int3(tid.xy,0)).x; }
    StrandVisibilityRecord pending = (StrandVisibilityRecord)0;
    bool hasPending = false;
    uint2 lastIdentity = uint2(0xffffffffu);
    float lastDepth = 0;
    for (uint i = 0; i<f.segmentCount; ++i) {
        StrandSegment s = p.segments.load(i); StrandVisibilityRecord hit; float2 hRange;
        if (!strandRaster(s,f,float2(tid.xy)+0.5,hit,hRange) || hit.depth>=opaque) { continue; }
        hit.segment = i;
        // Assets are sorted by strand ID and segment index. Form joint unions
        // BEFORE bounded insertion, so candidate/overflow counts do not depend on K.
        bool joined = hasPending && s.identity.x == lastIdentity.x && s.identity.y == lastIdentity.y+1 &&
            abs(hit.depth-lastDepth)<=2.0*max(s.a.w,s.b.w);
        if (joined) {
            if (hit.coverage>pending.coverage) { pending = hit; }
        } else {
            if (hasPending) { strandInsert(pending,p,f.capacity,layers,count,total); }
            pending = hit; hasPending = true;
        }
        lastIdentity = s.identity.xy; lastDepth = hit.depth;
    }
    if (hasPending) { strandInsert(pending,p,f.capacity,layers,count,total); }
    uint offset = (tid.y*f.width+tid.x)*f.capacity;
    for (uint i = 0; i<f.capacity; ++i) { p.records.store(offset+i,i<count ? layers[i] : (StrandVisibilityRecord)0); }
    resolveUniform(p.counts)[tid.xy] = uint4(count,total>count ? total-count : 0,total,f.historyValid);
}
float strandShadow(StrandParameters p, StrandFrame f, float3 origin, float3 direction, uint self)
{
    if (p.shadows == 0) { return 1; }
    float transmission = 1; uint last = 0xffffffffu;
    for (uint i = 0; i<f.segmentCount; ++i) {
        StrandSegment s = p.segments.load(i);
        if (s.identity.x == self || s.identity.x == last) { continue; }
        float3 a = strandPosition(s.a.xyz,s.normalU.w,f.phase,f.amplitude), b = strandPosition(s.b.xyz,s.shape.x,f.phase,f.amplitude);
        float3 axis = b-a, w = origin-a;
        float ab = dot(direction,axis), bb = dot(axis,axis), aw = dot(direction,w), bw = dot(axis,w);
        float u = saturate((bw-ab*aw)/max(bb-ab*ab,1e-12));
        float t = dot(a+u*axis-origin,direction);
        if (t<=1e-5) { continue; }
        if (length(origin+t*direction-(a+u*axis))<lerp(s.a.w,s.b.w,u)) {
            transmission *= 1-s.shape.y*smoothstep(s.shape.z,s.shape.z+0.03125,f.density); last = s.identity.x;
        }
    }
    return transmission;
}
[shader("compute")]
[numthreads(8,8,1)]
void strandLightingMain(uint3 tid : SV_DispatchThreadID)
{
    StrandParameters p = getParameters<StrandParameters>(); StrandFrame f = p.frame.load(0);
    if (tid.x>=f.width || tid.y>=f.height) { return; }
    uint4 counts = resolveUniform(p.counts)[tid.xy];
    float3 result = 0; float transmission = 1;
    float4 motion = 0;
    uint4 identity = uint4(0xffffffffu,0xffffffffu,0,0);
    for (uint layer = 0; layer<counts.x; ++layer) {
        StrandVisibilityRecord r = p.records.load((tid.y*f.width+tid.x)*f.capacity+layer);
        StrandSegment s = p.segments.load(r.segment); float u = lerp(s.normalU.w,s.shape.x,r.u);
        float3 a = strandPosition(s.a.xyz,s.normalU.w,f.phase,f.amplitude), b = strandPosition(s.b.xyz,s.shape.x,f.phase,f.amplitude);
        FiberMaterialContext context = (FiberMaterialContext)0;
        context.positionWS = lerp(a,b,r.u); context.tangentWS = safeNormalize(b-a,float3(0,1,0));
        context.normalWS = safeNormalize(s.normalU.xyz-context.tangentWS*dot(s.normalU.xyz,context.tangentWS),
            safeNormalize(cross(context.tangentWS,float3(1,0,0)),float3(0,0,1)));
        context.bitangentWS = cross(context.normalWS,context.tangentWS);
        context.radius = lerp(s.a.w,s.b.w,r.u); context.strandParameter = u;
        StrandVisibilityRecord unused; float2 hRange; strandRaster(s,f,float2(tid.xy)+0.5,unused,hRange);
        context.flags = kFiberHasRadius | kFiberHasStrandParameter | kFiberHasMaterialCoordinates | kFiberHasCrossSectionOffset;
        float3 wo = safeNormalize(f.camera.eyeNear.xyz-context.positionWS,context.normalWS);
        float3 light = safeNormalize(p.light.xyz,float3(0,0,1));
        float shadow = strandShadow(p,f,context.positionWS,light,s.identity.x);
        float3 radiance = 0;
        // Deterministic four-point midpoint quadrature over the visible width.
        for (uint sample = 0; sample<4; ++sample) {
            float h = lerp(hRange.x,hRange.y,(float(sample)+0.5)*0.25);
            context.crossSectionOffset = h; context.materialCoordinates = float2(u,h*0.5+0.5);
            let material = evaluateSceneFiberMaterial(p.materials.load(s.identity.z),context);
            let prepared = material.closure.prepare(wo,TransportMode.Radiance);
            radiance += 0.25*(material.emission + evaluateFiberDirect(prepared,light,float3(p.light.w))*shadow +
                evaluateFiberDirect(prepared,normalize(float3(-1,1,1)),float3(0.3)));
        }
        result += transmission*r.coverage*radiance; transmission *= 1-r.coverage;
        if (layer == 0) {
            float3 previous = lerp(
                strandPosition(s.previousA.xyz,s.normalU.w,f.previousPhase,f.previousAmplitude),
                strandPosition(s.previousB.xyz,s.shape.x,f.previousPhase,f.previousAmplitude),r.u);
            if (f.historyValid == 0) { previous = context.positionWS; }
            float3 oldPixel = strandProject(previous,f.previousCamera,uint2(f.width,f.height));
            float3 pixel = strandProject(context.positionWS,f.camera,uint2(f.width,f.height));
            motion = float4(oldPixel.xy-pixel.xy,u,r.depth);
            identity = uint4(s.identity.xy,asuint(u),asuint(r.coverage));
        }
    }
    float3 background = p.hasOpaqueColor != 0 ? resolveUniform(p.opaqueColor).Load(int3(tid.xy,0)).rgb : p.background.rgb;
    result += transmission*background;
    if (counts.y != 0 && p.overflowPolicy == 0) { result = float3(1,0,1); }
    resolveUniform(p.color)[tid.xy] = float4(max(result,0),1-transmission);
    resolveUniform(p.motion)[tid.xy] = motion;
    resolveUniform(p.identity)[tid.xy] = identity;
}
