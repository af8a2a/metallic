module LumenGlobalTracing;

import ShaderCore;
import ParameterRoot;
import LumenGlobalRay;

using Metallic;
using Metallic.Lumen;

typealias TraceRadianceTexture = _Texture<float4, __Shape2D, 0, 0, 0, 1, 0, 0, 2>;

struct LumenGlobalTraceParams
{
    LumenGlobalTraceBindings scene;
    ResourceHandle<StructuredBuffer<LumenGlobalTraceRay>> rays;
    ResourceHandle<RWStructuredBuffer<LumenGlobalTraceResult>> results;
    ResourceHandle<TraceRadianceTexture> radiance;
    uint rayCount;
    uint clipmapCount;
};

[shader("compute")]
[numthreads(64, 1, 1)]
void lumenGlobalTraceMain(uint3 thread : SV_DispatchThreadID)
{
    LumenGlobalTraceParams params = getParameters<LumenGlobalTraceParams>();
    if (thread.x >= params.rayCount) {
        return;
    }
    StructuredBuffer<LumenGlobalTraceRay> rays = resolveUniform(params.rays);
    LumenGlobalTraceScene scene = resolveLumenGlobalTraceScene(params.scene, params.clipmapCount);
    LumenGlobalTraceResult result = traceLumenGlobalRay(scene, rays[thread.x]);
    RWStructuredBuffer<LumenGlobalTraceResult> results = resolveUniform(params.results);
    results[thread.x] = result;
    TraceRadianceTexture radiance = resolveUniform(params.radiance);
    radiance[thread.xy] = float4(result.radiance.xyz, 1.0f);
}
