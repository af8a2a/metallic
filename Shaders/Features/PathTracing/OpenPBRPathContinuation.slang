#language slang 2026
module OpenPBRPathContinuation;

import OpenPBRPathTracing;

#if !METALLIC_PT_SPLIT_PRIMARY || !METALLIC_PT_SINGLE_SAMPLE || SHARC_UPDATE
#error Primary continuation requires the split single-sample guide request.
#endif

[shader("compute")]
[numthreads(8, 8, 1)]
void openPbrPathContinuationMain(uint3 id : SV_DispatchThreadID)
{
    Metallic.OpenPBRPathTracing::dispatchOpenPbrRayQueryPathTraceGuides(id);
}
