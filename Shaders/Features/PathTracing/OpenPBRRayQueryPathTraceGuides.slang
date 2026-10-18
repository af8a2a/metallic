#language slang 2026
module OpenPBRRayQueryPathTraceGuides;
import OpenPBRPathTracing;
#if METALLIC_TEXTURE_FEEDBACK_WAVE
[MaximallyReconverges]
#endif
[shader("compute")]
[numthreads(8, 8, 1)]
void openPbrRayQueryPathTraceGuidesMain(uint3 id : SV_DispatchThreadID)
{
    Metallic.OpenPBRPathTracing::dispatchOpenPbrRayQueryPathTraceGuides(id);
}
