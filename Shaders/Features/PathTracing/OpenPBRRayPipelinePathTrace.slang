#language slang 2026
module OpenPBRRayPipelinePathTrace;
import OpenPBRPathTracing;

#if !METALLIC_PT_RAY_PIPELINE
#error OpenPBR ray-pipeline entry requires the global METALLIC_PT_RAY_PIPELINE request policy.
#endif

[shader("raygeneration")]
void openPbrRayPipelinePathTraceMain()
{
    Metallic.OpenPBRPathTracing::dispatchOpenPbrRayQueryPathTrace(DispatchRaysIndex());
}
