#language slang 2026
module OpenPBRRayQueryPathTrace;
import OpenPBRPathTracing;
[shader("compute")]
[numthreads(8, 8, 1)]
void openPbrRayQueryPathTraceMain(uint3 id : SV_DispatchThreadID)
{
    Metallic.OpenPBRPathTracing::dispatchOpenPbrRayQueryPathTrace(id);
}
