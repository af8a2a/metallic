#language slang 2026
module ScenePathTrace;
import ScenePathTracing;
[shader("compute")]
[numthreads(8, 8, 1)]
void scenePathTraceMain(uint3 id : SV_DispatchThreadID)
{
    Metallic.ScenePathTracing::dispatchScenePathTrace(id);
}
