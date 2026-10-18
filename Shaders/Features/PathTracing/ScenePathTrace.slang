#language slang 2026
module ScenePathTrace;
import ScenePathTracing;
#if METALLIC_TEXTURE_FEEDBACK_WAVE
[MaximallyReconverges]
#endif
[shader("compute")]
[numthreads(8, 8, 1)]
void scenePathTraceMain(uint3 id : SV_DispatchThreadID)
{
    Metallic.ScenePathTracing::dispatchScenePathTrace(id);
}
