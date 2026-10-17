#language slang 2026
module VkGltfRayPipelinePathTrace;
import VkGltfPathTracing;

[shader("raygeneration")]
void vkGltfRayPipelinePathTraceMain()
{
    Metallic.ExperimentalPath::dispatchVkGltf(DispatchRaysIndex().xy);
}
