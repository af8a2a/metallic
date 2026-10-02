import Core;
using Metallic;

[shader("compute")]
[numthreads(1, 1, 1)]
void unsafeNativeAsMain()
{
    RayDesc ray = {float3(0, 0, 1), 0.001, float3(0, 0, -1), 100.0};
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    // Deliberately bypass resolveDescriptor: compilation must reject this ABI.
    query.TraceRayInline(*DescriptorHandle<RaytracingAccelerationStructure>(getComputeResources().resources[0].handle), 0, 255, ray);
    while (query.Proceed()) {}
    getResource<RWByteAddressBuffer>(1).Store(0, query.CommittedStatus());
}
