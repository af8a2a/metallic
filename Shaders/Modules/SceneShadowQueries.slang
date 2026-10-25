#language slang 2026
module SceneShadowQueries;
import Core;
import NRDEncoding;
using Metallic.Interop.NRDEncoding;
import Material;
import MaterialProgram;
import OpenPBRMaterialParameters;
import RTXCRMaterialParameters;
import Lighting;
import ScenePathTypes;
import ScenePathMath;
import SceneSurface;
import Atmosphere;
using Metallic;
using Metallic.Material;
using Metallic.Lighting;
using Metallic.SceneSurface;
using Metallic.ScenePath;
import ScenePathTracing;
using Metallic.ScenePathTracing;
namespace Metallic.SceneShadowQueries
{
#if METALLIC_SHADOW_RAY_CAPTURE
#include "../Interop/ShadowRayCapture.h"
#endif
public struct ShadowGeometryPush
{
    public uint materialTextureCount;
#if METALLIC_SHADOW_RAY_CAPTURE
    public uint padding;
    public uint64_t outputAddress;
#endif
};

public float traceGeometryShadow(float3 origin, float3 direction, float maxDistance
#if METALLIC_SHADOW_RAY_CAPTURE
    , inout ShadowRayCaptureRecord capture
#endif
)
{
    ShadowGeometryPush geometry = getConstants<ShadowGeometryPush>();
    ScenePathTracePush push = (ScenePathTracePush)0;
    push.materialTextureCount = geometry.materialTextureCount;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.001;
    ray.TMax = max(maxDistance, ray.TMin);
#if METALLIC_SHADOW_RAY_CAPTURE
    capture.originX = ray.Origin.x; capture.originY = ray.Origin.y; capture.originZ = ray.Origin.z;
    capture.tMin = ray.TMin;
    capture.directionX = ray.Direction.x; capture.directionY = ray.Direction.y; capture.directionZ = ray.Direction.z;
    capture.tMax = ray.TMax;
    capture.active = 1u;
#endif
    // SIGMA needs the closest blocker's distance, not the first traversal hit.
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(sceneAccelerationStructure(), RAY_FLAG_NONE, 0xff, ray);
    Ray surfaceRay; surfaceRay.origin = origin; surfaceRay.direction = direction;
    while (query.Proceed()) {
#if METALLIC_SHADOW_RAY_CAPTURE
        ++capture.proceedCalls;
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) { ++capture.nonOpaqueCandidates; }
#endif
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
            streamRayAlphaAccepts(query.CandidateInstanceID(), uint(query.CandidateClusterID()),
                query.CandidatePrimitiveIndex(), query.CandidateTriangleBarycentrics(), surfaceRay,
                query.CandidateTriangleRayT(), push)) { query.CommitNonOpaqueTriangleHit();
#if METALLIC_SHADOW_RAY_CAPTURE
            ++capture.acceptedCandidates;
#endif
        }
    }
#if METALLIC_SHADOW_RAY_CAPTURE
    ++capture.proceedCalls; // Includes the final false return, even for fully opaque traversal.
    capture.hit = query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 1u : 0u;
    if (capture.hit != 0u) {
        capture.rawCommittedT = query.CommittedRayT();
        capture.instanceId = query.CommittedInstanceID();
        capture.clusterId = uint(query.CommittedClusterID());
        capture.primitiveId = query.CommittedPrimitiveIndex();
    }
#endif
    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT
        ? min(query.CommittedRayT(), kNRDFP16Max - 1.0) : kNRDFP16Max;
}

}
