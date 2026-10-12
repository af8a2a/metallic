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
public struct ShadowGeometryPush
{
    public uint materialTextureCount;
};

public float traceGeometryShadow(float3 origin, float3 direction, float maxDistance)
{
    ShadowGeometryPush geometry = getConstants<ShadowGeometryPush>();
    ScenePathTracePush push = (ScenePathTracePush)0;
    push.materialTextureCount = geometry.materialTextureCount;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = direction;
    ray.TMin = 0.001;
    ray.TMax = max(maxDistance, ray.TMin);
    // SIGMA needs the closest blocker's distance, not the first traversal hit.
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> query;
    query.TraceRayInline(sceneAccelerationStructure(), RAY_FLAG_FORCE_NON_OPAQUE, 0xff, ray);
    Ray surfaceRay; surfaceRay.origin = origin; surfaceRay.direction = direction;
    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE &&
            streamRayAlphaAccepts(query.CandidateInstanceID(), uint(query.CandidateClusterID()),
                query.CandidatePrimitiveIndex(), query.CandidateTriangleBarycentrics(), surfaceRay,
                query.CandidateTriangleRayT(), push)) { query.CommitNonOpaqueTriangleHit(); }
    }
    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT
        ? min(query.CommittedRayT(), kNRDFP16Max - 1.0) : kNRDFP16Max;
}

}
