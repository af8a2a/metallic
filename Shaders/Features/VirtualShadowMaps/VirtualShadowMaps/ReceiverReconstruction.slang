#language slang 2026
implementing VirtualShadowMaps;

// Implementation units preprocess independently; retain this unit's permutation defaults.
#ifndef VSM_REFINED_RECEIVER
#define VSM_REFINED_RECEIVER 0
#endif

struct VSMReceiverBasis
{
    float3 eye;
    float3 forward;
    float3 right;
    float3 up;
    float3 rightAspect;
    float tanHalfFov;
    float orthoHalfHeight;
    float2 jitter;
    float2 invViewport;
    float2 ndcScale;
    int2 viewportSize;
    float nearPlane;
    float farPlane;
    float farMinusNear;
    float nearTimesFar;
    float depthParamA;
    float depthParamB;
    bool isOrtho;
    bool reversed;
};

VSMReceiverBasis vsmMakeReceiverBasis(ViewConstants view)
{
    VSMReceiverBasis basis;
    basis.eye = view.current.eye.xyz;
    basis.forward = normalize(view.current.center.xyz - view.current.eye.xyz);
    basis.right = normalize(cross(basis.forward, view.current.upProjection.xyz));
    basis.up = cross(basis.right, basis.forward);
    basis.rightAspect = basis.right * view.current.viewport.x;
    basis.tanHalfFov = tan(view.current.viewport.w * 0.5);
    basis.orthoHalfHeight = view.current.clipOrtho.z * 0.5;
    basis.jitter = view.frame.z != 0u ? view.jitter.xy : float2(0.0);
    basis.invViewport = 1.0 / view.current.viewport.yz;
    basis.ndcScale = float2(2.0, -2.0) * basis.invViewport;
    basis.viewportSize = int2(view.current.viewport.yz);
    basis.nearPlane = view.current.clipOrtho.x;
    basis.farPlane = view.current.clipOrtho.y;
    basis.farMinusNear = view.current.clipOrtho.y - view.current.clipOrtho.x;
    basis.nearTimesFar = view.current.clipOrtho.x * view.current.clipOrtho.y;
    basis.isOrtho = view.current.upProjection.w > 0.5;
    basis.reversed = view.current.clipOrtho.w > 0.5;
    if (basis.isOrtho) {
        basis.depthParamA = basis.reversed ? basis.farPlane : basis.nearPlane;
        basis.depthParamB = basis.reversed ? (basis.nearPlane - basis.farPlane) : (basis.farPlane - basis.nearPlane);
    } else {
        basis.depthParamA = basis.reversed ? basis.nearPlane : basis.farPlane;
        basis.depthParamB = basis.reversed ? basis.farMinusNear : -basis.farMinusNear;
    }
    return basis;
}

float vsmLinearDepth(float value, VSMReceiverBasis basis)
{
    if (basis.isOrtho) {
        return fma(value, basis.depthParamB, basis.depthParamA);
    } else {
        return basis.nearTimesFar / fma(value, basis.depthParamB, basis.depthParamA);
    }
}

float3 vsmReceiverPosition(int2 pixel, float linearDepth, VSMReceiverBasis basis)
{
    float2 ndc = fma(float2(pixel) + 0.5 + basis.jitter, basis.ndcScale, float2(-1.0, 1.0));
    float halfHeight = basis.isOrtho ? basis.orthoHalfHeight
        : linearDepth * basis.tanHalfFov;
    return basis.eye + basis.forward * linearDepth +
        halfHeight * (basis.rightAspect * ndc.x + basis.up * ndc.y);
}

// Uses the actual receiver projection, including temporal jitter in pixel units.
// clipOrtho.w selects reversed (1) or standard (0) receiver depth only; shadow
// projections always use reversed depth.
bool vsmReceiver(int2 pixel, VSMReceiverBasis basis, Texture2D<float> depth,
    out float3 position, out float linearDepth)
{
    position = float3(0.0);
    linearDepth = 0.0;
    if (any(pixel < 0) || any(pixel >= basis.viewportSize)) { return false; }
    float value = depth.Load(int3(pixel, 0));
    if (value == (basis.reversed ? 0.0 : 1.0)) { return false; }
    linearDepth = vsmLinearDepth(value, basis);
    position = vsmReceiverPosition(pixel, linearDepth, basis);
    return true;
}

bool vsmReceiver(int2 pixel, ViewConstants view, Texture2D<float> depth,
    out float3 position, out float linearDepth)
{
    return vsmReceiver(pixel, vsmMakeReceiverBasis(view), depth, position, linearDepth);
}

float3 vsmReceiverNormal(int2 pixel, VSMReceiverBasis basis, Texture2D<float> depth,
    float3 position, float linearDepth)
{
    float emptyDepth = basis.reversed ? 0.0 : 1.0;
    float zl = 0.0, zr = 0.0, zt = 0.0, zb = 0.0;
    bool hasLeft = false, hasRight = false, hasTop = false, hasBottom = false;

    if (pixel.x > 0) {
        float val = depth.Load(int3(pixel.x - 1, pixel.y, 0));
        if (val != emptyDepth) {
            hasLeft = true;
            zl = vsmLinearDepth(val, basis);
        }
    }
    if (pixel.x + 1 < basis.viewportSize.x) {
        float val = depth.Load(int3(pixel.x + 1, pixel.y, 0));
        if (val != emptyDepth) {
            hasRight = true;
            zr = vsmLinearDepth(val, basis);
        }
    }
    if (pixel.y > 0) {
        float val = depth.Load(int3(pixel.x, pixel.y - 1, 0));
        if (val != emptyDepth) {
            hasTop = true;
            zt = vsmLinearDepth(val, basis);
        }
    }
    if (pixel.y + 1 < basis.viewportSize.y) {
        float val = depth.Load(int3(pixel.x, pixel.y + 1, 0));
        if (val != emptyDepth) {
            hasBottom = true;
            zb = vsmLinearDepth(val, basis);
        }
    }

    bool useLeft = hasLeft && (!hasRight || abs(zl - linearDepth) < abs(zr - linearDepth));
    bool useTop = hasTop && (!hasBottom || abs(zt - linearDepth) < abs(zb - linearDepth));

    float3 dx = float3(0.0);
    if (useLeft) {
        dx = position - vsmReceiverPosition(pixel - int2(1, 0), zl, basis);
    } else if (hasRight) {
        dx = vsmReceiverPosition(pixel + int2(1, 0), zr, basis) - position;
    }

    float3 dy = float3(0.0);
    if (useTop) {
        dy = position - vsmReceiverPosition(pixel - int2(0, 1), zt, basis);
    } else if (hasBottom) {
        dy = vsmReceiverPosition(pixel + int2(0, 1), zb, basis) - position;
    }

    float3 normal = cross(dy, dx);
    float lengthSquared = dot(normal, normal);
    return lengthSquared > 1e-20 ? normal * rsqrt(lengthSquared) : float3(0.0);
}

float3 vsmReceiverNormal(int2 pixel, ViewConstants view, Texture2D<float> depth,
    float3 position, float linearDepth)
{
    return vsmReceiverNormal(pixel, vsmMakeReceiverBasis(view), depth, position, linearDepth);
}

// Screen depth limits the uncertain segment near a simplified shadow receiver.
// As in UE's screen-ray bias, this changes only the comparison depth, never the
// receiver's page/clipmap choice. Point samples are a bounded heuristic, not a
// visibility oracle for hidden or subpixel geometry.
float vsmLODScreenRayOffset(float3 position, float3 normal, float linearDepth, float3 toLight,
    ViewConstants view, Texture2D<float> depth, float screenLength, float blockerDistance, float bias)
{
    if (screenLength == 0.0) { return 0.0; }
    bool orthographic = view.current.upProjection.w > 0.5;
    float tangent = tan(view.current.viewport.w * 0.5);
    float viewHeight = orthographic ? view.current.clipOrtho.z : 2.0 * linearDepth * tangent;
    float maximumLength = screenLength * viewHeight;
    if (blockerDistance - bias > maximumLength) { return 0.0; }
    float normalLight = dot(normal, toLight);
    if (normalLight <= 1e-5) { return 0.0; }
    // Concentrate samples around the depth conflict. A fixed camera-scaled
    // stride can jump completely over a nearby thin blocker before its first
    // sample, even when that blocker is present in the primary depth buffer.
    float rayLength = min(maximumLength, max(0.0, blockerDistance) + bias);
    float3 forward = normalize(view.current.center.xyz - view.current.eye.xyz);
    float3 right = normalize(cross(forward, view.current.upProjection.xyz));
    float3 up = cross(right, forward);
    float2 jitter = view.frame.z != 0u ? view.jitter.xy : float2(0.0);
    for (uint step = 1u; step <= 4u; ++step) {
        float time = float(step) * 0.25;
        float3 delta = position + toLight * (time * rayLength) - view.current.eye.xyz;
        float rayDepth = dot(delta, forward);
        float safeLength = rayLength * max(0.0, time - 0.375);
        if (rayDepth <= view.current.clipOrtho.x || rayDepth >= view.current.clipOrtho.y) { return 0.0; }
        float halfHeight = orthographic ? viewHeight * 0.5 : rayDepth * tangent;
        float2 ndc = float2(dot(delta, right) / view.current.viewport.x, dot(delta, up)) / halfHeight;
        float2 screen = float2(ndc.x + 1.0, 1.0 - ndc.y) * 0.5 * view.current.viewport.yz - jitter;
        if (any(screen < 0.0) || any(screen >= view.current.viewport.yz)) { return 0.0; }
        float3 samplePosition;
        float sampleDepth;
        if (vsmReceiver(int2(screen), view, depth, samplePosition, sampleDepth)) {
            // A thin surface may already be behind the sampled ray point. Its
            // height above the receiver plane still indicates a possible near
            // blocker; keep the original shadow instead of stepping over it.
            float planeDistance = dot(normal, samplePosition - position) / normalLight;
            if (planeDistance > bias && planeDistance <= rayLength + bias) { return 0.0; }
            if (rayDepth > sampleDepth) { return safeLength; }
        }
    }
    return rayLength;
}

#if VSM_REFINED_RECEIVER
float3 vsmPrimaryCameraNormal(VirtualShadowMapReceiverTriangle triangle, ViewConstants view)
{
    float3 toCamera = view.current.upProjection.w > 0.5 ? view.current.eye.xyz - view.current.center.xyz :
        view.current.eye.xyz - triangle.a;
    return dot(triangle.normal, toCamera) >= 0.0 ? triangle.normal : -triangle.normal;
}

void vsmLimitReceiverRay(float planeDistance, float planeRate, inout float distance)
{
    if (planeRate < 0.0) { distance = min(distance, planeDistance / -planeRate); }
}

bool vsmReceiverCandidateBlocks(uint2 pixel, VirtualShadowMapReceiverContext context,
    float3 origin, float3 direction, bool nearReceiver, inout uint4 recent0, inout uint4 recent1,
    inout uint previousVisibility)
{
    if (nearReceiver) {
        return vsmPrimaryClusterBlocksRay(pixel, context, origin, direction, recent0, recent1);
    }
    return vsmPrimaryTriangleBlocksRayAtPixel(pixel, context, origin, direction, recent0, recent1, previousVisibility);
}

bool vsmAllowLODReceiverCorrection(uint2 pixel, float3 position, float linearDepth, float3 toLight,
    ViewConstants view, uint receiverGeometry, float screenLength)
{
    // Missing primary geometry is an optional representation, not permission
    // to infer that a depth-only miss is visible. Keep the original shadow.
    if (receiverGeometry == 0xffffffffu) { return false; }
    VirtualShadowMapReceiverContext context = vsmResolveReceiverContext(receiverGeometry);
    VirtualShadowMapReceiverTriangle receiver;
    if (!vsmPrimaryReceiverTriangle(pixel, context, receiver)) { return false; }
    if (dot(vsmPrimaryCameraNormal(receiver, view), toLight) <= 0.0) { return false; }

    // Missing/discontinuous primary ownership cannot establish that the shadow
    // is a receiver error. Adjacent facing alone does not establish occlusion;
    // the candidate clusters below test their actual triangles instead.
    static const int2 neighbors[4] = {int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1)};
    for (uint i = 0u; i < 4u; ++i) {
        int2 neighbor = int2(pixel) + neighbors[i];
        if (any(neighbor < 0) || any(neighbor >= int2(view.current.viewport.yz))) { return false; }
        VirtualShadowMapReceiverTriangle adjacent;
        if (!vsmPrimaryReceiverTriangle(uint2(neighbor), context, adjacent) ||
            adjacent.instanceIndex != receiver.instanceIndex) { return false; }
    }

    bool orthographic = view.current.upProjection.w > 0.5;
    float3 forward = normalize(view.current.center.xyz - view.current.eye.xyz);
    float3 cameraRay = orthographic ? forward : normalize(position - view.current.eye.xyz);
    float denominator = dot(receiver.normal, cameraRay);
    if (abs(denominator) <= 1e-5) { return false; }
    // Recover the primary triangle's surface rather than tracing from rounded
    // D32 reconstruction, which can lie just inside its own fine geometry.
    float3 origin = position + cameraRay * (dot(receiver.normal, receiver.a - position) / denominator);
    float3 right = normalize(cross(forward, view.current.upProjection.xyz));
    float3 up = cross(right, forward);
    float3 cameraOrigin = origin - view.current.eye.xyz;
    float axial = dot(cameraOrigin, forward), axialRate = dot(toLight, forward);
    float2 transverse = float2(dot(cameraOrigin, right), dot(cameraOrigin, up));
    float2 transverseRate = float2(dot(toLight, right), dot(toLight, up));
    float tangent = tan(view.current.viewport.w * 0.5);
    float2 scale = float2(view.current.viewport.x, 1.0) *
        (orthographic ? view.current.clipOrtho.z * 0.5 : tangent);
    float2 extent = scale * (orthographic ? 1.0 : axial);
    float2 extentRate = scale * (orthographic ? 0.0 : axialRate);
    float2 jitter = view.frame.z != 0u ? view.jitter.xy : float2(0.0);
    float2 lower = float2(-1.0, -1.0) + float2(2.0, -2.0) * jitter / view.current.viewport.yz;
    float2 upper = lower + 2.0;
    float2 lowerDistance = transverse - lower * extent;
    float2 upperDistance = upper * extent - transverse;
    if (axial <= view.current.clipOrtho.x || axial >= view.current.clipOrtho.y ||
        any(lowerDistance < 0.0) || any(upperDistance < 0.0)) { return false; }
    // Candidate collection ends at the camera frustum, not at the bias length.
    // A distant fine blocker can produce a short conflict in the LOD atlas.
    float rayLength = 1e30;
    vsmLimitReceiverRay(axial - view.current.clipOrtho.x, axialRate, rayLength);
    vsmLimitReceiverRay(view.current.clipOrtho.y - axial, -axialRate, rayLength);
    [unroll] for (uint axis = 0u; axis < 2u; ++axis) {
        vsmLimitReceiverRay(lowerDistance[axis], transverseRate[axis] - lower[axis] * extentRate[axis], rayLength);
        vsmLimitReceiverRay(upperDistance[axis], upper[axis] * extentRate[axis] - transverseRate[axis], rayLength);
    }
    float endDepth = axial + axialRate * rayLength;
    if (endDepth <= 0.0) { return false; }
    float2 ndc = (transverse + transverseRate * rayLength) / (scale * (orthographic ? 1.0 : endDepth));
    float2 endScreen = float2(ndc.x + 1.0, 1.0 - ndc.y) * 0.5 * view.current.viewport.yz - jitter;
    endScreen = clamp(endScreen, float2(0.0), view.current.viewport.yz);

    // Near the receiver, include cluster microtriangles that never covered a
    // pixel center. Beyond the original offset interval, query only the visible
    // primary triangles. All candidate intersections test the whole forward ray.
    // This finite screen representation does not replace hidden/offscreen scene
    // geometry, change the shadow caster cut, or require an additional AS.
    uint4 recent0 = uint4(0xffffffffu), recent1 = uint4(0xffffffffu);
    for (uint i = 0u; i < 4u; ++i) {
        if (vsmPrimaryClusterBlocksRay(uint2(int2(pixel) + neighbors[i]), context,
            origin, toLight, recent0, recent1)) { return false; }
    }
    float2 startScreen = float2(pixel) + 0.5;
    float2 deltaScreen = endScreen - startScreen;
    float viewHeight = orthographic ? view.current.clipOrtho.z : 2.0 * linearDepth * tangent;
    float nearDistance = min(screenLength * viewHeight, rayLength);
    float nearDepth = axial + axialRate * nearDistance;
    float2 nearNDC = (transverse + transverseRate * nearDistance) / (scale * (orthographic ? 1.0 : nearDepth));
    float2 nearScreen = float2(nearNDC.x + 1.0, 1.0 - nearNDC.y) * 0.5 * view.current.viewport.yz - jitter;
    float screenDistanceSquared = dot(deltaScreen, deltaScreen);
    float nearFraction = screenDistanceSquared > 1e-20 ?
        saturate(dot(nearScreen - startScreen, deltaScreen) / screenDistanceSquared) : 1.0;
    int2 cell = int2(pixel), lastCell = min(int2(endScreen), int2(view.current.viewport.yz) - 1);
    int2 direction = int2(deltaScreen.x >= 0.0 ? 1 : -1, deltaScreen.y >= 0.0 ? 1 : -1);
    float2 stepTime = 1.0 / max(abs(deltaScreen), float2(1e-30));
    float2 nextTime = stepTime * 0.5;
    float cellEntryTime = 0.0;
    uint previousVisibility = 0u;
    // A clipped 2D grid ray crosses at most width + height cell boundaries.
    uint maximumVisits = uint(view.current.viewport.y + view.current.viewport.z) + 1u;
    for (uint visited = 0u; visited < maximumVisits; ++visited) {
        // Accumulated step times can round a boundary tie toward an exterior
        // cell before lastCell. Do not read outside the primary image.
        if (any(cell < 0) || any(cell >= int2(view.current.viewport.yz))) { return false; }
        if (vsmReceiverCandidateBlocks(uint2(cell), context, origin, toLight, cellEntryTime <= nearFraction,
            recent0, recent1, previousVisibility)) { return false; }
        if (all(cell == lastCell)) { return true; }
        if (nextTime.x == nextTime.y) {
            // At a corner, the following iteration visits the Y-side cell.
            // Include the X-side cell as well before reaching the diagonal.
            int2 corner = cell + int2(direction.x, 0);
            if (all(corner >= 0) && all(corner < int2(view.current.viewport.yz)) &&
                vsmReceiverCandidateBlocks(uint2(corner), context, origin, toLight, nextTime.x <= nearFraction,
                    recent0, recent1, previousVisibility)) { return false; }
        }
        if (nextTime.x < nextTime.y) { cellEntryTime = nextTime.x; cell.x += direction.x; nextTime.x += stepTime.x; }
        else { cellEntryTime = nextTime.y; cell.y += direction.y; nextTime.y += stepTime.y; }
    }
    // Keep the original shadow if numerical traversal did not reach its exit.
    return false;
}
#endif

float vsmPerspectiveReceiverDepth(float3 position, float3 normal, uint2 virtualTexel,
    ShadowProjection projection, uint pageSize)
{
    float2 sampleNDC = (float2(virtualTexel) + 0.5) *
        (2.0 / float(projection.page.y * pageSize)) - 1.0;
    float3 forward = normalize(projection.rowW.xyz);
    float3 ray = forward + projection.metric.x *
        (normalize(projection.rowX.xyz) * sampleNDC.x + normalize(projection.rowY.xyz) * sampleNDC.y);
    float3 samplePosition = position;
    float denominator = dot(normal, ray);
    if (abs(denominator) > 1e-5 * length(ray)) {
        float distanceAlongRay = dot(normal, position - projection.originNear.xyz) / denominator;
        float3 intersection = projection.originNear.xyz + ray * distanceAlongRay;
        if (distanceAlongRay > 0.0 && all(isfinite(intersection))) { samplePosition = intersection; }
    }
    float axial = dot(projection.rowW, float4(samplePosition, 1.0));
    // A plane extrapolated to/before near cannot be meaningfully compared to
    // clipped caster depth. It is nearer than every stored sample.
    if (axial <= projection.originNear.w) { return 1.0; }
    float3 delta = samplePosition - projection.originNear.xyz;
    float distance = length(delta);
    float biasScale = max(1.0 - projection.metric.z / distance, projection.originNear.w / axial);
    float3 biasedPosition = projection.originNear.xyz + delta * biasScale;
    float4 clip = vsmProjectClip(biasedPosition, projection);
    return clip.z / clip.w;
}

