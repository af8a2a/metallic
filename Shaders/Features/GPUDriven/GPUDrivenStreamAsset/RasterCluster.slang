#language slang 2026
implementing GPUDrivenStreamAsset;

import ShaderCore;
using Metallic;

import GPUDriven;
using Metallic.GPUDriven;
import Material;
using Metallic.Material;
import AlphaCoverage;

// Raster cluster validation, geometry decoding, projection and alpha coverage.
public struct StreamRasterCluster
{
    public GPUDrivenStreamAssetParams params;
    public GPUDrivenStreamAssetActiveGroup group;
    public uint pageWord;
    public uint positionOffsetBytes;
    public uint positionStrideWords;
    public uint vertexOffset;
    public uint vertexCount;
    public uint triangleCount;
    public uint triangleBaseByte;
    public uint recordIndex;
    public uint clusterIndex;
    public uint instanceFlags;
    public float4 worldSphere;
    public uint classificationFlags;
    public bool forceHardware;
};

// A projected enclosing sphere bounds every triangle bbox, but a large sphere
// does not imply any large triangle. Only accept SW here; uncertainty falls back.
bool streamSphereFitsSoftware(StreamRasterCluster c, float2 viewport, float maxPixels)
{
    if (!all(isfinite(c.worldSphere)) || c.worldSphere.w < 0.0) { return false; }
    GPUDrivenStreamAssetParams p = c.params;
    GPUDrivenStreamAssetCameraData camera = currentStreamCamera(p);
    float2 jitter = float2(0);
    if (p.renderViewport.y > 0.0) {
        camera.eye = p.renderEye; camera.center = p.renderCenter;
        camera.upProjection = p.renderUpProjection; camera.viewport = p.renderViewport;
        camera.clipOrtho = p.renderClipOrtho;
        jitter = float2(p.renderEye.w, p.renderCenter.w);
    }
    float3 v = streamWorldToView(c.worldSphere.xyz, camera);
    // Outward slack covers sphere/affine/view arithmetic; uncertain boundaries
    // additionally retain a 1/16 pixel margin and use the exact classifier.
    float magnitude = max(max(abs(c.worldSphere.x), abs(c.worldSphere.y)), abs(c.worldSphere.z)) +
        max(max(abs(camera.eye.x), abs(camera.eye.y)), abs(camera.eye.z)) + 1.0;
    float r = c.worldSphere.w * 1.0001 + magnitude * 0.000002;
    float nearZ = max(camera.clipOrtho.x, 0.0001);
    float farZ = max(camera.clipOrtho.y, nearZ + 0.0001);
    if (!all(isfinite(v)) || v.z - r <= nearZ || v.z + r >= farZ) { return false; }
    float2 lo = v.xy - r, hi = v.xy + r;
    float aspect = max(camera.viewport.x, 0.001);
    if (camera.upProjection.w > 0.5) {
        float h = max(camera.clipOrtho.z * 0.5, 0.0001);
        lo /= float2(h * aspect, h); hi /= float2(h * aspect, h);
    } else {
        // Interval division with strictly positive depth. Includes off-axis
        // perspective growth and handles reflected/nonuniform instances.
        float2 a = lo / (v.z - r), b = lo / (v.z + r);
        float2 d = hi / (v.z - r), e = hi / (v.z + r);
        float yScale = 1.0 / tan(clamp(camera.viewport.w, 0.017453292, 3.12413936) * 0.5);
        lo = min(a,b) * float2(yScale / aspect,yScale);
        hi = max(d,e) * float2(yScale / aspect,yScale);
    }
    float2 lower = (float2(lo.x, -hi.y) * 0.5 + 0.5) * viewport - jitter - 0.0625;
    float2 upper = (float2(hi.x, -lo.y) * 0.5 + 0.5) * viewport - jitter + 0.0625;
    return all(isfinite(lower)) && all(isfinite(upper)) && all(upper - lower <= maxPixels) &&
        all(lower >= -32.0) && all(upper <= viewport + 32.0);
}

public bool loadStreamRasterCluster(uint localRecordIndex, bool testVisibility, bool publishRecord,
    GPUDrivenStreamAssetPush push, out StreamRasterCluster output, out bool retryLate)
{
    retryLate = false;
    output = (StreamRasterCluster)0;
    StructuredBuffer<uint> pageHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.pageBuffer));
    StructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroupHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetActiveGroup>>(push.activeGroupBuffer));
    StructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeaderHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetActiveHeader>>(push.activeHeaderBuffer));
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTableHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry>>(push.pageTableBuffer));
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetParams>>(push.paramsBuffer));
    RWStructuredBuffer<uint> requestHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.requestBuffer));
    StructuredBuffer<GPUDrivenStreamAssetRasterBindings> rasterBindingsHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(push.rasterBindingsBuffer));

    StructuredBuffer<uint> pageWords = pageHandle;
    StructuredBuffer<GPUDrivenStreamAssetActiveGroup> activeGroups = activeGroupHandle;
    StructuredBuffer<GPUDrivenStreamAssetActiveHeader> activeHeader = activeHeaderHandle;
    RWStructuredBuffer<GPUDrivenStreamAssetPageTableEntry> pageTable = pageTableHandle;
    StructuredBuffer<GPUDrivenStreamAssetParams> paramsBuffer = paramsHandle;
    RWStructuredBuffer<uint> requests = requestHandle;
    StructuredBuffer<GPUDrivenStreamAssetRasterBindings> rasterBindingsBuffer = rasterBindingsHandle;

    GPUDrivenStreamAssetParams params = paramsBuffer[0];
    GPUDrivenStreamAssetRasterBindings rasterBindings = rasterBindingsBuffer[0];
    if (localRecordIndex >= params.drawTaskCount / kMeshletTriangleChunkCount) { return false; }
    uint recordCapacity = rasterBindings.visibleRecordCapacity != 0u
        ? rasterBindings.visibleRecordCapacity
        : params.drawTaskCount / kMeshletTriangleChunkCount;
    uint recordIndex = 0u;
    if (!resolveVisibilityRecordIndex(
            localRecordIndex,
            rasterBindings.visibleRecordBase,
            recordCapacity,
            recordIndex)) {
        return false;
    }

    GPUDrivenStreamAssetActiveHeader header = activeHeader[0];
    uint maxActiveGroupClusters = max(header.maxActiveGroupClusters, 1u);
    uint activeGroupIndex = localRecordIndex / maxActiveGroupClusters;
    uint clusterIndex = localRecordIndex - activeGroupIndex * maxActiveGroupClusters;
    uint activeGroupCount = min(header.activeGroupCount, header.activeGroupCapacity);
    if (activeGroupIndex >= activeGroupCount) {
        return false;
    }

    GPUDrivenStreamAssetActiveGroup group = activeGroups[activeGroupIndex];
    uint instanceFlags = 0u;
    if (rasterBindings.gpuSceneInstanceBuffer != 0xffffffffu) {
        StructuredBuffer<GPUDrivenPreviewInstance> instances =
            resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenPreviewInstance>>(rasterBindings.gpuSceneInstanceBuffer));
        StructuredBuffer<GPUDrivenPreviewInstance> instanceBuffer = instances;
        uint count, stride;
        instanceBuffer.GetDimensions(count, stride);
        if (group.gpuSceneInstanceIndex >= count) { return false; }
        instanceFlags = instances[group.gpuSceneInstanceIndex].identity.w;
    }
    if (group.pageIndex >= params.scenePageCount ||
        group.gpuSceneInstanceIndex == kInvalidClusterIndex ||
        clusterIndex >= group.clusterCount ||
        clusterIndex >= 32u) {
        return false;
    }
    if ((group.flags & kActiveGroupLoadRequest) != 0u) {
        if (testVisibility && clusterIndex == 0u) {
            requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
        }
        return false;
    }
    if ((group.flags & kActiveGroupUnloadRequest) != 0u) {
        if (testVisibility && clusterIndex == 0u) {
            requestStreamPageUnload(requests, params, group.pageIndex);
        }
        return false;
    }

    GPUDrivenStreamAssetPageTableEntry pageEntry = pageTable[group.pageIndex];
    uint pageState = streamPageState(pageEntry);
    uint pageDeviceOffset = streamPageDeviceOffset(pageEntry);
    if (((group.clusterSelectionMask >> clusterIndex) & 1u) == 0u ||
        ((pageState != kStreamStateResident && pageState != kStreamStateLockedFallback) ||
        pageDeviceOffset == kInvalidStreamDeviceOffset)) {
        if (testVisibility && clusterIndex == 0u) {
            requestStreamPageLoad(pageTable, requests, params, group.pageIndex);
        }
        return false;
    }
    if (params.pageBufferBytes < kPayloadHeaderBytes ||
        pageDeviceOffset > params.pageBufferBytes - kPayloadHeaderBytes) {
        return false;
    }

    uint pageWord = pageDeviceOffset / 4u;
    uint payloadBytes = pageWords[pageWord + 12u];
    if (payloadBytes < kPayloadHeaderBytes ||
        payloadBytes > params.pageBufferBytes ||
        pageDeviceOffset > params.pageBufferBytes - payloadBytes) {
        return false;
    }
    uint clusterCount = pageWords[pageWord + 2u];
    uint payloadVertexCount = pageWords[pageWord + 3u];
    uint triangleIndexCount = pageWords[pageWord + 4u];
    uint attributeFlags = pageWords[pageWord + 14u];
    uint positionFormat = pageWords[pageWord + 20u];
    if (clusterIndex >= clusterCount ||
        (attributeFlags & kPayloadAttributePosition) == 0u ||
        (positionFormat != kPayloadFormatFloat32x4 && positionFormat != kPayloadFormatFloat32x3)) {
        return false;
    }

    uint clusterOffsetBytes = pageWords[pageWord + 9u];
    uint positionOffsetBytes = pageWords[pageWord + 10u];
    uint triangleOffsetBytes = pageWords[pageWord + 11u];
    if ((clusterOffsetBytes & 3u) != 0u ||
        (positionOffsetBytes & 3u) != 0u ||
        !streamPayloadArrayRangeValid(
            clusterOffsetBytes,
            clusterCount,
            kPayloadClusterBytes,
            payloadBytes) ||
        !streamPayloadArrayRangeValid(
            positionOffsetBytes,
            payloadVertexCount,
            positionFormat == kPayloadFormatFloat32x3 ? 12u : 16u,
            payloadBytes) ||
        !streamPayloadRangeValid(
            triangleOffsetBytes,
            triangleIndexCount,
            payloadBytes)) {
        return false;
    }
    uint clusterWord = pageWord + clusterOffsetBytes / 4u +
        clusterIndex * kPayloadClusterWords;
    uint vertexOffset = pageWords[clusterWord + 0u];
    uint vertexCount = pageWords[clusterWord + 1u];
    uint triangleOffset = pageWords[clusterWord + 2u];
    uint clusterTriangleCount = pageWords[clusterWord + 3u];
    if (vertexCount == 0u ||
        vertexCount > kMaxMeshletVertices ||
        vertexOffset > payloadVertexCount ||
        vertexCount > payloadVertexCount - vertexOffset ||
        clusterTriangleCount > kMaxMeshletTriangles ||
        triangleOffset > triangleIndexCount ||
        clusterTriangleCount > (triangleIndexCount - triangleOffset) / 3u) {
        return false;
    }
    float3 localSphereCenter = float3(
        asfloat(pageWords[clusterWord + 12u]),
        asfloat(pageWords[clusterWord + 13u]),
        asfloat(pageWords[clusterWord + 14u]));
    float rawSphereRadius = asfloat(pageWords[clusterWord + 15u]);
    float localSphereRadius = max(rawSphereRadius, 0.0);
    float3 worldSphereCenter =
        group.world0.xyz * localSphereCenter.x +
        group.world1.xyz * localSphereCenter.y +
        group.world2.xyz * localSphereCenter.z +
        group.world3.xyz;
    float worldScale = conservativeTransformScale(group.world0.xyz, group.world1.xyz, group.world2.xyz);
    float worldSphereRadius = localSphereRadius * worldScale + rasterBindings.displacementBound;
    GPUDrivenStreamAssetCameraData rasterCamera = currentStreamCamera(params);
    if (testVisibility && (rasterBindings.cullingFlags & (1u << 2u)) != 0u &&
        !streamSphereInsideFrustum(
            worldSphereCenter,
            worldSphereRadius,
            rasterCamera)) {
        return false;
    }

    float coneCutoff = asfloat(pageWords[clusterWord + 19u]);
    if (testVisibility && rasterBindings.displacementBound == 0.0 && (instanceFlags & 2u) == 0u &&
        (rasterBindings.cullingFlags & (1u << 3u)) != 0u && coneCutoff < 0.9999) {
        float3 localConeApex = float3(
            asfloat(pageWords[clusterWord + 16u]),
            asfloat(pageWords[clusterWord + 17u]),
            asfloat(pageWords[clusterWord + 18u]));
        float3 localConeAxis = float3(
            asfloat(pageWords[clusterWord + 20u]),
            asfloat(pageWords[clusterWord + 21u]),
            asfloat(pageWords[clusterWord + 22u]));
        float3 worldConeApex =
            group.world0.xyz * localConeApex.x +
            group.world1.xyz * localConeApex.y +
            group.world2.xyz * localConeApex.z +
            group.world3.xyz;
        float3 worldConeAxis = transformStreamNormal(
            group,
            localConeAxis,
            float3(0.0, 0.0, 1.0));
        // Cofactors include the determinant sign. Match the corrected raster
        // winding without changing authored normals used by material shading.
        if (dot(cross(group.world0.xyz, group.world1.xyz), group.world2.xyz) < 0.0) {
            worldConeAxis = -worldConeAxis;
        }
        float3 viewDirection = safeNormalize(
            worldConeApex - rasterCamera.eye.xyz,
            -worldConeAxis);
        if (dot(viewDirection, worldConeAxis) >= coneCutoff) {
            return false;
        }
    }

    StructuredBuffer<uint> visibilityStateHandle =
        resolveUniform(ResourceHandle<StructuredBuffer<uint>>(rasterBindings.instanceVisibilityBuffer));
    StructuredBuffer<uint> visibilityStates = visibilityStateHandle;
    if (testVisibility && !streamClusterVisibleInOcclusionPhase(worldSphereCenter, worldSphereRadius,
        visibilityStates[group.gpuSceneInstanceIndex], params, rasterBindings, push, retryLate)) {
        return false;
    }

    RWStructuredBuffer<CompactStreamVisibleRecord> visibleClusterHandle =
        resolveUniform(ResourceHandle<RWStructuredBuffer<CompactStreamVisibleRecord>>(rasterBindings.visibleClusterBuffer));
    RWStructuredBuffer<CompactStreamVisibleRecord> visibleClusters = visibleClusterHandle;
    if (publishRecord) {
        uint flags = ((instanceFlags & 2u) != 0u ? 1u : 0u) | ((instanceFlags & 4u) != 0u ? 2u : 0u);
        visibleClusters[localRecordIndex] = packStreamVisibleRecord(activeGroupIndex, clusterIndex, flags);
    }

    output.params = params;
    output.group = group;
    output.pageWord = pageWord;
    output.positionOffsetBytes = positionOffsetBytes;
    output.positionStrideWords = positionFormat == kPayloadFormatFloat32x3 ? 3u : 4u;
    output.vertexOffset = vertexOffset;
    output.vertexCount = vertexCount;
    output.triangleCount = clusterTriangleCount;
    output.triangleBaseByte = pageDeviceOffset + triangleOffsetBytes + triangleOffset;
    output.recordIndex = recordIndex;
    output.clusterIndex = clusterIndex;
    output.instanceFlags = instanceFlags;
    // Include cancellation in local-to-world arithmetic in the fast-path
    // slack; the culling sphere and exact classifier remain unchanged.
    float3 affineMagnitude = abs(group.world0.xyz * localSphereCenter.x) +
        abs(group.world1.xyz * localSphereCenter.y) + abs(group.world2.xyz * localSphereCenter.z) + abs(group.world3.xyz);
    float affineSlack = max(max(affineMagnitude.x, affineMagnitude.y), affineMagnitude.z) * 0.000002;
    output.worldSphere = float4(worldSphereCenter, worldSphereRadius + affineSlack);
    output.classificationFlags = rasterBindings.classificationFlags |
        ((!isfinite(rawSphereRadius) || rawSphereRadius < 0.0) ? 1u : 0u);
    output.forceHardware = (instanceFlags & 12u) != 0u || rasterBindings.tessellationBuffer != 0xffffffffu;
    return true;
}

public float4 streamClusterClip(StreamRasterCluster c, uint vertex, GPUDrivenStreamAssetPush push)
{
    StructuredBuffer<uint> pages =
        resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.pageBuffer));
    uint word = c.pageWord + c.positionOffsetBytes / 4u + (c.vertexOffset + vertex) * c.positionStrideWords;
    float3 local = float3(asfloat(pages[word]), asfloat(pages[word + 1u]), asfloat(pages[word + 2u]));
    float3 world = c.group.world0.xyz * local.x + c.group.world1.xyz * local.y +
        c.group.world2.xyz * local.z + c.group.world3.xyz;
    float4 clip = projectPosition(world, c.params);
    clip.y = -clip.y;
    return clip;
}

// Preserve projection arithmetic, but build camera-only terms once per cluster.
public struct StreamRasterProjection
{
    public float3 eye; float3 forward; float3 right; float3 up;
    public float nearZ; float farZ; float aspect; float halfHeight; float yScale;
    public float2 jitter;
    public bool ortho; bool jitterEnabled;
};
public StreamRasterProjection prepareStreamRasterProjection(GPUDrivenStreamAssetParams p)
{
    StreamRasterProjection r;
    r.jitterEnabled = p.renderViewport.y > 0.0;
    r.jitter = r.jitterEnabled ? 2.0 * float2(p.renderEye.w,-p.renderCenter.w) / p.renderViewport.yz : float2(0);
    if (r.jitterEnabled) {
        p.eye=p.renderEye; p.center=p.renderCenter; p.upProjection=p.renderUpProjection;
        p.viewport=p.renderViewport; p.clipOrtho=p.renderClipOrtho;
    }
    r.eye=p.eye.xyz;
    r.forward=safeNormalize(p.center.xyz-p.eye.xyz,float3(0,0,-1));
    float3 cameraUp=safeNormalize(p.upProjection.xyz,float3(0,1,0));
    r.right=safeNormalize(cross(r.forward,cameraUp),float3(1,0,0));
    r.up=safeNormalize(cross(r.right,r.forward),float3(0,1,0));
    r.nearZ=max(p.clipOrtho.x,0.0001); r.farZ=max(p.clipOrtho.y,r.nearZ+0.0001);
    r.aspect=max(p.viewport.x,0.001); r.halfHeight=max(p.clipOrtho.z*0.5,0.0001);
    r.yScale=1.0/tan(clamp(p.viewport.w,0.017453292,3.12413936)*0.5);
    r.ortho=p.upProjection.w>0.5;
    return r;
}
public float4 streamClusterClipCached(StreamRasterCluster c, uint vertex, GPUDrivenStreamAssetPush push, StreamRasterProjection r)
{
    StructuredBuffer<uint> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.pageBuffer));
    uint word=c.pageWord+c.positionOffsetBytes/4u+(c.vertexOffset+vertex)*c.positionStrideWords;
    float3 local=asfloat(uint3(pages[word],pages[word+1u],pages[word+2u]));
    float3 world=c.group.world0.xyz*local.x+c.group.world1.xyz*local.y+c.group.world2.xyz*local.z+c.group.world3.xyz;
    float3 delta=world-r.eye;
    float3 view=float3(dot(delta,r.right),dot(delta,r.up),dot(delta,r.forward));
    float z=conservativeProjectDepth(view.z,r.nearZ,r.farZ,r.ortho);
    float4 clip=r.ortho ? float4(view.x/(r.halfHeight*r.aspect),view.y/r.halfHeight,z,1.0) :
        float4(view.x*r.yScale/r.aspect,view.y*r.yScale,z*view.z,view.z);
    if (r.jitterEnabled) { clip.xy-=r.jitter*clip.w; }
    clip.y=-clip.y;
    return clip;
}
public groupshared StreamRasterProjection g_StreamRasterProjection;

public uint3 streamClusterTriangle(StreamRasterCluster c, uint triangle, GPUDrivenStreamAssetPush push)
{
    StructuredBuffer<uint> pages =
        resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.pageBuffer));
    uint address = c.triangleBaseByte + triangle * 3u;
    uint3 indices = uint3(readPageByte(pages, address), readPageByte(pages, address + 1u), readPageByte(pages, address + 2u));
    // Retain the existing malformed-index fallback in the stream HW decoder.
    indices = uint3(indices.x < c.vertexCount ? indices.x : 0u,
        indices.y < c.vertexCount ? indices.y : 0u, indices.z < c.vertexCount ? indices.z : 0u);
    // A reflected instance reverses projected winding. Keep authored front
    // faces front-facing in both hardware and software rasterization; the
    // visibility ID still names the original payload triangle for shading.
    bool reflected = dot(cross(c.group.world0.xyz, c.group.world1.xyz), c.group.world2.xyz) < 0.0;
    return reflected ? indices.xzy : indices;
}

float2 streamClusterUv(StreamRasterCluster c, uint vertex, GPUDrivenStreamAssetPush push)
{
    StructuredBuffer<uint> pages = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(push.pageBuffer));
    uint bytes = pages[c.pageWord + 12u], offset = pages[c.pageWord + 17u];
    if ((pages[c.pageWord + 14u] & 4u) == 0u || pages[c.pageWord + 22u] != 1u ||
        (offset & 3u) != 0u || offset > bytes || c.vertexOffset + vertex >= (bytes - offset) / 8u) { return float2(0); }
    uint uv = c.pageWord + offset / 4u + (c.vertexOffset + vertex) * 2u;
    return asfloat(uint2(pages[uv], pages[uv + 1u]));
}

// Alpha is evaluated before depth/visibility writes. Masked and blended
// clusters bypass the software queue, which has no material coverage stage.
bool streamAlphaAccepts(uint instanceIndex, float2 uv, GPUDrivenStreamAssetPush push)
{
    StructuredBuffer<GPUDrivenStreamAssetRasterBindings> rh = resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenStreamAssetRasterBindings>>(push.rasterBindingsBuffer));
    GPUDrivenStreamAssetRasterBindings bindings = rh[0];
    if (bindings.gpuSceneInstanceBuffer == 0xffffffffu || bindings.materialBuffer == 0xffffffffu) { return true; }
    StructuredBuffer<GPUDrivenPreviewInstance> instances = resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenPreviewInstance>>(bindings.gpuSceneInstanceBuffer));
    GPUDrivenPreviewInstance instance = instances[instanceIndex];
    if ((instance.identity.w & 12u) == 0u) { return true; }
    StructuredBuffer<GPUDrivenPreviewMaterial> materials = resolveUniform(ResourceHandle<StructuredBuffer<GPUDrivenPreviewMaterial>>(bindings.materialBuffer));
    GPUDrivenPreviewMaterial material = materials[instance.identity.y];
    CoverageContext context;
    context.uv = uv; context.baseAlpha = 1.0;
    context.alphaMode = (instance.identity.w & 4u) != 0u ? 1.0 : 2.0;
    context.cutoff = material.params.z;
    MaterialInstanceRef reference = coverageInstanceRef(material.identity.z == 0xffffffffu ? 0u : material.identity.z);
    bool needsAlpha = true;
    if (bindings.coverageInputs != 0xffffffffu && material.identity.z != 0xffffffffu) {
        CoverageProgram program;
        program.inputs = resolveUniform(ResourceHandle<ByteAddressBuffer>(bindings.coverageInputs));
        needsAlpha = program.usesBaseAlpha(reference);
    }
    float alpha = material.baseColor.a;
    GPUDrivenPreviewTextureInfo info = material.baseColorTexture;
    if (needsAlpha && info.textureIndex < bindings.materialTextureCount && bindings.materialTextureRemapBuffer != 0xffffffffu) {
        StructuredBuffer<uint> remap = resolveUniform(ResourceHandle<StructuredBuffer<uint>>(bindings.materialTextureRemapBuffer));
        Texture2D<float4> texture = resolveNonUniform(ResourceHandle<Texture2D<float4>>(remap[info.textureIndex]));
        float2 transformed = float2(dot(info.transform0.xy, uv) + info.transform0.z, dot(info.transform1.xy, uv) + info.transform1.z);
        alpha *= sampleAlphaCoverage(texture, transformed);
    }
    context.baseAlpha = alpha;
    if (bindings.coverageInputs != 0xffffffffu && material.identity.z != 0xffffffffu) {
        // Resolve inside the branch; never merge a buffer with an undefined arm.
        CoverageProgram program;
        program.inputs = resolveUniform(ResourceHandle<ByteAddressBuffer>(bindings.coverageInputs));
        return program.evaluateCoverage(context, reference).accepted;
    }
    return coverageFromOpacity(context, context.baseAlpha).accepted;
}

// Used by the bounded-queue HW fallback as well as tessellation task shaders.
void publishStreamHardwareRetry(uint record, bool retryLate, GPUDrivenStreamAssetPush push)
{
    if (push.traversalPhase == 0u && retryLate) {
        RWStructuredBuffer<uint> bins =
            resolveUniform(ResourceHandle<RWStructuredBuffer<uint>>(push.hybridClusterBuffer));
        InterlockedOr(bins[hybridRetryBase(bins[5]) + record / bins[13]], 1u << (record % bins[13]));
    }
}

