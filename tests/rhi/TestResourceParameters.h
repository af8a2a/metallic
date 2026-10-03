#pragma once

// Shared CPU/Slang resource declarations for GPU fixtures. Binding IDs exist
// only on the CPU; shaders load these named descriptor handles and spans.
#ifdef __cplusplus
#include "Runtime/Render/Core/ShaderResourceABI.h"
namespace metallic::tests {
#define TEST_PUBLIC
#define TEST_BUFFER render::GPUResourceHandle<render::ResourceViewKind::RawBuffer>
#define TEST_SAMPLED(T) render::GPUResourceHandle<render::ResourceViewKind::SampledImage>
#define TEST_STORAGE(T) render::GPUResourceHandle<render::ResourceViewKind::StorageImage>
#define TEST_SPAN render::GPUBufferSpan
#define TEST_SAMPLER render::GPUSamplerHandle
#else
#define TEST_PUBLIC public
#define TEST_BUFFER ResourceHandle<ByteAddressBuffer>
#define TEST_SAMPLED(T) ResourceHandle<Texture2D<T>>
#define TEST_STORAGE(T) ResourceHandle<RWTexture2D<T>>
#define TEST_SPAN BufferSpan<uint>
#define TEST_SAMPLER SamplerHandle
#endif

TEST_PUBLIC struct AutoExposureFixtureResources
{
    TEST_PUBLIC TEST_STORAGE(float4) output;
};
#ifdef __cplusplus
static_assert(sizeof(AutoExposureFixtureResources) == 4);
#endif

TEST_PUBLIC struct BatchBarrierProbeResources
{
    TEST_PUBLIC TEST_BUFFER output;
};
#ifdef __cplusplus
static_assert(sizeof(BatchBarrierProbeResources) == 4);
#endif

TEST_PUBLIC struct ClusterLightGridLookupProbeResources
{
    TEST_PUBLIC TEST_BUFFER parameters;
    TEST_PUBLIC TEST_BUFFER lights;
    TEST_PUBLIC TEST_BUFFER candidates;
    TEST_PUBLIC TEST_BUFFER cells;
    TEST_PUBLIC TEST_BUFFER lightIndices;
    TEST_PUBLIC TEST_BUFFER output;
};
#ifdef __cplusplus
static_assert(sizeof(ClusterLightGridLookupProbeResources) == 24);
#endif

TEST_PUBLIC struct DataSliceProbeResources
{
    TEST_PUBLIC TEST_SPAN output;
};
#ifdef __cplusplus
static_assert(sizeof(DataSliceProbeResources) == 12);
#endif

TEST_PUBLIC struct DLSSMotionVectorProbeResources
{
    TEST_PUBLIC TEST_BUFFER motionResults;
};
#ifdef __cplusplus
static_assert(sizeof(DLSSMotionVectorProbeResources) == 4);
#endif

TEST_PUBLIC struct EnvironmentPrefilterFieldProbeResources
{
    TEST_PUBLIC TEST_BUFFER output;
    TEST_PUBLIC TEST_BUFFER coefficients;
    TEST_PUBLIC TEST_SAMPLED(float4) environment;
};
#ifdef __cplusplus
static_assert(sizeof(EnvironmentPrefilterFieldProbeResources) == 12);
#endif

TEST_PUBLIC struct FrameEnvironmentProbeResources
{
    TEST_PUBLIC TEST_SAMPLED(float4) environment;
    TEST_PUBLIC TEST_BUFFER output;
};
#ifdef __cplusplus
static_assert(sizeof(FrameEnvironmentProbeResources) == 8);
#endif

TEST_PUBLIC struct FrameCopyProbeResources
{
    TEST_PUBLIC TEST_BUFFER input;
    TEST_PUBLIC TEST_BUFFER output;
};
#ifdef __cplusplus
static_assert(sizeof(FrameCopyProbeResources) == 8);
#endif

TEST_PUBLIC struct FrameHistoryProbeResources
{
    TEST_PUBLIC TEST_STORAGE(float4) previous;
    TEST_PUBLIC TEST_STORAGE(float4) current;
    TEST_PUBLIC TEST_BUFFER output;
};
#ifdef __cplusplus
static_assert(sizeof(FrameHistoryProbeResources) == 12);
#endif

TEST_PUBLIC struct FrameImagesProbeResources
{
    TEST_PUBLIC TEST_SPAN images;
    TEST_PUBLIC TEST_BUFFER output;
};
#ifdef __cplusplus
static_assert(sizeof(FrameImagesProbeResources) == 16);
#endif

TEST_PUBLIC struct GPUDrivenConeProbeResources
{
    TEST_PUBLIC TEST_BUFFER output;
};
#ifdef __cplusplus
static_assert(sizeof(GPUDrivenConeProbeResources) == 4);
#endif

TEST_PUBLIC struct HZBSPDFixtureResources
{
    TEST_PUBLIC TEST_STORAGE(float) depth;
    TEST_PUBLIC TEST_BUFFER counter;
};
#ifdef __cplusplus
static_assert(sizeof(HZBSPDFixtureResources) == 8);
#endif

TEST_PUBLIC struct MaterialBinningProbeResources
{
    TEST_PUBLIC TEST_STORAGE(uint) visibility;
    TEST_PUBLIC TEST_BUFFER records;
    TEST_PUBLIC TEST_BUFFER instances;
    TEST_PUBLIC TEST_BUFFER materials;
    TEST_PUBLIC TEST_BUFFER shadingMaterials;
    TEST_PUBLIC TEST_BUFFER bins;
    TEST_PUBLIC TEST_BUFFER tiles;
    TEST_PUBLIC TEST_BUFFER arguments;
    TEST_PUBLIC TEST_BUFFER output;
};
#ifdef __cplusplus
static_assert(sizeof(MaterialBinningProbeResources) == 36);
#endif

TEST_PUBLIC struct MaterialRuntimeProbeResources
{
    TEST_PUBLIC TEST_BUFFER materials;
    TEST_PUBLIC TEST_BUFFER output;
};
#ifdef __cplusplus
static_assert(sizeof(MaterialRuntimeProbeResources) == 8);
#endif

TEST_PUBLIC struct NativeDescriptorHandlesResources
{
    TEST_PUBLIC TEST_BUFFER records;
    TEST_PUBLIC TEST_BUFFER output;
};
#ifdef __cplusplus
static_assert(sizeof(NativeDescriptorHandlesResources) == 8);
#endif

TEST_PUBLIC struct PhotometricProbeResources
{
    TEST_PUBLIC TEST_BUFFER output;
    TEST_PUBLIC TEST_BUFFER irradiance;
    TEST_PUBLIC TEST_BUFFER lights;
};
#ifdef __cplusplus
static_assert(sizeof(PhotometricProbeResources) == 12);
#endif

TEST_PUBLIC struct RealtimeGuideProbeResources
{
    TEST_PUBLIC TEST_SAMPLED(float2) motion;
    TEST_PUBLIC TEST_SAMPLED(float) depth;
    TEST_PUBLIC TEST_BUFFER data;
    TEST_PUBLIC TEST_BUFFER prefilter;
};
#ifdef __cplusplus
static_assert(sizeof(RealtimeGuideProbeResources) == 16);
#endif

TEST_PUBLIC struct ReGIRVirtualLightProbeResources
{
    TEST_PUBLIC TEST_BUFFER output;
    TEST_PUBLIC TEST_BUFFER lights;
    TEST_PUBLIC TEST_BUFFER lightAlias;
    TEST_PUBLIC TEST_SAMPLED(float) lightsPdf;
};
#ifdef __cplusplus
static_assert(sizeof(ReGIRVirtualLightProbeResources) == 16);
#endif

TEST_PUBLIC struct SceneUploadProbeResources
{
    TEST_PUBLIC TEST_SAMPLED(float4) texture;
    TEST_PUBLIC TEST_BUFFER output;
};
#ifdef __cplusplus
static_assert(sizeof(SceneUploadProbeResources) == 8);
#endif

TEST_PUBLIC struct SliderDebugFixtureResources
{
    TEST_PUBLIC TEST_STORAGE(float4) output;
    TEST_PUBLIC TEST_SAMPLED(float4) input;
    TEST_PUBLIC TEST_BUFFER readback;
};
#ifdef __cplusplus
static_assert(sizeof(SliderDebugFixtureResources) == 12);
#endif

TEST_PUBLIC struct SphericalHarmonicsProbeResources
{
    TEST_PUBLIC TEST_BUFFER output;
    TEST_PUBLIC TEST_BUFFER input;
};
#ifdef __cplusplus
static_assert(sizeof(SphericalHarmonicsProbeResources) == 8);
#endif

TEST_PUBLIC struct TextureFootprintProbeResources
{
    TEST_PUBLIC TEST_BUFFER output;
};
#ifdef __cplusplus
static_assert(sizeof(TextureFootprintProbeResources) == 4);
#endif

TEST_PUBLIC struct TextureStreamingProbeResources
{
    TEST_PUBLIC TEST_SPAN textures;
    TEST_PUBLIC TEST_BUFFER feedback;
    TEST_PUBLIC TEST_BUFFER output;
    TEST_PUBLIC TEST_SAMPLER sampler;
};
#ifdef __cplusplus
static_assert(sizeof(TextureStreamingProbeResources) == 24);
#endif

TEST_PUBLIC struct TwoPassOcclusionProbeResources
{
    TEST_PUBLIC TEST_BUFFER output;
    TEST_PUBLIC TEST_BUFFER previousHzb;
    TEST_PUBLIC TEST_BUFFER currentHzb;
};
#ifdef __cplusplus
static_assert(sizeof(TwoPassOcclusionProbeResources) == 12);
#endif

TEST_PUBLIC struct UnifiedTopLevelProbeResources
{
    TEST_PUBLIC uint64_t scene;
    TEST_PUBLIC TEST_BUFFER output;
};
#ifdef __cplusplus
static_assert(sizeof(UnifiedTopLevelProbeResources) == 16);
#endif

TEST_PUBLIC struct ViewConstantsProbeResources
{
    TEST_PUBLIC TEST_BUFFER output;
    TEST_PUBLIC TEST_BUFFER input;
};
#ifdef __cplusplus
static_assert(sizeof(ViewConstantsProbeResources) == 8);
#endif

#undef TEST_PUBLIC
#undef TEST_BUFFER
#undef TEST_SAMPLED
#undef TEST_STORAGE
#undef TEST_SPAN
#undef TEST_SAMPLER
#ifdef __cplusplus
} // namespace metallic::tests
#endif
