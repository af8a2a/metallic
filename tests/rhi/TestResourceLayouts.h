#pragma once

#include "TestResourceParameters.h"
#include "Runtime/Render/Core/ComputeProgram.h"
#include <cstddef>

namespace metallic::tests {

inline constexpr render::ComputeResourceField kAutoExposureFixtureFields[] = {
    {0, render::ComputeResourceBindingKind::StorageImage, offsetof(AutoExposureFixtureResources, output), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kAutoExposureFixtureLayout{sizeof(AutoExposureFixtureResources), kAutoExposureFixtureFields};

inline constexpr render::ComputeResourceField kBatchBarrierProbeFields[] = {
    {0, render::ComputeResourceBindingKind::StorageBuffer, offsetof(BatchBarrierProbeResources, output), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kBatchBarrierProbeLayout{sizeof(BatchBarrierProbeResources), kBatchBarrierProbeFields};

inline constexpr render::ComputeResourceField kClusterLightGridLookupProbeFields[] = {
    {0, render::ComputeResourceBindingKind::StorageBuffer, offsetof(ClusterLightGridLookupProbeResources, parameters), render::ComputeResourceFieldFormat::Handle},
    {1, render::ComputeResourceBindingKind::StorageBuffer, offsetof(ClusterLightGridLookupProbeResources, lights), render::ComputeResourceFieldFormat::Handle},
    {2, render::ComputeResourceBindingKind::StorageBuffer, offsetof(ClusterLightGridLookupProbeResources, candidates), render::ComputeResourceFieldFormat::Handle},
    {3, render::ComputeResourceBindingKind::StorageBuffer, offsetof(ClusterLightGridLookupProbeResources, cells), render::ComputeResourceFieldFormat::Handle},
    {4, render::ComputeResourceBindingKind::StorageBuffer, offsetof(ClusterLightGridLookupProbeResources, lightIndices), render::ComputeResourceFieldFormat::Handle},
    {5, render::ComputeResourceBindingKind::StorageBuffer, offsetof(ClusterLightGridLookupProbeResources, output), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kClusterLightGridLookupProbeLayout{sizeof(ClusterLightGridLookupProbeResources), kClusterLightGridLookupProbeFields};

inline constexpr render::ComputeResourceField kDataSliceProbeFields[] = {
    {0, render::ComputeResourceBindingKind::DataBuffer, offsetof(DataSliceProbeResources, output), render::ComputeResourceFieldFormat::DataSpan},
};
inline constexpr render::ComputeResourceLayout kDataSliceProbeLayout{sizeof(DataSliceProbeResources), kDataSliceProbeFields};

inline constexpr render::ComputeResourceField kDLSSMotionVectorProbeFields[] = {
    {63, render::ComputeResourceBindingKind::StorageBuffer, offsetof(DLSSMotionVectorProbeResources, motionResults), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kDLSSMotionVectorProbeLayout{sizeof(DLSSMotionVectorProbeResources), kDLSSMotionVectorProbeFields};

inline constexpr render::ComputeResourceField kEnvironmentPrefilterFieldProbeFields[] = {
    {0, render::ComputeResourceBindingKind::StorageBuffer, offsetof(EnvironmentPrefilterFieldProbeResources, output), render::ComputeResourceFieldFormat::Handle},
    {1, render::ComputeResourceBindingKind::StorageBuffer, offsetof(EnvironmentPrefilterFieldProbeResources, coefficients), render::ComputeResourceFieldFormat::Handle},
    {2, render::ComputeResourceBindingKind::SampledImage, offsetof(EnvironmentPrefilterFieldProbeResources, environment), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kEnvironmentPrefilterFieldProbeLayout{sizeof(EnvironmentPrefilterFieldProbeResources), kEnvironmentPrefilterFieldProbeFields};

inline constexpr render::ComputeResourceField kFrameEnvironmentProbeFields[] = {
    {0, render::ComputeResourceBindingKind::SampledImage, offsetof(FrameEnvironmentProbeResources, environment), render::ComputeResourceFieldFormat::Handle},
    {1, render::ComputeResourceBindingKind::StorageBuffer, offsetof(FrameEnvironmentProbeResources, output), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kFrameEnvironmentProbeLayout{sizeof(FrameEnvironmentProbeResources), kFrameEnvironmentProbeFields};

inline constexpr render::ComputeResourceField kFrameCopyProbeFields[] = {
    {0, render::ComputeResourceBindingKind::StorageBuffer, offsetof(FrameCopyProbeResources, input), render::ComputeResourceFieldFormat::Handle},
    {1, render::ComputeResourceBindingKind::StorageBuffer, offsetof(FrameCopyProbeResources, output), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kFrameCopyProbeLayout{sizeof(FrameCopyProbeResources), kFrameCopyProbeFields};

inline constexpr render::ComputeResourceField kFrameHistoryProbeFields[] = {
    {0, render::ComputeResourceBindingKind::StorageImage, offsetof(FrameHistoryProbeResources, previous), render::ComputeResourceFieldFormat::Handle},
    {1, render::ComputeResourceBindingKind::StorageImage, offsetof(FrameHistoryProbeResources, current), render::ComputeResourceFieldFormat::Handle},
    {2, render::ComputeResourceBindingKind::StorageBuffer, offsetof(FrameHistoryProbeResources, output), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kFrameHistoryProbeLayout{sizeof(FrameHistoryProbeResources), kFrameHistoryProbeFields};

inline constexpr render::ComputeResourceField kFrameImagesProbeFields[] = {
    {0, render::ComputeResourceBindingKind::SampledImage, offsetof(FrameImagesProbeResources, images), render::ComputeResourceFieldFormat::IndexSpan},
    {1, render::ComputeResourceBindingKind::StorageBuffer, offsetof(FrameImagesProbeResources, output), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kFrameImagesProbeLayout{sizeof(FrameImagesProbeResources), kFrameImagesProbeFields};

inline constexpr render::ComputeResourceField kGPUDrivenConeProbeFields[] = {
    {0, render::ComputeResourceBindingKind::StorageBuffer, offsetof(GPUDrivenConeProbeResources, output), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kGPUDrivenConeProbeLayout{sizeof(GPUDrivenConeProbeResources), kGPUDrivenConeProbeFields};

inline constexpr render::ComputeResourceField kHZBSPDFixtureFields[] = {
    {0, render::ComputeResourceBindingKind::StorageImage, offsetof(HZBSPDFixtureResources, depth), render::ComputeResourceFieldFormat::Handle},
    {1, render::ComputeResourceBindingKind::StorageBuffer, offsetof(HZBSPDFixtureResources, counter), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kHZBSPDFixtureLayout{sizeof(HZBSPDFixtureResources), kHZBSPDFixtureFields};

inline constexpr render::ComputeResourceField kMaterialBinningProbeFields[] = {
    {0, render::ComputeResourceBindingKind::StorageImage, offsetof(MaterialBinningProbeResources, visibility), render::ComputeResourceFieldFormat::Handle},
    {1, render::ComputeResourceBindingKind::StorageBuffer, offsetof(MaterialBinningProbeResources, records), render::ComputeResourceFieldFormat::Handle},
    {2, render::ComputeResourceBindingKind::StorageBuffer, offsetof(MaterialBinningProbeResources, instances), render::ComputeResourceFieldFormat::Handle},
    {3, render::ComputeResourceBindingKind::StorageBuffer, offsetof(MaterialBinningProbeResources, materials), render::ComputeResourceFieldFormat::Handle},
    {4, render::ComputeResourceBindingKind::StorageBuffer, offsetof(MaterialBinningProbeResources, shadingMaterials), render::ComputeResourceFieldFormat::Handle},
    {5, render::ComputeResourceBindingKind::StorageBuffer, offsetof(MaterialBinningProbeResources, bins), render::ComputeResourceFieldFormat::Handle},
    {6, render::ComputeResourceBindingKind::StorageBuffer, offsetof(MaterialBinningProbeResources, tiles), render::ComputeResourceFieldFormat::Handle},
    {7, render::ComputeResourceBindingKind::StorageBuffer, offsetof(MaterialBinningProbeResources, arguments), render::ComputeResourceFieldFormat::Handle},
    {8, render::ComputeResourceBindingKind::StorageBuffer, offsetof(MaterialBinningProbeResources, output), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kMaterialBinningProbeLayout{sizeof(MaterialBinningProbeResources), kMaterialBinningProbeFields};

inline constexpr render::ComputeResourceField kMaterialRuntimeProbeFields[] = {
    {0, render::ComputeResourceBindingKind::StorageBuffer, offsetof(MaterialRuntimeProbeResources, materials), render::ComputeResourceFieldFormat::Handle},
    {1, render::ComputeResourceBindingKind::StorageBuffer, offsetof(MaterialRuntimeProbeResources, output), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kMaterialRuntimeProbeLayout{sizeof(MaterialRuntimeProbeResources), kMaterialRuntimeProbeFields};

inline constexpr render::ComputeResourceField kNativeDescriptorHandlesFields[] = {
    {0, render::ComputeResourceBindingKind::StorageBuffer, offsetof(NativeDescriptorHandlesResources, records), render::ComputeResourceFieldFormat::Handle},
    {1, render::ComputeResourceBindingKind::StorageBuffer, offsetof(NativeDescriptorHandlesResources, output), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kNativeDescriptorHandlesLayout{sizeof(NativeDescriptorHandlesResources), kNativeDescriptorHandlesFields};

inline constexpr render::ComputeResourceField kPhotometricProbeFields[] = {
    {0, render::ComputeResourceBindingKind::StorageBuffer, offsetof(PhotometricProbeResources, output), render::ComputeResourceFieldFormat::Handle},
    {1, render::ComputeResourceBindingKind::StorageBuffer, offsetof(PhotometricProbeResources, irradiance), render::ComputeResourceFieldFormat::Handle},
    {50, render::ComputeResourceBindingKind::StorageBuffer, offsetof(PhotometricProbeResources, lights), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kPhotometricProbeLayout{sizeof(PhotometricProbeResources), kPhotometricProbeFields};

inline constexpr render::ComputeResourceField kRealtimeGuideProbeFields[] = {
    {0, render::ComputeResourceBindingKind::SampledImage, offsetof(RealtimeGuideProbeResources, motion), render::ComputeResourceFieldFormat::Handle},
    {1, render::ComputeResourceBindingKind::SampledImage, offsetof(RealtimeGuideProbeResources, depth), render::ComputeResourceFieldFormat::Handle},
    {2, render::ComputeResourceBindingKind::StorageBuffer, offsetof(RealtimeGuideProbeResources, data), render::ComputeResourceFieldFormat::Handle},
    {3, render::ComputeResourceBindingKind::StorageBuffer, offsetof(RealtimeGuideProbeResources, prefilter), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kRealtimeGuideProbeLayout{sizeof(RealtimeGuideProbeResources), kRealtimeGuideProbeFields};

inline constexpr render::ComputeResourceField kReGIRVirtualLightProbeFields[] = {
    {0, render::ComputeResourceBindingKind::StorageBuffer, offsetof(ReGIRVirtualLightProbeResources, output), render::ComputeResourceFieldFormat::Handle},
    {50, render::ComputeResourceBindingKind::StorageBuffer, offsetof(ReGIRVirtualLightProbeResources, lights), render::ComputeResourceFieldFormat::Handle},
    {52, render::ComputeResourceBindingKind::StorageBuffer, offsetof(ReGIRVirtualLightProbeResources, lightAlias), render::ComputeResourceFieldFormat::Handle},
    {53, render::ComputeResourceBindingKind::SampledImage, offsetof(ReGIRVirtualLightProbeResources, lightsPdf), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kReGIRVirtualLightProbeLayout{sizeof(ReGIRVirtualLightProbeResources), kReGIRVirtualLightProbeFields};

inline constexpr render::ComputeResourceField kSceneUploadProbeFields[] = {
    {0, render::ComputeResourceBindingKind::SampledImage, offsetof(SceneUploadProbeResources, texture), render::ComputeResourceFieldFormat::Handle},
    {1, render::ComputeResourceBindingKind::StorageBuffer, offsetof(SceneUploadProbeResources, output), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kSceneUploadProbeLayout{sizeof(SceneUploadProbeResources), kSceneUploadProbeFields};

inline constexpr render::ComputeResourceField kSliderDebugFixtureFields[] = {
    {0, render::ComputeResourceBindingKind::StorageImage, offsetof(SliderDebugFixtureResources, output), render::ComputeResourceFieldFormat::Handle},
    {1, render::ComputeResourceBindingKind::SampledImage, offsetof(SliderDebugFixtureResources, input), render::ComputeResourceFieldFormat::Handle},
    {2, render::ComputeResourceBindingKind::StorageBuffer, offsetof(SliderDebugFixtureResources, readback), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kSliderDebugFixtureLayout{sizeof(SliderDebugFixtureResources), kSliderDebugFixtureFields};

inline constexpr render::ComputeResourceField kSphericalHarmonicsProbeFields[] = {
    {0, render::ComputeResourceBindingKind::StorageBuffer, offsetof(SphericalHarmonicsProbeResources, output), render::ComputeResourceFieldFormat::Handle},
    {1, render::ComputeResourceBindingKind::StorageBuffer, offsetof(SphericalHarmonicsProbeResources, input), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kSphericalHarmonicsProbeLayout{sizeof(SphericalHarmonicsProbeResources), kSphericalHarmonicsProbeFields};

inline constexpr render::ComputeResourceField kTextureFootprintProbeFields[] = {
    {0, render::ComputeResourceBindingKind::StorageBuffer, offsetof(TextureFootprintProbeResources, output), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kTextureFootprintProbeLayout{sizeof(TextureFootprintProbeResources), kTextureFootprintProbeFields};

inline constexpr render::ComputeResourceField kTextureStreamingProbeFields[] = {
    {0, render::ComputeResourceBindingKind::SampledImage, offsetof(TextureStreamingProbeResources, textures), render::ComputeResourceFieldFormat::IndexSpan},
    {1, render::ComputeResourceBindingKind::StorageBuffer, offsetof(TextureStreamingProbeResources, feedback), render::ComputeResourceFieldFormat::Handle},
    {2, render::ComputeResourceBindingKind::StorageBuffer, offsetof(TextureStreamingProbeResources, output), render::ComputeResourceFieldFormat::Handle},
    {3, render::ComputeResourceBindingKind::Sampler, offsetof(TextureStreamingProbeResources, sampler), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kTextureStreamingProbeLayout{sizeof(TextureStreamingProbeResources), kTextureStreamingProbeFields};

inline constexpr render::ComputeResourceField kTwoPassOcclusionProbeFields[] = {
    {0, render::ComputeResourceBindingKind::StorageBuffer, offsetof(TwoPassOcclusionProbeResources, output), render::ComputeResourceFieldFormat::Handle},
    {1, render::ComputeResourceBindingKind::StorageBuffer, offsetof(TwoPassOcclusionProbeResources, previousHzb), render::ComputeResourceFieldFormat::Handle},
    {2, render::ComputeResourceBindingKind::StorageBuffer, offsetof(TwoPassOcclusionProbeResources, currentHzb), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kTwoPassOcclusionProbeLayout{sizeof(TwoPassOcclusionProbeResources), kTwoPassOcclusionProbeFields};

inline constexpr render::ComputeResourceField kUnifiedTopLevelProbeFields[] = {
    {0, render::ComputeResourceBindingKind::AccelerationStructure, offsetof(UnifiedTopLevelProbeResources, scene), render::ComputeResourceFieldFormat::Handle},
    {1, render::ComputeResourceBindingKind::StorageBuffer, offsetof(UnifiedTopLevelProbeResources, output), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kUnifiedTopLevelProbeLayout{sizeof(UnifiedTopLevelProbeResources), kUnifiedTopLevelProbeFields};

inline constexpr render::ComputeResourceField kViewConstantsProbeFields[] = {
    {0, render::ComputeResourceBindingKind::StorageBuffer, offsetof(ViewConstantsProbeResources, output), render::ComputeResourceFieldFormat::Handle},
    {1, render::ComputeResourceBindingKind::StorageBuffer, offsetof(ViewConstantsProbeResources, input), render::ComputeResourceFieldFormat::Handle},
};
inline constexpr render::ComputeResourceLayout kViewConstantsProbeLayout{sizeof(ViewConstantsProbeResources), kViewConstantsProbeFields};

} // namespace metallic::tests
