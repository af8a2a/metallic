#pragma once

#ifdef _WIN32

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

// Forward declarations
struct VmaAllocator_T;
typedef VmaAllocator_T* VmaAllocator;
struct VmaAllocation_T;
typedef VmaAllocation_T* VmaAllocation;

class RhiContext;

// -------------------------------------------------------------------------
// Shader group descriptor — describes one entry in the SBT.
// -------------------------------------------------------------------------

enum class RtShaderGroupType : uint8_t {
    RayGen,
    Miss,
    TrianglesHit,    // closest-hit + optional any-hit for triangle geometry
    ProceduralHit,   // closest-hit + any-hit + intersection for procedural geometry
};

struct RtShaderGroupDesc {
    RtShaderGroupType type = RtShaderGroupType::RayGen;
    std::string closestHitEntry;      // required for hit groups
    std::string anyHitEntry;          // optional
    std::string intersectionEntry;    // required for ProceduralHit only
};

// -------------------------------------------------------------------------
// Ray tracing pipeline descriptor.
// -------------------------------------------------------------------------

struct RtPipelineDesc {
    // SPIR-V binary containing all RT shader stages.
    const void* shaderCode = nullptr;
    size_t shaderCodeSize  = 0;

    // Entry point for the ray generation shader (exactly one required).
    std::string rayGenEntry = "raygenMain";

    // Miss shader groups.
    std::vector<std::string> missEntries;

    // Hit groups.
    std::vector<RtShaderGroupDesc> hitGroups;

    // Callable shader entries (optional, rarely used).
    std::vector<std::string> callableEntries;

    uint32_t maxRecursionDepth = 1;
};

// -------------------------------------------------------------------------
// VulkanRayTracingPipeline — owns VkPipeline, SBT buffer, and SBT regions.
// -------------------------------------------------------------------------

class VulkanRayTracingPipeline {
public:
    VulkanRayTracingPipeline() = default;
    ~VulkanRayTracingPipeline();

    // Non-copyable, movable
    VulkanRayTracingPipeline(const VulkanRayTracingPipeline&) = delete;
    VulkanRayTracingPipeline& operator=(const VulkanRayTracingPipeline&) = delete;
    VulkanRayTracingPipeline(VulkanRayTracingPipeline&& other) noexcept;
    VulkanRayTracingPipeline& operator=(VulkanRayTracingPipeline&& other) noexcept;

    VkPipeline pipeline() const { return m_pipeline; }
    VkPipelineLayout layout() const { return m_layout; }

    const VkStridedDeviceAddressRegionKHR& raygenRegion()   const { return m_raygenRegion; }
    const VkStridedDeviceAddressRegionKHR& missRegion()     const { return m_missRegion; }
    const VkStridedDeviceAddressRegionKHR& hitRegion()      const { return m_hitRegion; }
    const VkStridedDeviceAddressRegionKHR& callableRegion() const { return m_callableRegion; }

    bool isValid() const { return m_pipeline != VK_NULL_HANDLE; }

private:
    friend std::unique_ptr<VulkanRayTracingPipeline>
        createVulkanRayTracingPipelineImpl(VkDevice device,
                                           VkPhysicalDevice physicalDevice,
                                           VmaAllocator allocator,
                                           VkPipelineCache pipelineCache,
                                           const RtPipelineDesc& desc,
                                           uint32_t shaderGroupHandleSize,
                                           uint32_t shaderGroupHandleAlignment,
                                           uint32_t shaderGroupBaseAlignment,
                                           std::string& errorMessage,
                                           bool useDescriptorBuffer);

    VkDevice     m_device    = VK_NULL_HANDLE;
    VkPipeline   m_pipeline  = VK_NULL_HANDLE;
    VkPipelineLayout m_layout = VK_NULL_HANDLE; // owned by this object, destroyed in destructor
    VkBuffer     m_sbtBuffer = VK_NULL_HANDLE;
    VmaAllocation m_sbtAllocation = nullptr;
    VmaAllocator  m_allocator = nullptr;

    VkStridedDeviceAddressRegionKHR m_raygenRegion{};
    VkStridedDeviceAddressRegionKHR m_missRegion{};
    VkStridedDeviceAddressRegionKHR m_hitRegion{};
    VkStridedDeviceAddressRegionKHR m_callableRegion{};
};

// -------------------------------------------------------------------------
// Public API — called from renderer code via vulkan_backend.h declarations.
// -------------------------------------------------------------------------

// Create a ray tracing pipeline + SBT from the given descriptor.
// Returns nullptr on failure (errorMessage populated).
std::unique_ptr<VulkanRayTracingPipeline> createVulkanRayTracingPipeline(
    RhiContext& context,
    const RtPipelineDesc& desc,
    std::string& errorMessage);

// Dispatch vkCmdTraceRaysKHR using the pipeline's SBT regions.
void vulkanTraceRays(RhiContext& context,
                     const VulkanRayTracingPipeline& pipeline,
                     uint32_t width,
                     uint32_t height,
                     uint32_t depth = 1);

#endif // _WIN32
