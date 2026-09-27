#include "VulkanDiagnostics.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include <iomanip>
#include <sstream>

namespace metallic::tests::bench {

bool nativeDescriptorPointersEnabled(render::Device& device)
{
    return render::vulkan::nativeDevice(device).shaderUntypedPointersEnabled;
}

Validation activeValidation(render::Device& device)
{
    const auto native = render::vulkan::nativeDevice(device);
    if (!native.validationEnabled || !native.validationMessengerActive) { return Validation::Off; }
    return native.synchronizationValidationEnabled ? Validation::Synchronization : Validation::Core;
}

Json describeDevice(render::Device& device, const Profile& profile)
{
    const auto native = render::vulkan::nativeDevice(device);
    VkPhysicalDeviceIDProperties ids{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceDriverProperties driver{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, &ids};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &driver};
    vkGetPhysicalDeviceProperties2(native.physicalDevice, &properties);
    std::ostringstream uuid;
    for (auto byte : ids.deviceUUID) { uuid << std::hex << std::setw(2) << std::setfill('0') << uint32_t(byte); }
    Json queues = Json::array();
    for (const auto type : {render::QueueType::Graphics, render::QueueType::Compute, render::QueueType::Copy}) {
        if (auto* queue = device.getQueue(type)) {
            const auto info = render::vulkan::nativeQueue(*queue);
            queues.push_back({{"type", int(type)}, {"family", info.familyIndex},
                {"sameAsGraphics", queue->sameQueue(*device.getQueue(render::QueueType::Graphics))},
                {"timestampValidBits", queue->timestampValidBits()}});
        }
    }
    Json capabilities = Json::array();
    for (const auto capability : {Capability::ShaderObject, Capability::TimestampQueries, Capability::Bindless,
        Capability::IndependentCopy, Capability::IndependentCompute, Capability::RayQuery, Capability::PositionFetch,
        Capability::OpacityMicromap, Capability::UnifiedLayouts, Capability::PartitionedAS, Capability::ClusterAS,
        Capability::GeneratedCommands, Capability::MemoryDecompression}) {
        const bool usable = enabled(capability, device.capabilities());
        capabilities.push_back({{"id", name(capability)}, {"requested", requested(capability, profile)},
            {"enabled", usable}, {"physicalSupport", usable ? "True" : "Unknown"}});
    }
    Json layers = Json::array();
    uint32_t count = 0;
    if (vkEnumerateInstanceLayerProperties(&count, nullptr) != VK_SUCCESS) { throw std::runtime_error("layer enumeration failed"); }
    std::vector<VkLayerProperties> available(count);
    if (vkEnumerateInstanceLayerProperties(&count, available.data()) != VK_SUCCESS) { throw std::runtime_error("layer enumeration failed"); }
    for (uint32_t i = 0; i < count; ++i) {
        layers.push_back({{"name", available[i].layerName}, {"specVersion", available[i].specVersion},
            {"implementationVersion", available[i].implementationVersion}});
    }
    return {{"device", properties.properties.deviceName}, {"uuid", uuid.str()}, {"apiVersion", properties.properties.apiVersion},
        {"driver", driver.driverName}, {"driverInfo", driver.driverInfo}, {"driverVersion", properties.properties.driverVersion},
        {"nativeDescriptorPointersEnabled", native.shaderUntypedPointersEnabled}, {"validationEnabled", native.validationEnabled}, {"messengerActive", native.validationMessengerActive},
        {"validationMode", name(activeValidation(device))}, {"availableLayers", layers},
        {"queues", queues}, {"capabilities", capabilities}};
}

} // namespace metallic::tests::bench
