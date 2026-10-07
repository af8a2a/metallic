#include "Runtime/Render/GAPI/ShaderTarget.h"

#include <volk.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace metallic::render {
namespace {

class VulkanQueryLoader {
public:
    VulkanQueryLoader()
    {
#ifdef _WIN32
        library_ = LoadLibraryW(L"vulkan-1.dll");
        if (library_) {
            getInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
                GetProcAddress(library_, "vkGetInstanceProcAddr"));
        }
#else
        library_ = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
        if (library_) {
            getInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
                dlsym(library_, "vkGetInstanceProcAddr"));
        }
#endif
    }

    ~VulkanQueryLoader()
    {
#ifdef _WIN32
        if (library_) { FreeLibrary(library_); }
#else
        if (library_) { dlclose(library_); }
#endif
    }

    VulkanQueryLoader(const VulkanQueryLoader&) = delete;
    VulkanQueryLoader& operator=(const VulkanQueryLoader&) = delete;
    PFN_vkGetInstanceProcAddr getInstanceProcAddr = nullptr;

private:
#ifdef _WIN32
    HMODULE library_ = nullptr;
#else
    void* library_ = nullptr;
#endif
};

struct QueryInstance {
    VkInstance handle = VK_NULL_HANDLE;
    PFN_vkDestroyInstance destroy = nullptr;
    ~QueryInstance()
    {
        if (handle && destroy) { destroy(handle, nullptr); }
    }
};

bool querySucceeded(VkResult result, const char* operation, std::string& diagnostics)
{
    if (result == VK_SUCCESS) { return true; }
    diagnostics = std::string("Vulkan descriptor heap stride query: ") + operation +
        " failed with VkResult " + std::to_string(static_cast<int>(result));
    return false;
}

bool descriptorHeapStrides(const VkPhysicalDeviceDescriptorHeapPropertiesEXT& properties,
    DescriptorHeapShaderStrides& strides, std::string& diagnostics)
{
    if (!properties.imageDescriptorSize || !properties.bufferDescriptorSize ||
        !properties.samplerDescriptorSize || !properties.maxPushDataSize ||
        !properties.imageDescriptorAlignment || !properties.bufferDescriptorAlignment ||
        !properties.samplerDescriptorAlignment ||
        properties.samplerDescriptorSize % properties.samplerDescriptorAlignment) {
        diagnostics = "Vulkan descriptor heap stride query: invalid descriptor sizes or alignments";
        return false;
    }
    const auto alignedSize = [](VkDeviceSize size, VkDeviceSize alignment, VkDeviceSize& result) {
        if (size > std::numeric_limits<VkDeviceSize>::max() - (alignment - 1)) { return false; }
        result = (size + alignment - 1) / alignment * alignment;
        return true;
    };
    VkDeviceSize imageSize = 0, bufferSize = 0;
    if (!alignedSize(properties.imageDescriptorSize, properties.imageDescriptorAlignment, imageSize) ||
        !alignedSize(properties.bufferDescriptorSize, properties.bufferDescriptorAlignment, bufferSize)) {
        diagnostics = "Vulkan descriptor heap stride query: descriptor alignment overflows";
        return false;
    }
    const VkDeviceSize resourceStride = std::max(imageSize, bufferSize);
    if (resourceStride % properties.imageDescriptorAlignment ||
        resourceStride % properties.bufferDescriptorAlignment ||
        resourceStride > std::numeric_limits<int32_t>::max() ||
        properties.samplerDescriptorSize > std::numeric_limits<int32_t>::max()) {
        diagnostics = "Vulkan descriptor heap stride query: descriptor strides cannot be represented by Slang";
        return false;
    }
    strides = {static_cast<uint32_t>(resourceStride), static_cast<uint32_t>(properties.samplerDescriptorSize)};
    return true;
}

} // namespace

bool queryVulkanDescriptorHeapShaderStrides(DescriptorHeapShaderStrides& out,
    std::string& diagnostics)
{
    // TO-REMOVE(VVL payload-size): native OpConstantSizeOfEXT currently leaves
    // unrelated task/mesh payload sizes unresolved in VVL, including 1.4.363.
    // Query only instance/physical-device state for literal Slang heap strides;
    // retain local dispatch so startup warmup cannot replace the RHI's loader.
    diagnostics.clear();
    VulkanQueryLoader loader;
    if (!loader.getInstanceProcAddr) {
        diagnostics = "Vulkan descriptor heap stride query: Vulkan loader is unavailable";
        return false;
    }
    const auto enumerateVersion = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
        loader.getInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
    const auto createInstance = reinterpret_cast<PFN_vkCreateInstance>(
        loader.getInstanceProcAddr(VK_NULL_HANDLE, "vkCreateInstance"));
    uint32_t version = VK_API_VERSION_1_0;
    if (!enumerateVersion || !createInstance ||
        !querySucceeded(enumerateVersion(&version), "vkEnumerateInstanceVersion", diagnostics)) {
        if (diagnostics.empty()) { diagnostics = "Vulkan descriptor heap stride query: required loader entry points unavailable"; }
        return false;
    }
    if (version < VK_API_VERSION_1_4) {
        diagnostics = "Vulkan descriptor heap stride query: Vulkan 1.4 is required";
        return false;
    }
    const VkApplicationInfo application{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Metallic descriptor heap shader strides",
        .apiVersion = VK_API_VERSION_1_4,
    };
    const VkInstanceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &application,
    };
    QueryInstance instance;
    if (!querySucceeded(createInstance(&createInfo, nullptr, &instance.handle), "vkCreateInstance", diagnostics)) {
        return false;
    }
    instance.destroy = reinterpret_cast<PFN_vkDestroyInstance>(
        loader.getInstanceProcAddr(instance.handle, "vkDestroyInstance"));
    const auto enumerateDevices = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
        loader.getInstanceProcAddr(instance.handle, "vkEnumeratePhysicalDevices"));
    const auto enumerateExtensions = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
        loader.getInstanceProcAddr(instance.handle, "vkEnumerateDeviceExtensionProperties"));
    const auto queryFeatures = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
        loader.getInstanceProcAddr(instance.handle, "vkGetPhysicalDeviceFeatures2"));
    const auto queryProperties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
        loader.getInstanceProcAddr(instance.handle, "vkGetPhysicalDeviceProperties2"));
    if (!instance.destroy || !enumerateDevices || !enumerateExtensions || !queryFeatures || !queryProperties) {
        diagnostics = "Vulkan descriptor heap stride query: required instance entry points unavailable";
        return false;
    }
    uint32_t deviceCount = 0;
    if (!querySucceeded(enumerateDevices(instance.handle, &deviceCount, nullptr), "vkEnumeratePhysicalDevices", diagnostics)) {
        return false;
    }
    if (!deviceCount) {
        diagnostics = "Vulkan descriptor heap stride query: no adapter supports descriptor heaps and untyped pointers";
        return false;
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    if (!querySucceeded(enumerateDevices(instance.handle, &deviceCount, devices.data()), "vkEnumeratePhysicalDevices", diagnostics)) {
        return false;
    }
    devices.resize(deviceCount);
    for (VkPhysicalDevice device : devices) {
        uint32_t extensionCount = 0;
        if (!querySucceeded(enumerateExtensions(device, nullptr, &extensionCount, nullptr), "vkEnumerateDeviceExtensionProperties", diagnostics)) {
            return false;
        }
        std::vector<VkExtensionProperties> extensions(extensionCount);
        if (!querySucceeded(enumerateExtensions(device, nullptr, &extensionCount, extensions.data()), "vkEnumerateDeviceExtensionProperties", diagnostics)) {
            return false;
        }
        extensions.resize(extensionCount);
        const auto hasExtension = [&extensions](const char* name) {
            return std::ranges::any_of(extensions, [name](const auto& extension) {
                return std::strcmp(extension.extensionName, name) == 0;
            });
        };
        if (!hasExtension(VK_EXT_DESCRIPTOR_HEAP_EXTENSION_NAME) ||
            !hasExtension(VK_KHR_SHADER_UNTYPED_POINTERS_EXTENSION_NAME)) {
            continue;
        }
        VkPhysicalDeviceShaderUntypedPointersFeaturesKHR untyped{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_UNTYPED_POINTERS_FEATURES_KHR,
        };
        VkPhysicalDeviceDescriptorHeapFeaturesEXT heap{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_FEATURES_EXT,
            .pNext = &untyped,
        };
        VkPhysicalDeviceFeatures2 features{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
            .pNext = &heap,
        };
        queryFeatures(device, &features);
        if (!heap.descriptorHeap || !untyped.shaderUntypedPointers) { continue; }
        VkPhysicalDeviceDescriptorHeapPropertiesEXT heapProperties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_HEAP_PROPERTIES_EXT,
        };
        VkPhysicalDeviceProperties2 properties{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &heapProperties,
        };
        queryProperties(device, &properties);
        if (properties.properties.apiVersion < VK_API_VERSION_1_4) { continue; }
        DescriptorHeapShaderStrides strides;
        if (!descriptorHeapStrides(heapProperties, strides, diagnostics)) { return false; }
        out = strides;
        diagnostics.clear();
        return true;
    }
    diagnostics = "Vulkan descriptor heap stride query: no adapter supports descriptor heaps and untyped pointers";
    return false;
}

} // namespace metallic::render
