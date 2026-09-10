#include "Runtime/Render/GAPI/Vulkan/VulkanDlssNr.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include "Runtime/Render/GAPI/Vulkan/VulkanStreamline.h"

#include <spdlog/spdlog.h>

#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <sstream>

#if METALLIC_HAS_DLSS_NR
#include <Windows.h>
#include <nvsdk_ngx_vk.h>
#endif

namespace metallic::render::vulkan {

Result validateDlssNrDesc(const DlssNrDesc& desc, std::string& log)
{
    log.clear();
    const std::array refs{desc.inputColor, desc.outputColor, desc.motionVectors, desc.depth};
    const std::array formats{Format::Rgba16Sfloat, Format::Rgba16Sfloat, Format::Rg16Sfloat, Format::R32Sfloat};
    for (size_t i = 0; i < refs.size(); ++i) {
        if (refs[i].texture == nullptr || refs[i].view == nullptr) {
            log = "DLSS-NR requires color, output, motion vectors and depth with valid views";
            return makeError(Error::InvalidArgument);
        }
        const auto& texture = refs[i].texture->desc();
        const bool colorFormat = i < 2 && texture.format == Format::Rgba8Unorm;
        if ((!colorFormat && texture.format != formats[i]) || texture.width == 0 || texture.height == 0 ||
            texture.width > INT32_MAX || texture.height > INT32_MAX || texture.depth != 1 ||
            texture.type != TextureType::Texture2D ||
            texture.mipCount != 1 || texture.layerCount != 1 ||
            !hasFlag(texture.usage, TextureUsageBits::Storage) ||
            (i != 1 && !hasFlag(texture.usage, TextureUsageBits::Sampled))) {
            log = "DLSS-NR requires single-mip storage textures and sampled inputs: RGBA8/RGBA16F display color, RG16F motion, R32F depth";
            return makeError(Error::InvalidArgument);
        }
        for (size_t j = 0; j < i; ++j) {
            if (refs[i].texture == refs[j].texture) {
                log = "DLSS-NR requires four distinct textures";
                return makeError(Error::InvalidArgument);
            }
        }
    }
    const auto& input = desc.inputColor.texture->desc();
    const auto& output = desc.outputColor.texture->desc();
    if (input.format != output.format) {
        log = "DLSS-NR input and output color formats must match";
        return makeError(Error::InvalidArgument);
    }
    for (const auto& guide : {desc.motionVectors, desc.depth}) {
        if (guide.texture->desc().width != input.width || guide.texture->desc().height != input.height) {
            log = "DLSS-NR motion vectors and depth must match the input color extent";
            return makeError(Error::InvalidArgument);
        }
    }
    const uint64_t scale = desc.settings.upscaling ? 2 : 1;
    if (output.width != input.width * scale || output.height != input.height * scale) {
        log = "DLSS-NR requires matching extents in native mode or exactly 2x output in upscaling mode";
        return makeError(Error::InvalidArgument);
    }
    const auto& s = desc.settings;
    if (s.preset > 3 || s.style > 2 || !std::isfinite(s.motionVectorScaleX) ||
        !std::isfinite(s.motionVectorScaleY) || !std::isfinite(s.intensity) ||
        !std::isfinite(s.localToneStrength) || !std::isfinite(s.localStructureStrength) ||
        !std::isfinite(s.skinStructureStrength) || s.intensity < 0 || s.intensity > 1 ||
        s.localToneStrength < 0 || s.localToneStrength > 1 ||
        s.localStructureStrength < 0 || s.localStructureStrength > 1 ||
        s.skinStructureStrength < -1 || s.skinStructureStrength > 1) {
        log = "DLSS-NR settings contain an invalid preset, style, strength or motion-vector scale";
        return makeError(Error::InvalidArgument);
    }
    return {};
}

bool dlssNrSdkAvailable()
{
    return METALLIC_HAS_DLSS_NR != 0;
}

#if METALLIC_HAS_DLSS_NR
namespace {

// Recovered contract from Unity-DLSS-RR/src/DLSSNRRuntime.cpp. This is not a
// public Streamline feature. Keep all snippet-specific behavior in this file.
constexpr auto kFeature = static_cast<NVSDK_NGX_Feature>(18);
constexpr auto kApiVersion = static_cast<NVSDK_NGX_Version>(0x15);
constexpr unsigned long long kApplicationId = 0x0876232Cull;

using InitFn = NVSDK_NGX_Result(NVSDK_CONV*)(unsigned long long, const wchar_t*,
    VkInstance, VkPhysicalDevice, VkDevice, PFN_vkGetInstanceProcAddr,
    PFN_vkGetDeviceProcAddr, NVSDK_NGX_Version, const NVSDK_NGX_Parameter*);
using CreateFn = NVSDK_NGX_Result(NVSDK_CONV*)(VkDevice, VkCommandBuffer,
    NVSDK_NGX_Feature, const NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
using EvaluateFn = NVSDK_NGX_Result(NVSDK_CONV*)(VkCommandBuffer,
    const NVSDK_NGX_Handle*, const NVSDK_NGX_Parameter*, PFN_NVSDK_NGX_ProgressCallback);
using ReleaseFn = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Handle*);
using ShutdownFn = NVSDK_NGX_Result(NVSDK_CONV*)(VkDevice);
using AllocateParametersFn = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter**);
using DestroyParametersFn = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter*);
using ModuleNameFn = DWORD(WINAPI*)(HMODULE, LPWSTR, DWORD);

std::mutex runtimeMutex;

DWORD WINAPI snippetCallerModuleName(HMODULE module, LPWSTR name, DWORD size)
{
    const HMODULE snippet = GetModuleHandleW(L"nvngx_dlssnr.dll");
    HMODULE core = GetModuleHandleW(L"_nvngx.dll");
    if (core == nullptr) { core = GetModuleHandleW(L"nvngx.dll"); }
    // This is the EXE's unmodified import. No temporary callback state is shared
    // with worker threads that may enter while the snippet API is running.
    return GetModuleFileNameW(module == snippet || core == nullptr ? module : core, name, size);
}

// Match the reference's direct-snippet caller compatibility shim. It affects
// only this DLL's import, only during snippet API calls, and never patches the
// DLL file. This runtime also checks the caller on create/evaluate/shutdown.
class ScopedSnippetCaller {
public:
    explicit ScopedSnippetCaller(HMODULE module)
    {
        HMODULE coreModule = GetModuleHandleW(L"_nvngx.dll");
        if (coreModule == nullptr) { coreModule = GetModuleHandleW(L"nvngx.dll"); }
        if (coreModule == nullptr) { return; }
        auto* base = reinterpret_cast<BYTE*>(module);
        auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(base);
        if (dos->e_magic != IMAGE_DOS_SIGNATURE) { return; }
        auto* nt = reinterpret_cast<IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
        if (nt->Signature != IMAGE_NT_SIGNATURE) { return; }
        const auto& directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
        if (directory.VirtualAddress == 0) { return; }
        auto* imports = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base + directory.VirtualAddress);
        for (; imports->Name != 0; ++imports) {
            const char* dllName = reinterpret_cast<const char*>(base + imports->Name);
            if (_stricmp(dllName, "KERNEL32.dll") != 0 && _stricmp(dllName, "KERNELBASE.dll") != 0) { continue; }
            if (imports->OriginalFirstThunk == 0) { continue; }
            auto* names = reinterpret_cast<IMAGE_THUNK_DATA*>(base + imports->OriginalFirstThunk);
            auto* slots = reinterpret_cast<IMAGE_THUNK_DATA*>(base + imports->FirstThunk);
            for (size_t i = 0; slots[i].u1.Function != 0; ++i) {
                if (IMAGE_SNAP_BY_ORDINAL(names[i].u1.Ordinal)) { continue; }
                auto* import = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(base + names[i].u1.AddressOfData);
                if (std::strcmp(reinterpret_cast<const char*>(import->Name), "GetModuleFileNameW") != 0) { continue; }
                auto** slot = reinterpret_cast<void**>(&slots[i].u1.Function);
                if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &protection_)) { return; }
                slot_ = slot;
                originalModuleName_ = reinterpret_cast<ModuleNameFn>(*slot);
                InterlockedExchangePointer(slot_, reinterpret_cast<void*>(&snippetCallerModuleName));
                return;
            }
        }
    }

    ~ScopedSnippetCaller()
    {
        if (slot_ != nullptr) {
            InterlockedExchangePointer(slot_, reinterpret_cast<void*>(originalModuleName_));
            DWORD ignored = 0;
            VirtualProtect(slot_, sizeof(void*), protection_, &ignored);
        }
    }

    bool valid() const { return slot_ != nullptr; }

private:
    void** slot_ = nullptr;
    ModuleNameFn originalModuleName_ = nullptr;
    DWORD protection_ = 0;
};

Result ngxResult(NVSDK_NGX_Result result, const char* operation, std::string& log)
{
    if (NVSDK_NGX_SUCCEED(result)) { return {}; }
    std::ostringstream message;
    message << "DLSS-NR " << operation << " failed (NGX 0x" << std::hex << static_cast<uint32_t>(result) << ')';
    log = message.str();
    return makeError(result == NVSDK_NGX_Result_FAIL_FeatureNotSupported ||
        result == NVSDK_NGX_Result_FAIL_PlatformError ? Error::Unsupported : Error::Failure);
}

struct Runtime {
    HMODULE module = nullptr;
    NativeDevice native;
    InitFn init = nullptr;
    CreateFn create = nullptr;
    EvaluateFn evaluate = nullptr;
    ReleaseFn release = nullptr;
    ShutdownFn shutdown = nullptr;
    AllocateParametersFn allocateParameters = nullptr;
    DestroyParametersFn destroyParameters = nullptr;
    bool initialized = false;

    ~Runtime()
    {
        if (initialized) {
            ScopedSnippetCaller caller(module);
            if (caller.valid()) { shutdown(native.device); }
        }
        if (module != nullptr) { FreeLibrary(module); }
    }
};

std::weak_ptr<Runtime> sharedRuntime;

NVSDK_NGX_Result NVSDK_CONV computeScalingRatio(NVSDK_NGX_Parameter* parameters)
{
    if (parameters == nullptr) { return NVSDK_NGX_Result_FAIL_InvalidParameter; }
    unsigned int upscaling = 0;
    parameters->Get("DLSSNR.Upscaling", &upscaling);
    parameters->Set("DLSSNR.ScalingRatio", upscaling ? 0.5f : 1.0f);
    return NVSDK_NGX_Result_Success;
}

void setSubrect(NVSDK_NGX_Parameter* parameters, const char* input, uint32_t width, uint32_t height)
{
    const std::string prefix = std::string("DLSSNR.") + input + "Subrect";
    parameters->Set((prefix + "BaseX").c_str(), 0);
    parameters->Set((prefix + "BaseY").c_str(), 0);
    parameters->Set((prefix + "Width").c_str(), static_cast<int>(width));
    parameters->Set((prefix + "Height").c_str(), static_cast<int>(height));
}

NVSDK_NGX_Resource_VK resourceFrom(DlssNrTextureRef ref)
{
    const auto native = nativeTexture(*ref.texture);
    NVSDK_NGX_Resource_VK resource{};
    resource.Resource.ImageViewInfo = {
        nativeImageView(*ref.view), native.image, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
        native.format, native.width, native.height};
    resource.Type = NVSDK_NGX_RESOURCE_VK_TYPE_VK_IMAGEVIEW;
    resource.ReadWrite = true;
    return resource;
}

} // namespace

struct DlssNrContext::Impl {
    std::shared_ptr<Runtime> runtime;
    Device* device = nullptr;
    NVSDK_NGX_Parameter* parameters = nullptr;
    NVSDK_NGX_Handle* handle = nullptr;
    // Retired handles remain alive until already-recorded work is submitted and
    // completed. Never release a handle immediately after recording its create.
    std::vector<NVSDK_NGX_Handle*> retired;
    std::array<uint32_t, 6> configuration{};
    DlssNrSettings settings;

    ~Impl()
    {
        if (device != nullptr) { (void)device->waitIdle(); }
        if (runtime != nullptr) {
            ScopedSnippetCaller caller(runtime->module);
            if (caller.valid()) {
                if (handle != nullptr) { runtime->release(handle); }
                for (auto* oldHandle : retired) { runtime->release(oldHandle); }
            }
        }
        if (parameters != nullptr) { runtime->destroyParameters(parameters); }
    }
};
#else
struct DlssNrContext::Impl {};
#endif

DlssNrContext::DlssNrContext() = default;
DlssNrContext::~DlssNrContext()
{
#if METALLIC_HAS_DLSS_NR
    std::scoped_lock lock(runtimeMutex);
#endif
    impl_.reset();
}

Result DlssNrContext::initialize(Device& device, std::string& log)
{
    log.clear();
#if METALLIC_HAS_DLSS_NR
    std::scoped_lock lock(runtimeMutex);
    if (impl_ != nullptr) {
        if (impl_->device == &device) { return {}; }
        log = "DLSS-NR context cannot be moved between devices";
        return makeError(Error::InvalidArgument);
    }
    if (!device.capabilities().streamline) {
        log = "DLSS-NR requires a device created with enableStreamline and supported NVIDIA Vulkan extensions";
        return makeError(Error::Unsupported);
    }
    auto runtime = sharedRuntime.lock();
    const auto native = nativeDevice(device);
    if (runtime != nullptr && runtime->native.device != native.device) {
        log = "Experimental DLSS-NR supports one Vulkan device at a time";
        return makeError(Error::Unsupported);
    }
    if (runtime == nullptr) {
        runtime = std::make_shared<Runtime>();
        runtime->native = native;
        std::array<wchar_t, 32768> executable{};
        const DWORD length = GetModuleFileNameW(nullptr, executable.data(), static_cast<DWORD>(executable.size()));
        if (length == 0 || length >= executable.size()) {
            log = "DLSS-NR could not locate the executable directory";
            return makeError(Error::Unsupported);
        }
        const auto directory = std::filesystem::path(executable.data()).parent_path();
        runtime->module = LoadLibraryExW((directory / "nvngx_dlssnr.dll").c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (runtime->module == nullptr) {
            log = "DLSS-NR could not load nvngx_dlssnr.dll next to the executable (Win32 " +
                std::to_string(GetLastError()) + ')';
            return makeError(Error::Unsupported);
        }
        runtime->init = reinterpret_cast<InitFn>(GetProcAddress(runtime->module, "NVSDK_NGX_VULKAN_Init_Ext2"));
        runtime->create = reinterpret_cast<CreateFn>(GetProcAddress(runtime->module, "NVSDK_NGX_VULKAN_CreateFeature1"));
        runtime->evaluate = reinterpret_cast<EvaluateFn>(GetProcAddress(runtime->module, "NVSDK_NGX_VULKAN_EvaluateFeature"));
        runtime->release = reinterpret_cast<ReleaseFn>(GetProcAddress(runtime->module, "NVSDK_NGX_VULKAN_ReleaseFeature"));
        runtime->shutdown = reinterpret_cast<ShutdownFn>(GetProcAddress(runtime->module, "NVSDK_NGX_VULKAN_Shutdown1"));
        if (!runtime->init || !runtime->create || !runtime->evaluate || !runtime->release || !runtime->shutdown) {
            log = "DLSS-NR runtime is missing required Vulkan exports";
            return makeError(Error::Unsupported);
        }
        // Use Streamline's already initialized NGX core. Linking another NGX
        // loader creates independent initialization/shutdown state in the EXE.
        HMODULE ngxCore = GetModuleHandleW(L"_nvngx.dll");
        if (ngxCore == nullptr) { ngxCore = GetModuleHandleW(L"nvngx.dll"); }
        if (ngxCore != nullptr) {
            runtime->allocateParameters = reinterpret_cast<AllocateParametersFn>(
                GetProcAddress(ngxCore, "NVSDK_NGX_VULKAN_AllocateParameters"));
            runtime->destroyParameters = reinterpret_cast<DestroyParametersFn>(
                GetProcAddress(ngxCore, "NVSDK_NGX_VULKAN_DestroyParameters"));
        }
        if (!runtime->allocateParameters || !runtime->destroyParameters) {
            log = "DLSS-NR requires parameter allocation exports from Streamline's loaded NGX core";
            return makeError(Error::Unsupported);
        }
        NVSDK_NGX_Result initialized;
        {
            ScopedSnippetCaller caller(runtime->module);
            if (!caller.valid()) {
                log = "DLSS-NR direct-snippet initialization requires the loaded NGX core and caller compatibility import";
                return makeError(Error::Unsupported);
            }
            initialized = runtime->init(kApplicationId, directory.c_str(), native.instance,
                native.physicalDevice, native.device, vkGetInstanceProcAddr, vkGetDeviceProcAddr, kApiVersion, nullptr);
        }
        auto result = ngxResult(initialized, "Vulkan Init_Ext2 (API 0x15)", log);
        if (!result) { return result; }
        runtime->initialized = true;
        sharedRuntime = runtime;
        spdlog::info("[DLSS-NR] Experimental Vulkan runtime initialized (feature 18, API 0x15)");
    }
    auto impl = std::make_unique<Impl>();
    impl->runtime = std::move(runtime);
    impl->device = &device;
    auto result = ngxResult(impl->runtime->allocateParameters(&impl->parameters), "AllocateParameters", log);
    if (!result) { return result; }
    if (impl->parameters == nullptr) {
        log = "DLSS-NR NGX core returned an empty parameter map";
        return makeError(Error::Failure);
    }
    impl_ = std::move(impl);
    return {};
#else
    (void)device;
    log = "DLSS-NR is disabled; configure METALLIC_ENABLE_DLSS_NR=ON and METALLIC_DLSS_NR_RUNTIME";
    return makeError(Error::Unsupported);
#endif
}

Result DlssNrContext::evaluate(CommandBuffer& commandBuffer, const DlssNrDesc& desc, std::string& log)
{
    auto valid = validateDlssNrDesc(desc, log);
    if (!valid) { return valid; }
#if METALLIC_HAS_DLSS_NR
    std::scoped_lock lock(runtimeMutex);
    if (impl_ == nullptr) {
        log = "DLSS-NR context is not initialized";
        return makeError(Error::Unsupported);
    }
    const VkCommandBuffer command = nativeCommandBuffer(commandBuffer);
    if (command == VK_NULL_HANDLE) { return makeError(Error::InvalidArgument); }
    auto& impl = *impl_;
    ScopedSnippetCaller caller(impl.runtime->module);
    if (!caller.valid()) {
        log = "DLSS-NR could not prepare the snippet caller compatibility import";
        return makeError(Error::Unsupported);
    }
    const auto& input = desc.inputColor.texture->desc();
    const auto& output = desc.outputColor.texture->desc();
    const auto& s = desc.settings;
    const std::array configuration{input.width, input.height, output.width, output.height,
        s.preset, static_cast<uint32_t>(s.upscaling)};
    auto* p = impl.parameters;
    p->Set("DLSSNR.Enabled", 1u);
    p->Set("DLSSNR.Intensity", s.intensity);
    p->Set("DLSSNR.LocalToneStrength", s.localToneStrength);
    p->Set("DLSSNR.LocalStructureStrength", s.localStructureStrength);
    p->Set("DLSSNR.SkinStructureStrength", s.skinStructureStrength);
    p->Set("DLSSNR.UseAutoMask", static_cast<unsigned int>(s.useAutoMask));
    p->Set("DLSSNR.Style", static_cast<int>(s.style));
    p->Set("DLSSNR.UICorrection", static_cast<unsigned int>(s.uiCorrection));
    // This snippet captures artistic tuning at feature creation.
    const auto& previous = impl.settings;
    const bool recreate = impl.handle == nullptr || impl.configuration != configuration ||
        previous.intensity != s.intensity || previous.localToneStrength != s.localToneStrength ||
        previous.localStructureStrength != s.localStructureStrength ||
        previous.skinStructureStrength != s.skinStructureStrength || previous.style != s.style ||
        previous.useAutoMask != s.useAutoMask || previous.uiCorrection != s.uiCorrection;
    if (recreate) {
        if (impl.handle != nullptr) {
            // The graph serializes evaluations. A wait is needed only on
            // configuration changes, never during normal temporal evaluation.
            auto idle = impl.device->waitIdle();
            if (!idle) { return idle; }
            impl.runtime->release(impl.handle);
            impl.handle = nullptr;
        }
        p->Set("Width", static_cast<int>(input.width));
        p->Set("Height", static_cast<int>(input.height));
        p->Set("OutWidth", static_cast<int>(output.width));
        p->Set("OutHeight", static_cast<int>(output.height));
        p->Set("DLSSNR.Width", static_cast<int>(output.width));
        p->Set("DLSSNR.Height", static_cast<int>(output.height));
        p->Set("DLSSNR.InputWidth", static_cast<int>(input.width));
        p->Set("DLSSNR.InputHeight", static_cast<int>(input.height));
        p->Set("DLSSNR.OutputWidth", static_cast<int>(output.width));
        p->Set("DLSSNR.OutputHeight", static_cast<int>(output.height));
        p->Set("DLSSNR.Output.Width", static_cast<int>(output.width));
        p->Set("DLSSNR.Output.Height", static_cast<int>(output.height));
        p->Set("DLSSNR.Hint.Render.Preset", static_cast<int>(s.preset));
        p->Set("DLSS.Output.Subrect.Base.X", 0u);
        p->Set("DLSS.Output.Subrect.Base.Y", 0u);
        p->Set("DLSSNR.Upscaling", static_cast<unsigned int>(s.upscaling));
        p->Set("DLSSNR.Scale", s.upscaling ? 0.5f : 1.0f);
        p->Set("DLSSNR.ScalingRatio", s.upscaling ? 0.5f : 1.0f);
        // The reference writes this callback through slot 0: Set(void*) in
        // the MSVC SDK ABI. An integer-valued parameter is not a callback.
        p->Set("DLSSNRComputeScalingRatioCallback", reinterpret_cast<void*>(&computeScalingRatio));
        auto result = ngxResult(impl.runtime->create(impl.runtime->native.device, command,
            kFeature, p, &impl.handle), "Vulkan CreateFeature1 (feature 18)", log);
        notifyExternalDescriptorSetBinding(commandBuffer);
        if (!result || impl.handle == nullptr) {
            if (impl.handle != nullptr) { impl.retired.push_back(impl.handle); impl.handle = nullptr; }
            if (result) { log = "DLSS-NR create succeeded without a feature handle"; }
            return result ? makeError(Error::Failure) : result;
        }
        impl.configuration = configuration;
        impl.settings = s;
        spdlog::info("[DLSS-NR] Created Vulkan feature 18: {}x{} -> {}x{}, preset {}",
            input.width, input.height, output.width, output.height, s.preset);
    }
    std::array resources{resourceFrom(desc.inputColor), resourceFrom(desc.outputColor),
        resourceFrom(desc.motionVectors), resourceFrom(desc.depth)};
    p->Set("DLSSNR.Color", static_cast<void*>(&resources[0]));
    p->Set("DLSSNR.Output", static_cast<void*>(&resources[1]));
    p->Set("DLSSNR.MVec", static_cast<void*>(&resources[2]));
    p->Set("DLSSNR.Depth", static_cast<void*>(&resources[3]));
    setSubrect(p, "Color", input.width, input.height);
    setSubrect(p, "MVec", input.width, input.height);
    setSubrect(p, "Depth", input.width, input.height);
    setSubrect(p, "Output", output.width, output.height);
    p->Set("DLSSNR.MVecScaleX", s.motionVectorScaleX);
    p->Set("DLSSNR.MVecScaleY", s.motionVectorScaleY);
    p->Set("DLSSNR.DepthInverted", static_cast<unsigned int>(s.depthInverted));
    p->Set("DLSSNR.Reset", static_cast<unsigned int>(s.reset || recreate));
    prepareStreamlineNgxCommandBuffer(commandBuffer);
    auto result = ngxResult(impl.runtime->evaluate(command, impl.handle, p, nullptr), "Vulkan EvaluateFeature", log);
    notifyExternalDescriptorSetBinding(commandBuffer);
    return result;
#else
    (void)commandBuffer;
    log = "DLSS-NR is not compiled into this build";
    return makeError(Error::Unsupported);
#endif
}

} // namespace metallic::render::vulkan
