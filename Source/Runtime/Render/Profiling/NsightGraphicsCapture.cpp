#include "Runtime/Render/Profiling/NsightGraphicsCapture.h"

#include <algorithm>
#include <system_error>
#include <utility>
#include <vector>

#ifndef METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
#define METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE 0
#endif

#ifndef METALLIC_NSIGHT_GRAPHICS_DEFAULT_INSTALLATION_ROOT
#define METALLIC_NSIGHT_GRAPHICS_DEFAULT_INSTALLATION_ROOT ""
#endif

#if METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
#include <NGFX_GraphicsCapture_Vulkan.h>
#include <Windows.h>

#include <cwchar>
#endif

namespace metallic::render::profiling {
namespace {

#if METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE

std::filesystem::path& allowedLibraryDirectory()
{
    static std::filesystem::path directory;
    return directory;
}

bool pathsEqual(const std::filesystem::path& lhs, const std::filesystem::path& rhs)
{
    const std::wstring lhsText = lhs.native();
    const std::wstring rhsText = rhs.native();
    return lhsText.size() == rhsText.size() &&
        _wcsnicmp(lhsText.c_str(), rhsText.c_str(), lhsText.size()) == 0;
}

void* loadNsightGraphicsLibrary(const NGFX_PathChar* libraryName)
{
    if (libraryName == nullptr || libraryName[0] == L'\0') {
        return nullptr;
    }

    std::error_code error;
    const std::filesystem::path libraryPath = std::filesystem::canonical(libraryName, error);
    if (error || libraryPath.extension() != L".dll") {
        return nullptr;
    }

    const std::filesystem::path parentPath = std::filesystem::canonical(libraryPath.parent_path(), error);
    if (error || !pathsEqual(parentPath, allowedLibraryDirectory())) {
        return nullptr;
    }

    return reinterpret_cast<void*>(LoadLibraryExW(
        libraryPath.c_str(),
        nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS));
}

const char* resultName(NGFX_Result result)
{
    switch (result) {
    case NGFX_Result_Success:
        return "Success";
    case NGFX_Result_NotImplemented:
        return "NotImplemented";
    case NGFX_Result_LibNotFound:
        return "LibNotFound";
    case NGFX_Result_InvalidLib:
        return "InvalidLib";
    case NGFX_Result_DifferentActivityInjected:
        return "DifferentActivityInjected";
    case NGFX_Result_InvalidParameter:
        return "InvalidParameter";
    case NGFX_Result_InvalidState:
        return "InvalidState";
    case NGFX_Result_UnspecifiedError:
        return "UnspecifiedError";
    case NGFX_Result_Timeout:
        return "Timeout";
    case NGFX_Result_InsufficientBuffer:
        return "InsufficientBuffer";
    case NGFX_Result_COUNT:
        return "Unknown";
    }

    return "Unknown";
}

std::string ngfxError(const char* operation, NGFX_Result result)
{
    return std::string(operation) + " returned NGFX_Result_" + resultName(result);
}

std::string pathToUtf8(const std::filesystem::path& path)
{
    const std::u8string utf8 = path.u8string();
    return std::string(reinterpret_cast<const char*>(utf8.data()), utf8.size());
}

#endif

const char* stateName(NsightGraphicsCaptureState state)
{
    switch (state) {
    case NsightGraphicsCaptureState::Unavailable:
        return "Unavailable";
    case NsightGraphicsCaptureState::Uninitialized:
        return "Uninitialized";
    case NsightGraphicsCaptureState::Ready:
        return "Ready";
    case NsightGraphicsCaptureState::CapturePending:
        return "Capture pending";
    case NsightGraphicsCaptureState::CaptureCompleted:
        return "Capture completed";
    case NsightGraphicsCaptureState::Error:
        return "Error";
    }

    return "Unknown";
}

} // namespace

NsightGraphicsCapture::NsightGraphicsCapture()
    : state_(compiledAvailable()
              ? NsightGraphicsCaptureState::Uninitialized
              : NsightGraphicsCaptureState::Unavailable)
{
}

bool NsightGraphicsCapture::compiledAvailable()
{
#if METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
    return true;
#else
    return false;
#endif
}

std::filesystem::path NsightGraphicsCapture::defaultInstallationRoot()
{
    return std::filesystem::path(METALLIC_NSIGHT_GRAPHICS_DEFAULT_INSTALLATION_ROOT);
}

bool NsightGraphicsCapture::initializeBeforeGraphics(
    const NsightGraphicsCaptureConfig& config,
    std::string& error)
{
    error.clear();

#if !METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
    state_ = NsightGraphicsCaptureState::Unavailable;
    lastError_ = "Nsight Graphics Capture SDK support was not compiled";
    error = lastError_;
    return false;
#else
    if (state_ == NsightGraphicsCaptureState::Ready ||
        state_ == NsightGraphicsCaptureState::CapturePending ||
        state_ == NsightGraphicsCaptureState::CaptureCompleted) {
        return true;
    }
    if (state_ == NsightGraphicsCaptureState::Error) {
        error = lastError_;
        return false;
    }

    installationRoot_ = config.installationRoot.empty()
        ? defaultInstallationRoot()
        : config.installationRoot;
    if (installationRoot_.empty()) {
        return fail("Nsight Graphics installation root is empty", &error);
    }

    std::error_code filesystemError;
    installationRoot_ = std::filesystem::canonical(installationRoot_, filesystemError);
    if (filesystemError) {
        return fail("Nsight Graphics installation root does not exist", &error);
    }

    allowedLibraryDirectory() = std::filesystem::canonical(
        installationRoot_ / "target" / "windows-desktop-nomad-x64",
        filesystemError);
    if (filesystemError) {
        return fail("Nsight Graphics x64 target directory was not found", &error);
    }

    constexpr const wchar_t* requiredLibraries[] = {
        L"ngfx-api-bootstrap.dll",
        L"ngfx-capture-injection.dll",
        L"ngfx-capture-interception.dll",
    };
    for (const wchar_t* libraryName : requiredLibraries) {
        if (!std::filesystem::is_regular_file(allowedLibraryDirectory() / libraryName, filesystemError) ||
            filesystemError) {
            return fail("Nsight Graphics capture runtime is incomplete", &error);
        }
    }

    outputDirectory_ = config.outputDirectory;
    outputDirectoryUtf8_.clear();
    if (!outputDirectory_.empty()) {
        std::filesystem::create_directories(outputDirectory_, filesystemError);
        if (filesystemError) {
            return fail("Failed to create the Nsight Graphics capture output directory", &error);
        }
        outputDirectory_ = std::filesystem::canonical(outputDirectory_, filesystemError);
        if (filesystemError) {
            return fail("Failed to resolve the Nsight Graphics capture output directory", &error);
        }
        outputDirectoryUtf8_ = pathToUtf8(outputDirectory_);
    }

    NGFX_SetLibraryLoadFn(loadNsightGraphicsLibrary);

    NGFX_GraphicsCapture_InjectionSettings settings{};
    NGFX_Result result = NGFX_GraphicsCapture_InjectionSettings_SetDefaults(&settings);
    if (result != NGFX_Result_Success) {
        return fail(ngfxError("NGFX_GraphicsCapture_InjectionSettings_SetDefaults", result), &error);
    }
    settings.noHUD = !config.showHud;
    if (!outputDirectoryUtf8_.empty()) {
        settings.outputDir = outputDirectoryUtf8_.c_str();
    }

    NGFX_GraphicsCapture_Inject_Vulkan_Params injectParams{};
    injectParams.version = NGFX_GraphicsCapture_Inject_Vulkan_Params_VER;
    injectParams.installationPath = installationRoot_.c_str();
    injectParams.settings = &settings;
    result = NGFX_GraphicsCapture_Inject_Vulkan(&injectParams);
    if (result != NGFX_Result_Success) {
        return fail(ngfxError("NGFX_GraphicsCapture_Inject_Vulkan", result), &error);
    }

    NGFX_GraphicsCapture_InitializeActivity_Vulkan_Params initializeParams{};
    initializeParams.version = NGFX_GraphicsCapture_InitializeActivity_Vulkan_Params_VER;
    result = NGFX_GraphicsCapture_InitializeActivity_Vulkan(&initializeParams);
    if (result != NGFX_Result_Success) {
        return fail(ngfxError("NGFX_GraphicsCapture_InitializeActivity_Vulkan", result), &error);
    }

    state_ = NsightGraphicsCaptureState::Ready;
    lastError_.clear();
    return true;
#endif
}

bool NsightGraphicsCapture::requestCapture(
    const NsightGraphicsCaptureRequest& request,
    std::string& error)
{
    error.clear();

#if !METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
    lastError_ = "Nsight Graphics Capture SDK support was not compiled";
    error = lastError_;
    return false;
#else
    if (state_ == NsightGraphicsCaptureState::CapturePending) {
        lastError_ = "An Nsight Graphics capture is already pending";
        error = lastError_;
        return false;
    }
    if (state_ != NsightGraphicsCaptureState::Ready &&
        state_ != NsightGraphicsCaptureState::CaptureCompleted) {
        lastError_ = "Nsight Graphics Capture is not ready";
        error = lastError_;
        return false;
    }
    if (request.framesToCapture == 0 || request.framesToCapture > 60) {
        lastError_ = "Nsight Graphics framesToCapture must be in [1, 60]";
        error = lastError_;
        return false;
    }

    NGFX_ArtifactFileCount_Params countParams{};
    countParams.version = NGFX_ArtifactFileCount_Params_VER;
    NGFX_Result result = NGFX_GraphicsCapture_GetCaptureFileCount(&countParams);
    if (result != NGFX_Result_Success) {
        lastError_ = ngfxError("NGFX_GraphicsCapture_GetCaptureFileCount", result);
        error = lastError_;
        return false;
    }

    NGFX_GraphicsCapture_RequestCapture_Vulkan_Params captureParams{};
    captureParams.version = NGFX_GraphicsCapture_RequestCapture_Vulkan_Params_VER;
    captureParams.delimiter = NGFX_GraphicsCapture_Delimiter_Present;
    captureParams.framesBeforeStart = request.framesBeforeStart;
    captureParams.framesToCapture = request.framesToCapture;
    result = NGFX_GraphicsCapture_RequestCapture_Vulkan(&captureParams);
    if (result != NGFX_Result_Success) {
        lastError_ = ngfxError("NGFX_GraphicsCapture_RequestCapture_Vulkan", result);
        error = lastError_;
        return false;
    }

    pendingCaptureIndex_ = countParams.count;
    capturePath_.clear();
    lastError_.clear();
    state_ = NsightGraphicsCaptureState::CapturePending;
    return true;
#endif
}

NsightGraphicsCapturePollResult NsightGraphicsCapture::poll()
{
#if METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
    if (state_ == NsightGraphicsCaptureState::CapturePending) {
        NGFX_WaitForArtifactFilePath_Params sizeParams{};
        sizeParams.version = NGFX_WaitForArtifactFilePath_Params_VER;
        sizeParams.artifactIndex = pendingCaptureIndex_;
        sizeParams.timeoutMs = 0;

        NGFX_Result result = NGFX_GraphicsCapture_WaitForCaptureFilePath(&sizeParams);
        if (result == NGFX_Result_Timeout) {
            return {state_, {}, {}};
        }
        if (result != NGFX_Result_InsufficientBuffer && result != NGFX_Result_Success) {
            fail(ngfxError("NGFX_GraphicsCapture_WaitForCaptureFilePath", result));
            return {state_, {}, lastError_};
        }
        if (sizeParams.requiredPathCapacity == 0) {
            fail("Nsight Graphics returned an empty capture path");
            return {state_, {}, lastError_};
        }

        std::vector<NGFX_PathChar> pathBuffer(sizeParams.requiredPathCapacity);
        for (uint32_t attempt = 0; attempt < 2; ++attempt) {
            NGFX_WaitForArtifactFilePath_Params pathParams{};
            pathParams.version = NGFX_WaitForArtifactFilePath_Params_VER;
            pathParams.artifactIndex = pendingCaptureIndex_;
            pathParams.timeoutMs = 0;
            pathParams.filePath = pathBuffer.data();
            pathParams.filePathCapacity = static_cast<uint32_t>(pathBuffer.size());

            result = NGFX_GraphicsCapture_WaitForCaptureFilePath(&pathParams);
            if (result == NGFX_Result_Success) {
                capturePath_ = std::filesystem::path(pathBuffer.data());
                state_ = NsightGraphicsCaptureState::CaptureCompleted;
                lastError_.clear();
                return {state_, capturePath_, {}};
            }
            if (result == NGFX_Result_Timeout) {
                return {state_, {}, {}};
            }
            if (result == NGFX_Result_InsufficientBuffer &&
                pathParams.requiredPathCapacity > pathBuffer.size()) {
                pathBuffer.resize(pathParams.requiredPathCapacity);
                continue;
            }

            fail(ngfxError("NGFX_GraphicsCapture_WaitForCaptureFilePath", result));
            return {state_, {}, lastError_};
        }

        fail("Nsight Graphics capture path changed size repeatedly");
    }
#endif

    return {state_, capturePath_, lastError_};
}

NsightGraphicsCaptureState NsightGraphicsCapture::state() const
{
    return state_;
}

const char* NsightGraphicsCapture::statusText() const
{
    if (!lastError_.empty()) {
        return lastError_.c_str();
    }
    return stateName(state_);
}

bool NsightGraphicsCapture::hasOutstandingCapture() const
{
    return state_ == NsightGraphicsCaptureState::CapturePending;
}

const std::filesystem::path& NsightGraphicsCapture::capturePath() const
{
    return capturePath_;
}

const std::string& NsightGraphicsCapture::lastError() const
{
    return lastError_;
}

bool NsightGraphicsCapture::fail(std::string message, std::string* error)
{
    state_ = NsightGraphicsCaptureState::Error;
    lastError_ = std::move(message);
    if (error != nullptr) {
        *error = lastError_;
    }
    return false;
}

} // namespace metallic::render::profiling
