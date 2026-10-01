#include "Runtime/Render/Profiling/NsightGraphicsCapture.h"
#include "Runtime/Render/Profiling/NsightReplayProtocol.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <fstream>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#ifndef METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
#define METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE 0
#endif

#ifndef METALLIC_NSIGHT_GRAPHICS_DEFAULT_INSTALLATION_ROOT
#define METALLIC_NSIGHT_GRAPHICS_DEFAULT_INSTALLATION_ROOT ""
#endif

#if METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
#include "Runtime/Render/GAPI/Vulkan/VulkanNative.h"
#include <NGFX_GraphicsCapture_Vulkan.h>
#include <NGFX_Vulkan.h>
#if __has_include(<NGFX_GPUTrace_Vulkan.h>)
#include <NGFX_GPUTrace_Vulkan.h>
#define METALLIC_HAS_NSIGHT_GPU_TRACE 1
#endif

#include <cwchar>
#endif

#ifdef _WIN32
#include <Windows.h>
#endif

namespace metallic::render::profiling {
namespace {

std::atomic_bool gVulkanCaptureInjected{false};
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
bool gExternalGpuTraceActive = false;
#endif

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

std::string traceHostFailure(const std::filesystem::path& logPath)
{
    // Keep the first backend error in the application's own startup log.
    std::ifstream log(logPath);
    for (std::string line; std::getline(log, line);) {
        if (line.find("ERROR:") != std::string::npos || line.starts_with("No metric sets") ||
            line.starts_with("Invalid activity")) {
            return line + "; see " + logPath.string();
        }
    }
    return "GPU Trace host exited; see " + logPath.string();
}

// Windows argv quoting, including nested --args and paths containing spaces.
std::wstring quoteArgument(const std::wstring& value)
{
    std::wstring result = L"\"";
    size_t slashes = 0;
    for (wchar_t c : value) {
        if (c == L'\\') { ++slashes; continue; }
        result.append(c == L'"' ? slashes * 2 + 1 : slashes, L'\\');
        result += c;
        slashes = 0;
    }
    result.append(slashes * 2, L'\\');
    return result + L'"';
}

bool writeTraceMetrics(const std::filesystem::path& path)
{
    std::ofstream metrics(path);
    metrics << R"([
  {"architecture":"Turing","metric-set-name":"Throughput Metrics","multi-pass-metrics":"false"},
  {"architecture":"Ampere GA10x","metric-set-name":"Top-Level Triage","multi-pass-metrics":"false"},
  {"architecture":"Orin GA10B","metric-set-name":"Top-Level Triage","multi-pass-metrics":"false"},
  {"architecture":"Ada","metric-set-name":"Top-Level Triage","multi-pass-metrics":"false"},
  {"architecture":"Thor GB10B","metric-set-name":"Top-Level Triage","multi-pass-metrics":"false"},
  {"architecture":"Blackwell GB20x","metric-set-name":"Top-Level Triage","multi-pass-metrics":"false"},
  {"architecture":"T25x GB20x","metric-set-name":"Top-Level Triage","multi-pass-metrics":"false"}
])";
    metrics.close();
    return static_cast<bool>(metrics);
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

bool beginExternalNsightGpuTrace(std::string& error)
{
    error.clear();
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
    if (gExternalGpuTraceActive) {
        error = "An external GPU Trace is already active";
        return false;
    }
    if (NsightGraphicsCapture::vulkanInjectionActive()) {
        error = "GPU Trace cannot share a process with Graphics Capture";
        return false;
    }
    NGFX_GPUTrace_InitializeActivity_Vulkan_Params initialize{};
    initialize.version = NGFX_GPUTrace_InitializeActivity_Vulkan_Params_VER;
    auto result = NGFX_GPUTrace_InitializeActivity_Vulkan(&initialize);
    if (result == NGFX_Result_Success) {
        NGFX_GPUTrace_StartTrace_Vulkan_Params start{};
        start.version = NGFX_GPUTrace_StartTrace_Vulkan_Params_VER;
        result = NGFX_GPUTrace_StartTrace_Vulkan(&start);
    }
    if (result != NGFX_Result_Success) {
        error = ngfxError("Initialize/start externally injected GPU Trace", result);
        return false;
    }
    gExternalGpuTraceActive = true;
    return true;
#else
    error = "Nsight GPU Trace SDK support was not compiled";
    return false;
#endif
}

bool endExternalNsightGpuTrace(std::string& error)
{
    error.clear();
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
    if (!gExternalGpuTraceActive) {
        error = "No external GPU Trace was started";
        return false;
    }
    NGFX_GPUTrace_StopTrace_Vulkan_Params stop{};
    stop.version = NGFX_GPUTrace_StopTrace_Vulkan_Params_VER;
    // No queue is needed: the workload owner has drained all submissions.
    const auto result = NGFX_GPUTrace_StopTrace_Vulkan(&stop);
    if (result != NGFX_Result_Success) {
        error = ngfxError("Stop externally injected GPU Trace", result);
        return false;
    }
    gExternalGpuTraceActive = false;
    return true;
#else
    error = "Nsight GPU Trace SDK support was not compiled";
    return false;
#endif
}

NsightGraphicsCapture::NsightGraphicsCapture()
    : state_(compiledAvailable()
              ? NsightGraphicsCaptureState::Uninitialized
              : NsightGraphicsCaptureState::Unavailable)
{
}

NsightGraphicsCapture::~NsightGraphicsCapture()
{
    closeTraceHost();
}

void NsightGraphicsCapture::closeTraceHost()
{
#ifdef _WIN32
    if (traceHostJob_ != nullptr) {
        TerminateJobObject(traceHostJob_, 0);
        CloseHandle(traceHostJob_);
        traceHostJob_ = nullptr;
    }
    if (traceHostProcess_ != nullptr) {
        // Only the host/worker created by this instance is owned here.
        if (WaitForSingleObject(traceHostProcess_, 1000) == WAIT_TIMEOUT) {
            TerminateProcess(traceHostProcess_, 0);
        }
        CloseHandle(traceHostProcess_);
        traceHostProcess_ = nullptr;
    }
#endif
}

bool NsightGraphicsCapture::startTraceHost(std::string& error)
{
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
    const auto executable = installationRoot_ / "host/windows-desktop-nomad-x64/ngfx.exe";
    traceHostLog_ = outputDirectory_ / ("GpuTraceHost-" + std::to_string(GetCurrentProcessId()) + ".log");
    std::wstring command = quoteArgument(executable.wstring()) + L" --activity \"GPU Trace Profiler\" --attach-pid " +
        std::to_wstring(GetCurrentProcessId()) + L" --output-dir " + quoteArgument(outputDirectory_.wstring()) +
        L" --per-arch-config-path " + quoteArgument(traceMetricsConfig_.wstring()) +
        L" --start-with-ngfx-sdk --stop-with-ngfx-sdk --keep-going --set-gpu-clocks unaltered --trace-timeout 120";
    if (!launchTraceHost(executable, std::move(command), error)) { return fail(error); }
    return true;
#else
    return fail("Nsight GPU Trace SDK support was not compiled", &error);
#endif
}

bool NsightGraphicsCapture::launchTraceHost(const std::filesystem::path& executable, std::wstring command, std::string& error)
{
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE log = CreateFileW(traceHostLog_.c_str(), GENERIC_WRITE, FILE_SHARE_READ,
        &security, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (log == INVALID_HANDLE_VALUE) { error = "Cannot create GPU Trace host log"; return false; }
    HANDLE input = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
        &security, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (input == INVALID_HANDLE_VALUE) {
        CloseHandle(log);
        error = "Cannot open GPU Trace host input"; return false;
    }
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (job == nullptr || !SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
        if (job != nullptr) { CloseHandle(job); }
        CloseHandle(input);
        CloseHandle(log);
        error = "Cannot create GPU Trace host lifetime job"; return false;
    }
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
    startup.wShowWindow = SW_HIDE;
    startup.hStdOutput = log;
    startup.hStdError = log;
    startup.hStdInput = input;
    PROCESS_INFORMATION process{};
    const BOOL launched = CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr,
        TRUE, CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, installationRoot_.c_str(), &startup, &process);
    const DWORD launchError = GetLastError();
    CloseHandle(log);
    if (input != INVALID_HANDLE_VALUE) { CloseHandle(input); }
    if (!launched) {
        CloseHandle(job);
        error = "Cannot start GPU Trace host (Win32 " + std::to_string(launchError) + ")"; return false;
    }
    if (!AssignProcessToJobObject(job, process.hProcess) || ResumeThread(process.hThread) == static_cast<DWORD>(-1)) {
        TerminateProcess(process.hProcess, 1);
        CloseHandle(process.hThread);
        CloseHandle(process.hProcess);
        CloseHandle(job);
        error = "Cannot manage GPU Trace host lifetime"; return false;
    }
    CloseHandle(process.hThread);
    traceHostProcess_ = process.hProcess;
    traceHostJob_ = job;
    return true;
#else
    (void)command;
    (void)executable;
    error = "Nsight GPU Trace SDK support was not compiled";
    return false;
#endif
}

bool NsightGraphicsCapture::compiledAvailable()
{
#if METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
    return true;
#else
    return false;
#endif
}

bool NsightGraphicsCapture::vulkanInjectionActive()
{
    if (gVulkanCaptureInjected.load(std::memory_order_relaxed)) {
        return true;
    }
#ifdef _WIN32
    return GetModuleHandleW(L"ngfx-capture-interception.dll") != nullptr;
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
    const auto requestedMode = config.mode == NsightCaptureMode::Default ? NsightCaptureMode::GPUTrace : config.mode;
#if !defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
    if (requestedMode == NsightCaptureMode::GPUTrace) {
        return fail("Nsight GPU Trace SDK support was not compiled", &error);
    }
#endif
    if (state_ == NsightGraphicsCaptureState::Ready ||
        state_ == NsightGraphicsCaptureState::CapturePending ||
        state_ == NsightGraphicsCaptureState::CaptureCompleted) {
        if (mode_ != requestedMode) {
            error = "Changing Nsight activity requires restarting the process";
            return false;
        }
        return true;
    }
    if (state_ == NsightGraphicsCaptureState::Error) {
        error = lastError_;
        return false;
    }
    mode_ = requestedMode;

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

    const wchar_t* requiredLibraries[] = {
        L"ngfx-api-bootstrap.dll",
        mode_ == NsightCaptureMode::GPUTrace ? L"WarpViz.Injection.dll" : L"ngfx-capture-injection.dll",
        mode_ == NsightCaptureMode::GPUTrace ? L"WarpVizTarget.dll" : L"ngfx-capture-interception.dll",
    };
    for (const wchar_t* libraryName : requiredLibraries) {
        if (!std::filesystem::is_regular_file(allowedLibraryDirectory() / libraryName, filesystemError) ||
            filesystemError) {
            return fail("Nsight Graphics capture runtime is incomplete", &error);
        }
    }

    outputDirectory_ = config.outputDirectory.empty() ? std::filesystem::current_path() : config.outputDirectory;
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

    if (config.enableReplayCollection && mode_ == NsightCaptureMode::GraphicsCapture) {
        // Start before injection: Nsight propagates Graphics Capture into child
        // processes. Only this clean worker may launch a GPU Trace replay.
        wchar_t modulePath[32768]{};
        if (GetModuleFileNameW(nullptr, modulePath, 32768) == 0) {
            return fail("Cannot locate the replay worker", &error);
        }
        const auto worker = std::filesystem::path(modulePath).parent_path() / "MetallicNsightReplay.exe";
        replayMailbox_ = outputDirectory_ / ("ReplayWorker-" + std::to_string(GetCurrentProcessId()) +
            "-" + std::to_string(GetTickCount64()));
        if (!std::filesystem::create_directory(replayMailbox_, filesystemError)) {
            return fail("Cannot create replay worker mailbox", &error);
        }
        traceHostLog_ = replayMailbox_ / "Worker.log";
        const auto host = installationRoot_ / "host/windows-desktop-nomad-x64/ngfx.exe";
        const auto command = quoteArgument(worker.wstring()) + L" --nsight-replay-worker " +
            quoteArgument(replayMailbox_.wstring()) + L" " + quoteArgument(host.wstring());
        if (!launchTraceHost(worker, command, error)) { return fail(error); }
    }

    NGFX_SetLibraryLoadFn(loadNsightGraphicsLibrary);

#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
    if (mode_ == NsightCaptureMode::GPUTrace) {
        traceMetricsConfig_ = config.gpuTraceMetricsConfig;
        if (traceMetricsConfig_.empty()) {
            traceMetricsConfig_ = outputDirectory_ / ("GpuTraceMetrics-" + std::to_string(GetCurrentProcessId()) + ".json");
            // Names verified against ngfx --help-all (SDK 0.9.2). Every supported
            // architecture gets an explicit single-pass set, including Turing,
            // which does not expose Top-Level Triage. Numeric IDs are not stable.
            if (!writeTraceMetrics(traceMetricsConfig_)) {
                return fail("Cannot write GPU Trace metrics configuration", &error);
            }
        }
        traceMetricsConfig_ = std::filesystem::canonical(traceMetricsConfig_, filesystemError);
        if (filesystemError) { return fail("GPU Trace metrics configuration does not exist", &error); }
        NGFX_GPUTrace_InjectionSettings settings{};
        auto result = NGFX_GPUTrace_InjectionSettings_SetDefaults(&settings);
        if (result != NGFX_Result_Success) {
            return fail(ngfxError("NGFX_GPUTrace_InjectionSettings_SetDefaults", result), &error);
        }
        settings.startEvent = NGFX_GPUTrace_StartEvent_NGFXSDK;
        settings.stopEvent = NGFX_GPUTrace_StopEvent_NGFXSDK;
        settings.gpuClockMode = NGFX_GPUTrace_GPUClockMode_Unaltered;
        settings.vsyncMode = NGFX_GPUTrace_VSyncMode_ApplicationControlled;
        settings.hudPosition = config.showHud ? NGFX_GPUTrace_HUDPosition_TopLeft : NGFX_GPUTrace_HUDPosition_Hidden;
        NGFX_GPUTrace_Inject_Vulkan_Params inject{};
        inject.version = NGFX_GPUTrace_Inject_Vulkan_Params_VER;
        inject.installationPath = installationRoot_.c_str();
        inject.settings = &settings;
        result = NGFX_GPUTrace_Inject_Vulkan(&inject);
        if (result != NGFX_Result_Success) {
            return fail(ngfxError("NGFX_GPUTrace_Inject_Vulkan", result), &error);
        }
        NGFX_GPUTrace_InitializeActivity_Vulkan_Params initialize{};
        initialize.version = NGFX_GPUTrace_InitializeActivity_Vulkan_Params_VER;
        result = NGFX_GPUTrace_InitializeActivity_Vulkan(&initialize);
        if (result != NGFX_Result_Success) {
            return fail(ngfxError("NGFX_GPUTrace_InitializeActivity_Vulkan", result), &error);
        }
        if (!startTraceHost(error)) { return false; }
        state_ = NsightGraphicsCaptureState::Ready;
        lastError_.clear();
        return true;
    }
#endif

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
    // Injection remains active even if subsequent activity initialization fails.
    gVulkanCaptureInjected.store(true, std::memory_order_relaxed);

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

bool NsightGraphicsCapture::prepareBeforeSubmission(Queue& queue, std::string& error)
{
    error.clear();
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
    if (mode_ != NsightCaptureMode::GPUTrace || traceHostProcess_ == nullptr) { return true; }
    if (state_ != NsightGraphicsCaptureState::Ready) { error = lastError_; return false; }
    if (WaitForSingleObject(traceHostProcess_, 0) == WAIT_OBJECT_0) {
        return fail(traceHostFailure(traceHostLog_), &error);
    }

    // QueueSubmit implicitly activates Nsight, and can block indefinitely if
    // its host rejects the metrics. Do it explicitly before scene resources or
    // rendering are submitted, with an independent bounded startup watchdog.
    // Only this thread calls NGFX. The watchdog touches only the child process.
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    std::jthread watchdog([&](std::stop_token stop) {
        while (!stop.stop_requested()) {
            const bool exited = WaitForSingleObject(traceHostProcess_, 50) == WAIT_OBJECT_0;
            const bool timedOut = std::chrono::steady_clock::now() >= deadline;
            if (!stop.stop_requested() && (exited || timedOut)) {
                const std::string message = exited ? traceHostFailure(traceHostLog_)
                    : "GPU Trace startup timed out after 60 seconds; see " + traceHostLog_.string();
                std::fprintf(stderr, "[Nsight startup] %s\n", message.c_str());
                std::fflush(stderr);
                // Unwinding into the intercepted Vulkan driver can hang too.
                // The owned job closes automatically, including on this path.
                // TerminateProcess can itself block inside Nsight's injected
                // teardown. A new, unnamed job contains only this process and
                // lets Windows terminate it without entering that interception.
                if (HANDLE failedStartup = CreateJobObjectW(nullptr, nullptr)) {
                    if (AssignProcessToJobObject(failedStartup, GetCurrentProcess())) {
                        TerminateJobObject(failedStartup, 1);
                    }
                    CloseHandle(failedStartup);
                }
                TerminateProcess(GetCurrentProcess(), 1);
                return;
            }
        }
    });
    NGFX_GPUTrace_ActivateTrace_Vulkan_Params activate{};
    activate.version = NGFX_GPUTrace_ActivateTrace_Vulkan_Params_VER;
    activate.queue = vulkan::nativeQueue(queue).queue;
    const auto result = NGFX_GPUTrace_ActivateTrace_Vulkan(&activate);
    watchdog.request_stop();
    watchdog.join();
    if (result != NGFX_Result_Success) {
        return fail(ngfxError("GPU Trace startup activation", result) + "; " + traceHostFailure(traceHostLog_), &error);
    }
#else
    (void)queue;
#endif
    return true;
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
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
    if (mode_ == NsightCaptureMode::GPUTrace) {
        if (request.explicitFrameBoundaries) {
            error = lastError_ = "GPU Trace export requires main viewport Present boundaries";
            return false;
        }
        const auto result = NGFX_GPUTrace_GetTraceFileCount(&countParams);
        if (result != NGFX_Result_Success) {
            return fail(ngfxError("NGFX_GPUTrace_GetTraceFileCount", result), &error);
        }
        pendingCaptureIndex_ = countParams.count;
        traceFramesBeforeStart_ = request.framesBeforeStart;
        traceFramesRemaining_ = request.framesToCapture;
        traceStarted_ = false;
        traceStopped_ = false;
        traceDeadline_ = std::chrono::steady_clock::now() + std::chrono::seconds(120);
        capturePath_.clear();
        lastError_.clear();
        state_ = NsightGraphicsCaptureState::CapturePending;
        return true;
    }
#endif
    NGFX_Result result = NGFX_GraphicsCapture_GetCaptureFileCount(&countParams);
    if (result != NGFX_Result_Success) {
        lastError_ = ngfxError("NGFX_GraphicsCapture_GetCaptureFileCount", result);
        error = lastError_;
        return false;
    }

    NGFX_GraphicsCapture_RequestCapture_Vulkan_Params captureParams{};
    captureParams.version = NGFX_GraphicsCapture_RequestCapture_Vulkan_Params_VER;
    captureParams.delimiter = request.explicitFrameBoundaries
        ? NGFX_GraphicsCapture_Delimiter_FrameBoundary : NGFX_GraphicsCapture_Delimiter_Present;
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

bool NsightGraphicsCapture::frameBoundary(Queue& queue, Texture* output, std::string& error)
{
    error.clear();
#if METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
    if (state_ != NsightGraphicsCaptureState::Ready &&
        state_ != NsightGraphicsCaptureState::CapturePending &&
        state_ != NsightGraphicsCaptureState::CaptureCompleted) {
        error = "Nsight Graphics capture is not ready for a frame boundary";
        return false;
    }
    NGFX_FrameBoundary_Vulkan_Params params{};
    params.version = NGFX_FrameBoundary_Vulkan_Params_VER;
    params.queue = vulkan::nativeQueue(queue).queue;
    NGFX_ResourceDescription_Vulkan resource{};
    if (output != nullptr) {
        resource.version = NGFX_ResourceDescription_Vulkan_VER;
        resource.type = NGFX_ResourceType_Vulkan_VkImage;
        resource.image = vulkan::nativeTexture(*output).image;
        params.outputResources = &resource;
        params.numOutputResources = 1;
    }
    const NGFX_Result result = NGFX_FrameBoundary_Vulkan(&params);
    if (result != NGFX_Result_Success) {
        return fail(ngfxError("NGFX_FrameBoundary_Vulkan", result), &error);
    }
    return true;
#else
    (void)queue;
    (void)output;
    error = "Nsight Graphics Capture SDK support was not compiled";
    return false;
#endif
}

bool NsightGraphicsCapture::afterPresent(Queue& queue, std::string& error)
{
    error.clear();
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
    if (mode_ != NsightCaptureMode::GPUTrace || !hasOutstandingCapture() || traceStopped_) { return true; }
    if (!traceStarted_) {
        if (traceFramesBeforeStart_ != 0) { --traceFramesBeforeStart_; return true; }
        NGFX_GPUTrace_GetStatus_Params status{};
        status.version = NGFX_GPUTrace_GetStatus_Params_VER;
        auto result = NGFX_GPUTrace_GetStatus(&status);
        if (result != NGFX_Result_Success) { return fail(ngfxError("GPU Trace status", result), &error); }
        if (status.status == NGFX_GPUTrace_Status_Inactive || status.status == NGFX_GPUTrace_Status_Draining) {
            return true;
        }
        if (status.status != NGFX_GPUTrace_Status_Active) {
            return fail("GPU Trace host is not ready; see " + traceHostLog_.string(), &error);
        }
        NGFX_GPUTrace_StartTrace_Vulkan_Params start{};
        start.version = NGFX_GPUTrace_StartTrace_Vulkan_Params_VER;
        result = NGFX_GPUTrace_StartTrace_Vulkan(&start);
        if (result != NGFX_Result_Success) { return fail(ngfxError("GPU Trace start", result), &error); }
        traceStarted_ = true;
    } else if (--traceFramesRemaining_ == 0) {
        NGFX_GPUTrace_StopTrace_Vulkan_Params stop{};
        stop.version = NGFX_GPUTrace_StopTrace_Vulkan_Params_VER;
        stop.queue = vulkan::nativeQueue(queue).queue;
        const auto result = NGFX_GPUTrace_StopTrace_Vulkan(&stop);
        if (result != NGFX_Result_Success) { return fail(ngfxError("GPU Trace stop", result), &error); }
        traceStopped_ = true;
    }
#else
    (void)queue;
#endif
    return true;
}

NsightGraphicsCapturePollResult NsightGraphicsCapture::poll()
{
#if METALLIC_HAS_NSIGHT_GRAPHICS_CAPTURE
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
    if (mode_ == NsightCaptureMode::GPUTrace && traceHostProcess_ != nullptr &&
        WaitForSingleObject(traceHostProcess_, 0) == WAIT_OBJECT_0) {
        fail(traceHostFailure(traceHostLog_));
        return {state_, capturePath_, lastError_};
    }
    if (mode_ == NsightCaptureMode::GPUTrace && hasOutstandingCapture() &&
        std::chrono::steady_clock::now() >= traceDeadline_) {
        fail("GPU Trace export timed out; see " + traceHostLog_.string());
        return {state_, capturePath_, lastError_};
    }
#endif
    if (state_ == NsightGraphicsCaptureState::CapturePending) {
        auto waitForPath = NGFX_GraphicsCapture_WaitForCaptureFilePath;
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
        if (mode_ == NsightCaptureMode::GPUTrace) { waitForPath = NGFX_GPUTrace_WaitForTraceFilePath; }
#endif
        NGFX_WaitForArtifactFilePath_Params sizeParams{};
        sizeParams.version = NGFX_WaitForArtifactFilePath_Params_VER;
        sizeParams.artifactIndex = pendingCaptureIndex_;
        sizeParams.timeoutMs = 0;

        NGFX_Result result = waitForPath(&sizeParams);
        if (result == NGFX_Result_Timeout) {
            return {state_, {}, {}};
        }
        if (result != NGFX_Result_InsufficientBuffer && result != NGFX_Result_Success) {
            fail(ngfxError("Wait for Nsight export path", result));
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

            result = waitForPath(&pathParams);
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

            fail(ngfxError("Wait for Nsight export path", result));
            return {state_, {}, lastError_};
        }

        fail("Nsight Graphics capture path changed size repeatedly");
    }
#endif

    return {state_, capturePath_, lastError_};
}

bool NsightGraphicsCapture::startReplayTrace(std::string& error)
{
    error.clear();
    if (replayTracePending_) { error = "Replay GPU Trace is already pending"; return false; }
    replayTracePath_.clear();
    replayTraceError_.clear();
    const auto reject = [&](std::string message) {
        replayTraceError_ = std::move(message);
        error = replayTraceError_;
        return false;
    };
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
    if (mode_ != NsightCaptureMode::GraphicsCapture || state_ != NsightGraphicsCaptureState::CaptureCompleted) {
        return reject("Replay profiling requires a completed Graphics Capture");
    }
    const auto host = installationRoot_ / "host/windows-desktop-nomad-x64/ngfx.exe";
    const auto replayer = installationRoot_ / "host/windows-desktop-nomad-x64/ngfx-replay.exe";
    replayTraceDirectory_ = capturePath_.parent_path() / (capturePath_.stem().wstring() + L"_Collected");
    std::error_code filesystemError;
    if (!std::filesystem::create_directory(replayTraceDirectory_, filesystemError)) {
        return reject("Cannot create a fresh replay collection directory: " + replayTraceDirectory_.string());
    }
    traceMetricsConfig_ = replayTraceDirectory_ / "Metrics.json";
    traceHostLog_ = replayTraceDirectory_ / "GpuTraceHost.log";
    if (!writeTraceMetrics(traceMetricsConfig_)) { return reject("Cannot write replay metrics configuration"); }
    const std::wstring replayArgs = L"--present-hidden --vsync-off --no-present-blit "
        L"--no-block-on-incompatibility --inject-full-frame-perf-marker --loop-count 3 " +
        quoteArgument(capturePath_.wstring());
    std::wstring command = quoteArgument(host.wstring()) + L" --activity \"GPU Trace Profiler\" --exe " +
        quoteArgument(replayer.wstring()) + L" --dir " + quoteArgument(capturePath_.parent_path().wstring()) +
        L" --args " + quoteArgument(replayArgs) + L" --output-dir " + quoteArgument(replayTraceDirectory_.wstring()) +
        L" --per-arch-config-path " + quoteArgument(traceMetricsConfig_.wstring()) +
        L" --start-on-replay-begin --stop-on-replay-end --max-duration-ms 1000 --set-gpu-clocks unaltered --trace-timeout 120";
    if (traceHostProcess_ == nullptr || WaitForSingleObject(traceHostProcess_, 0) != WAIT_TIMEOUT ||
        replayMailbox_.empty()) {
        return reject("Replay worker is unavailable; restart with --nsight-capture");
    }
    std::filesystem::remove(replayMailbox_ / "result.txt", filesystemError);
    if (filesystemError) { return reject("Cannot clear previous replay result"); }
    std::ofstream request(replayMailbox_ / "request.tmp", std::ios::binary);
    const bool written = writeReplayString(request, command) && writeReplayString(request, traceHostLog_.wstring()) &&
        writeReplayString(request, capturePath_.parent_path().wstring());
    request.close();
    if (!written || !request) { return reject("Cannot write replay request"); }
    std::filesystem::rename(replayMailbox_ / "request.tmp", replayMailbox_ / "request.bin", filesystemError);
    if (filesystemError) { return reject("Cannot submit replay request"); }
    traceDeadline_ = std::chrono::steady_clock::now() + std::chrono::seconds(180);
    replayTracePending_ = true;
    return true;
#else
    return reject("Nsight GPU Trace SDK support was not compiled");
#endif
}

void NsightGraphicsCapture::pollReplayTrace()
{
#if defined(METALLIC_HAS_NSIGHT_GPU_TRACE)
    if (!replayTracePending_) { return; }
    const bool workerExited = WaitForSingleObject(traceHostProcess_, 0) != WAIT_TIMEOUT;
    std::error_code resultError;
    const bool resultAvailable = std::filesystem::exists(replayMailbox_ / "result.txt", resultError);
    if (!resultAvailable && !workerExited && std::chrono::steady_clock::now() < traceDeadline_) { return; }
    DWORD exitCode = 1;
    std::ifstream result(replayMailbox_ / "result.txt");
    if (resultAvailable && (result >> exitCode) && exitCode == 0) {
        std::error_code error;
        for (std::filesystem::directory_iterator it(replayTraceDirectory_, error), end;
             !error && it != end; it.increment(error)) {
            if (it->path().extension() == ".ngfx-gputrace" && it->is_regular_file(error) &&
                it->file_size(error) > 0 && !error) {
                replayTracePath_ = it->path();
                break;
            }
        }
        if (replayTracePath_.empty()) {
            replayTraceError_ = "Replay finished without a nonempty GPU Trace; see " + traceHostLog_.string();
        }
    } else {
        replayTraceError_ = !resultAvailable && !workerExited
            ? "Replay GPU Trace timed out after 180 seconds; see " + traceHostLog_.string()
            : "Replay GPU Trace failed (exit " + std::to_string(exitCode) + "): " + traceHostFailure(traceHostLog_);
    }
    replayTracePending_ = false;
    if (!resultAvailable) { closeTraceHost(); }
#endif
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
