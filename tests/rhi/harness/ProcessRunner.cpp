#include "ProcessRunner.h"
#include <stdexcept>
#ifdef _WIN32
#include <Windows.h>
#include <shellapi.h>
#endif

namespace metallic::tests::bench {
#ifdef _WIN32
namespace {
struct Handle {
    HANDLE value = nullptr;
    ~Handle() { if (value && value != INVALID_HANDLE_VALUE) { CloseHandle(value); } }
};
std::wstring wide(const std::string& text)
{
    const int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), int(text.size()), nullptr, 0);
    if (!size && !text.empty()) { throw std::runtime_error("invalid UTF-8 process argument"); }
    std::wstring result(size, L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), int(text.size()), result.data(), size);
    return result;
}
std::wstring quote(const std::wstring& argument)
{
    std::wstring value = L"\"";
    size_t slashes = 0;
    for (const auto character : argument) {
        if (character == L'\\') { ++slashes; continue; }
        value.append(slashes * (character == L'"' ? 2 : 1), L'\\');
        slashes = 0;
        if (character == L'"') { value += L'\\'; }
        value += character;
    }
    value.append(slashes * 2, L'\\');
    return value + L'"';
}
} // namespace
#endif

std::filesystem::path executablePath()
{
#ifdef _WIN32
    std::wstring path(32768, L'\0');
    const auto size = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
    if (!size || size == path.size()) { throw std::runtime_error("cannot locate executable"); }
    path.resize(size);
    return path;
#else
    throw std::runtime_error("testbench process isolation currently requires Windows");
#endif
}

std::optional<std::filesystem::path> findExecutable(const std::string& name)
{
#ifdef _WIN32
    std::wstring path(32768, L'\0');
    const auto size = SearchPathW(nullptr, wide(name).c_str(), L".exe", DWORD(path.size()), path.data(), nullptr);
    if (!size || size >= path.size()) { return std::nullopt; }
    path.resize(size);
    return std::filesystem::path(path);
#else
    return std::nullopt;
#endif
}

std::vector<std::string> nativeArguments(int argc, char** argv)
{
#ifdef _WIN32
    int count = 0;
    auto* values = CommandLineToArgvW(GetCommandLineW(), &count);
    if (!values) { throw std::runtime_error("cannot read Unicode command line"); }
    struct Release { wchar_t** values; ~Release() { LocalFree(values); } } release{values};
    std::vector<std::string> result;
    for (int i = 0; i < count; ++i) {
        const auto text = std::filesystem::path(values[i]).u8string();
        result.emplace_back(reinterpret_cast<const char*>(text.data()), text.size());
    }
    return result;
#else
    return {argv, argv + argc};
#endif
}

ProcessResult runProcess(const std::filesystem::path& executable, const std::vector<std::string>& arguments,
    const std::filesystem::path& output, std::chrono::milliseconds timeout)
{
#ifdef _WIN32
    std::filesystem::create_directories(output);
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    Handle stdoutFile{CreateFileW((output / "stdout.log").c_str(), GENERIC_WRITE, FILE_SHARE_READ, &security,
        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr)};
    Handle stderrFile{CreateFileW((output / "stderr.log").c_str(), GENERIC_WRITE, FILE_SHARE_READ, &security,
        CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr)};
    Handle input{CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &security,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr)};
    if (stdoutFile.value == INVALID_HANDLE_VALUE || stderrFile.value == INVALID_HANDLE_VALUE || input.value == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("cannot open child output");
    }
    Handle job{CreateJobObjectW(nullptr, nullptr)};
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (!job.value || !SetInformationJobObject(job.value, JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
        throw std::runtime_error("cannot create process watchdog job");
    }
    std::wstring command = quote(executable.native());
    for (const auto& argument : arguments) { command += L" " + quote(wide(argument)); }
    STARTUPINFOEXW startup{};
    startup.StartupInfo.cb = sizeof(startup);
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdOutput = stdoutFile.value;
    startup.StartupInfo.hStdError = stderrFile.value;
    startup.StartupInfo.hStdInput = input.value;
    SIZE_T bytes = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
    std::vector<std::byte> attributes(bytes);
    startup.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
    if (!InitializeProcThreadAttributeList(startup.lpAttributeList, 1, 0, &bytes)) {
        throw std::runtime_error("cannot initialize child handle list");
    }
    struct AttributeCleanup {
        LPPROC_THREAD_ATTRIBUTE_LIST value;
        ~AttributeCleanup() { DeleteProcThreadAttributeList(value); }
    } cleanup{startup.lpAttributeList};
    HANDLE handles[]{stdoutFile.value, stderrFile.value, input.value};
    if (!UpdateProcThreadAttribute(startup.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
        handles, sizeof(handles), nullptr, nullptr)) { throw std::runtime_error("cannot restrict child handle inheritance"); }
    PROCESS_INFORMATION process{};
    if (!CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, TRUE,
        CREATE_NO_WINDOW | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr,
        &startup.StartupInfo, &process)) { throw std::runtime_error("CreateProcess failed: " + std::to_string(GetLastError())); }
    Handle processHandle{process.hProcess};
    Handle threadHandle{process.hThread};
    if (!AssignProcessToJobObject(job.value, process.hProcess)) {
        TerminateProcess(process.hProcess, 1);
        throw std::runtime_error("cannot isolate child in watchdog job");
    }
    if (ResumeThread(process.hThread) == DWORD(-1)) { throw std::runtime_error("cannot start child"); }
    const auto wait = WaitForSingleObject(process.hProcess, DWORD(timeout.count()));
    ProcessResult result;
    if (wait == WAIT_TIMEOUT) {
        result.timedOut = true;
        if (!TerminateJobObject(job.value, 124)) { throw std::runtime_error("cannot terminate timed-out child"); }
        if (WaitForSingleObject(process.hProcess, 5000) != WAIT_OBJECT_0) { throw std::runtime_error("child termination did not complete"); }
    } else if (wait != WAIT_OBJECT_0) { throw std::runtime_error("child wait failed"); }
    DWORD exitCode = 0;
    if (!GetExitCodeProcess(process.hProcess, &exitCode)) { throw std::runtime_error("cannot read child exit code"); }
    result.exitCode = exitCode;
    return result;
#else
    throw std::runtime_error("testbench process isolation currently requires Windows");
#endif
}

} // namespace metallic::tests::bench
