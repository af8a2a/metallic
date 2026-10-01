#include "Runtime/Render/Profiling/NsightReplayProtocol.h"

#include <Windows.h>
#include <filesystem>

namespace {

DWORD collect(const std::filesystem::path& executable, std::wstring command,
    const std::wstring& logPath, const std::wstring& directory)
{
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE log = CreateFileW(logPath.c_str(), GENERIC_WRITE, FILE_SHARE_READ,
        &security, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (log == INVALID_HANDLE_VALUE) { return GetLastError(); }
    HANDLE input = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
        &security, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    DWORD result = ERROR_PROCESS_ABORTED;
    if (input != INVALID_HANDLE_VALUE && job != nullptr &&
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits))) {
        STARTUPINFOW startup{};
        startup.cb = sizeof(startup);
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = input;
        startup.hStdOutput = log;
        startup.hStdError = log;
        PROCESS_INFORMATION process{};
        if (CreateProcessW(executable.c_str(), command.data(), nullptr, nullptr, TRUE,
                CREATE_NO_WINDOW | CREATE_SUSPENDED, nullptr, directory.c_str(), &startup, &process)) {
            if (AssignProcessToJobObject(job, process.hProcess) && ResumeThread(process.hThread) != DWORD(-1)) {
                if (WaitForSingleObject(process.hProcess, 150000) == WAIT_OBJECT_0) {
                    GetExitCodeProcess(process.hProcess, &result);
                } else {
                    result = WAIT_TIMEOUT;
                }
            }
            TerminateJobObject(job, result);
            TerminateProcess(process.hProcess, result); // Also covers failed job assignment.
            CloseHandle(process.hThread);
            CloseHandle(process.hProcess);
        } else {
            result = GetLastError();
        }
    }
    if (job != nullptr) { CloseHandle(job); }
    if (input != INVALID_HANDLE_VALUE) { CloseHandle(input); }
    CloseHandle(log);
    return result;
}

} // namespace

int wmain(int argc, wchar_t** argv)
{
    if (argc != 4 || std::wstring_view(argv[1]) != L"--nsight-replay-worker") { return 1; }
    const std::filesystem::path mailbox(argv[2]);
    const std::filesystem::path executable(argv[3]);
    for (;;) {
        std::error_code error;
        if (!std::filesystem::exists(mailbox / "request.bin", error)) { Sleep(50); continue; }
        std::ifstream request(mailbox / "request.bin", std::ios::binary);
        std::wstring command, logPath, directory;
        const bool valid = metallic::render::profiling::readReplayString(request, command) &&
            metallic::render::profiling::readReplayString(request, logPath) &&
            metallic::render::profiling::readReplayString(request, directory);
        request.close();
        if (!std::filesystem::remove(mailbox / "request.bin", error) || error) { return 1; }
        const DWORD result = valid ? collect(executable, std::move(command), logPath, directory) : ERROR_INVALID_DATA;
        std::ofstream output(mailbox / "result.tmp");
        output << result;
        output.close();
        if (!output) { return 1; }
        std::filesystem::rename(mailbox / "result.tmp", mailbox / "result.txt", error);
        if (error) { return 1; }
    }
}
