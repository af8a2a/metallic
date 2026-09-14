#include "TracyLauncherConfig.h"

#include <windows.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

namespace fs = std::filesystem;

// Quote arguments using the Windows C runtime rules, including backslashes
// before a quote or the closing quote. No command shell is involved.
std::wstring quoteArgument(std::wstring_view argument)
{
    std::wstring quoted = L"\"";
    size_t backslashes = 0;
    for (const wchar_t character : argument) {
        if (character == L'\\') {
            ++backslashes;
            continue;
        }
        quoted.append(backslashes * (character == L'"' ? 2 : 1), L'\\');
        backslashes = 0;
        if (character == L'"') {
            quoted += L'\\';
        }
        quoted += character;
    }
    quoted.append(backslashes * 2, L'\\');
    quoted += L'"';
    return quoted;
}

void requireFile(const fs::path& path, const char* reason)
{
    if (!fs::is_regular_file(path)) {
        std::wcerr << L"Missing file: " << path.wstring() << L'\n';
        throw std::runtime_error(reason);
    }
}

class ChildProcess
{
public:
    ChildProcess(const fs::path& executable, const std::vector<std::wstring>& arguments)
    {
        std::wstring commandLine = quoteArgument(executable.wstring());
        for (const auto& argument : arguments) {
            commandLine += L" " + quoteArgument(argument);
        }
        STARTUPINFOW startup{};
        startup.cb = sizeof(startup);
        PROCESS_INFORMATION process{};
        if (!CreateProcessW(executable.c_str(), commandLine.data(), nullptr, nullptr,
                FALSE, 0, nullptr, kRepositoryRoot, &startup, &process)) {
            throw std::system_error(GetLastError(), std::system_category(), "Cannot launch application");
        }
        handle_ = process.hProcess;
        CloseHandle(process.hThread);
    }

    ~ChildProcess()
    {
        CloseHandle(handle_);
    }

    ChildProcess(const ChildProcess&) = delete;
    ChildProcess& operator=(const ChildProcess&) = delete;

    int wait() const
    {
        if (WaitForSingleObject(handle_, INFINITE) != WAIT_OBJECT_0) {
            throw std::system_error(GetLastError(), std::system_category(), "Cannot wait for Viewer");
        }
        DWORD code = 0;
        if (!GetExitCodeProcess(handle_, &code)) {
            throw std::system_error(GetLastError(), std::system_category(), "Cannot read Viewer exit code");
        }
        return static_cast<int>(code);
    }

private:
    HANDLE handle_ = nullptr;
};

int run(int argc, wchar_t** argv)
{
    std::wstring address = L"127.0.0.1";
    std::wstring port = L"8086";
    fs::path capture;
    bool withMetallic = false;
    bool connectionOptions = false;
    for (int index = 1; index < argc; ++index) {
        const std::wstring_view argument = argv[index];
        const auto value = [&]() -> std::wstring {
            if (++index >= argc || std::wstring_view(argv[index]).empty()) {
                throw std::runtime_error("Option requires a non-empty value");
            }
            return argv[index];
        };
        if (argument == L"--help" || argument == L"-h") {
            std::wcout << L"MetallicTracy [--address HOST] [--port PORT] [--with-metallic]\n"
                          L"MetallicTracy --capture FILE.tracy\n"
                          L"MetallicTracy FILE.tracy\n\n"
                          L"Defaults to 127.0.0.1:8086. --with-metallic starts a new editor\n"
                          L"from this CMake configuration. Relative capture paths use the\n"
                          L"current working directory. The launcher waits for Viewer to exit.\n";
            return 0;
        } else if (argument == L"--address" || argument == L"-a") {
            address = value();
            connectionOptions = true;
        } else if (argument == L"--port" || argument == L"-p") {
            port = value();
            connectionOptions = true;
        } else if (argument == L"--with-metallic") {
            withMetallic = true;
            connectionOptions = true;
        } else if (argument == L"--capture") {
            if (!capture.empty()) {
                throw std::runtime_error("Specify only one capture file");
            }
            capture = value();
        } else if (!argument.empty() && argument.front() != L'-' && capture.empty()) {
            capture = argument;
        } else {
            throw std::runtime_error("Unknown argument; use --help for usage");
        }
    }
    if (port.empty() || port.size() > 5 || port.find_first_not_of(L"0123456789") != std::wstring::npos ||
        std::stoul(port) == 0 || std::stoul(port) > 65535) {
        throw std::runtime_error("Port must be between 1 and 65535");
    }
    if (!capture.empty() && connectionOptions) {
        throw std::runtime_error("Capture mode cannot be combined with connection options");
    }

    fs::path viewer = kViewerOverride;
    if (viewer.empty()) {
        viewer = fs::path(kRepositoryRoot) / "build-tracy-viewer/tracy-profiler.exe";
        if (!fs::is_regular_file(viewer)) {
            viewer = fs::path(kRepositoryRoot) / "build-tracy-viewer/Release/tracy-profiler.exe";
        }
    }
    viewer = fs::absolute(viewer);
    requireFile(viewer, "Build Tracy Viewer first or set METALLIC_TRACY_VIEWER in CMake; see Documentation/TracyGpuProfiling.md");
    std::vector<std::wstring> viewerArguments;
    if (!capture.empty()) {
        capture = fs::absolute(capture);
        requireFile(capture, "Capture file does not exist");
        viewerArguments = {capture.wstring()};
    } else {
        viewerArguments = {L"-a", address, L"-p", port};
    }
    if (withMetallic) {
        requireFile(kMetallicExecutable, "Build the Metallic target for this configuration first");
        const ChildProcess editor(kMetallicExecutable, {});
    }
    std::wcout << L"Launching Tracy Viewer: " << viewer.wstring() << std::endl;
    const ChildProcess profiler(viewer, viewerArguments);
    return profiler.wait();
}

} // namespace

int wmain(int argc, wchar_t** argv)
{
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "MetallicTracy: " << error.what() << '\n';
        return 1;
    }
}
