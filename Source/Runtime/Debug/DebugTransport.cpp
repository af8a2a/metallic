#include "Runtime/Debug/DebugTransport.h"

#include <array>
#include <future>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <sddl.h>
#endif

namespace metallic::debug {
#ifdef _WIN32
namespace {

struct Handle {
    HANDLE value = INVALID_HANDLE_VALUE;
    explicit Handle(HANDLE handle = INVALID_HANDLE_VALUE) : value(handle) {}
    ~Handle() { if (value && value != INVALID_HANDLE_VALUE) { CloseHandle(value); } }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    bool valid() const { return value && value != INVALID_HANDLE_VALUE; }
};

std::wstring pipeName(uint32_t pid)
{
    return L"\\\\.\\pipe\\Metallic.Debug." + std::to_wstring(pid);
}

DebugError windowsError(std::string operation)
{
    return {"TransportError", std::move(operation) + " (Win32 " + std::to_string(GetLastError()) + ")"};
}

bool awaitIo(HANDLE file, OVERLAPPED& operation, DWORD& transferred, std::stop_token stop,
    std::chrono::steady_clock::time_point deadline)
{
    while (!stop.stop_requested() && std::chrono::steady_clock::now() < deadline) {
        const DWORD result = WaitForSingleObject(operation.hEvent, 25);
        if (result == WAIT_OBJECT_0) { return GetOverlappedResult(file, &operation, &transferred, FALSE) != FALSE; }
        if (result != WAIT_TIMEOUT) { break; }
    }
    CancelIoEx(file, &operation);
    // The OVERLAPPED and its event must outlive cancellation completion.
    GetOverlappedResult(file, &operation, &transferred, TRUE);
    return false;
}

bool transfer(HANDLE file, void* data, uint32_t size, bool write, std::stop_token stop,
    std::chrono::steady_clock::time_point deadline)
{
    Handle event(CreateEventW(nullptr, TRUE, FALSE, nullptr));
    if (!event.valid()) { return false; }
    uint32_t offset = 0;
    while (offset < size) {
        ResetEvent(event.value);
        OVERLAPPED operation{};
        operation.hEvent = event.value;
        DWORD count = 0;
        const BOOL result = write
            ? WriteFile(file, static_cast<char*>(data) + offset, size - offset, &count, &operation)
            : ReadFile(file, static_cast<char*>(data) + offset, size - offset, &count, &operation);
        if (!result && (GetLastError() != ERROR_IO_PENDING || !awaitIo(file, operation, count, stop, deadline))) { return false; }
        if (!count) { return false; }
        offset += count;
    }
    return true;
}

bool send(HANDLE pipe, const DebugValue& value, std::stop_token stop, std::chrono::steady_clock::time_point deadline)
{
    std::string bytes = encodeLossless(value).dump(-1, ' ', false, DebugValue::error_handler_t::replace);
    if (bytes.size() > kDebugProtocolMaxBytes) {
        bytes = encodeLossless(debugErrorResponse(value.value("id", DebugValue(nullptr)), "ResponseTooLarge", "Response exceeds 1 MiB; use object.get pagination or artifact.read")).dump();
    }
    const uint32_t size = static_cast<uint32_t>(bytes.size());
    std::array<uint8_t, 4> prefix{uint8_t(size), uint8_t(size >> 8), uint8_t(size >> 16), uint8_t(size >> 24)};
    return transfer(pipe, prefix.data(), 4, true, stop, deadline) && transfer(pipe, bytes.data(), size, true, stop, deadline);
}

DebugResult<DebugValue> receive(HANDLE pipe, std::stop_token stop, std::chrono::steady_clock::time_point deadline)
{
    std::array<uint8_t, 4> prefix{};
    if (!transfer(pipe, prefix.data(), 4, false, stop, deadline)) { return std::unexpected(windowsError("Read length")); }
    const uint32_t size = uint32_t(prefix[0]) | (uint32_t(prefix[1]) << 8) | (uint32_t(prefix[2]) << 16) | (uint32_t(prefix[3]) << 24);
    if (!size || size > kDebugProtocolMaxBytes) { return std::unexpected(DebugError{"ProtocolError", "Message length exceeds 1 MiB"}); }
    std::string bytes(size, '\0');
    if (!transfer(pipe, bytes.data(), size, false, stop, deadline)) { return std::unexpected(windowsError("Read message")); }
    try {
        // Reject deeply nested input before constructing recursive JSON values.
        int depth = 0;
        auto parsed = DebugValue::parse(bytes, [&](int level, DebugValue::parse_event_t, DebugValue&) {
            depth = std::max(depth, level);
            if (depth > 64) { throw std::invalid_argument("JSON nesting exceeds 64"); }
            return true;
        });
        return decodeLossless(parsed);
    } catch (const std::exception& error) { return std::unexpected(DebugError{"ProtocolError", error.what()}); }
}

DebugResult<std::wstring> logonSid()
{
    HANDLE raw = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &raw)) { return std::unexpected(windowsError("OpenProcessToken")); }
    Handle token(raw);
    DWORD size = 0;
    GetTokenInformation(raw, TokenGroups, nullptr, 0, &size);
    std::vector<uint8_t> buffer(size);
    if (!GetTokenInformation(raw, TokenGroups, buffer.data(), size, &size)) { return std::unexpected(windowsError("TokenGroups")); }
    const auto groups = reinterpret_cast<TOKEN_GROUPS*>(buffer.data());
    for (DWORD i = 0; i < groups->GroupCount; ++i) {
        if ((groups->Groups[i].Attributes & SE_GROUP_LOGON_ID) == SE_GROUP_LOGON_ID) {
            LPWSTR sid = nullptr;
            if (!ConvertSidToStringSidW(groups->Groups[i].Sid, &sid)) { return std::unexpected(windowsError("ConvertSid")); }
            std::wstring result(sid); LocalFree(sid);
            return result;
        }
    }
    return std::unexpected(DebugError{"TransportError", "No interactive logon SID; debug server remains disabled"});
}

} // namespace
#endif

DebugServer::~DebugServer() { stop(); }

void DebugServer::stop()
{
    if (thread_.joinable()) { thread_.request_stop(); thread_.join(); }
}

DebugResult<void> DebugServer::start()
{
    if (thread_.joinable()) { return std::unexpected(DebugError{"InvalidState", "Server already started"}); }
#ifdef _WIN32
    auto sid = logonSid();
    if (!sid) { return std::unexpected(sid.error()); }
    const auto name = pipeName(GetCurrentProcessId());
    const std::wstring dacl = L"D:P(A;;GA;;;" + *sid + L")";
    FILETIME created{}, exited{}, kernel{}, user{};
    GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user);
    core_.setProcess(GetCurrentProcessId(), std::to_string((uint64_t(created.dwHighDateTime) << 32) | created.dwLowDateTime));
    std::promise<DebugResult<void>> initialized;
    auto ready = initialized.get_future();
    thread_ = std::jthread([this, name, dacl, promise = std::move(initialized)](std::stop_token stop) mutable {
        PSECURITY_DESCRIPTOR descriptor = nullptr;
        if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(dacl.c_str(), SDDL_REVISION_1, &descriptor, nullptr)) {
            promise.set_value(std::unexpected(windowsError("Create DACL"))); return;
        }
        SECURITY_ATTRIBUTES attributes{sizeof(SECURITY_ATTRIBUTES), descriptor, FALSE};
        Handle pipe(CreateNamedPipeW(name.c_str(), PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED | FILE_FLAG_FIRST_PIPE_INSTANCE,
            PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS, 1, 65536, 65536, 0, &attributes));
        LocalFree(descriptor);
        if (!pipe.valid()) { promise.set_value(std::unexpected(windowsError("CreateNamedPipe"))); return; }
        promise.set_value({});
        while (!stop.stop_requested()) {
            Handle event(CreateEventW(nullptr, TRUE, FALSE, nullptr));
            if (!event.valid()) { break; }
            OVERLAPPED operation{}; operation.hEvent = event.value;
            DWORD ignored = 0;
            bool connected = ConnectNamedPipe(pipe.value, &operation) != FALSE;
            if (!connected) {
                const DWORD error = GetLastError();
                connected = error == ERROR_PIPE_CONNECTED || (error == ERROR_IO_PENDING &&
                    awaitIo(pipe.value, operation, ignored, stop, std::chrono::steady_clock::time_point::max()));
            }
            if (!connected) {
                if (!stop.stop_requested() && GetLastError() == ERROR_NO_DATA) { DisconnectNamedPipe(pipe.value); continue; }
                break;
            }
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            const auto request = receive(pipe.value, stop, deadline);
            const bool sent = request ? send(pipe.value, core_.dispatch(*request), stop, deadline)
                : send(pipe.value, debugErrorResponse(nullptr, request.error().code, request.error().message), stop, deadline);
            uint8_t acknowledgement = 0;
            if (sent) { transfer(pipe.value, &acknowledgement, 1, false, stop, deadline); }
            DisconnectNamedPipe(pipe.value);
        }
    });
    auto result = ready.get();
    if (!result) { stop(); }
    return result;
#else
    return std::unexpected(DebugError{"Unsupported", "Named Pipe transport requires Windows"});
#endif
}

DebugResult<DebugValue> debugRequest(uint32_t pid, const DebugValue& request, uint32_t timeoutMs)
{
#ifdef _WIN32
    if (encodeLossless(request).dump().size() > kDebugProtocolMaxBytes) {
        return std::unexpected(DebugError{"ProtocolError", "Request exceeds 1 MiB"});
    }
    const auto name = pipeName(pid);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
    HANDLE raw = INVALID_HANDLE_VALUE;
    do {
        raw = CreateFileW(name.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
            FILE_FLAG_OVERLAPPED | SECURITY_SQOS_PRESENT | SECURITY_IDENTIFICATION, nullptr);
        if (raw != INVALID_HANDLE_VALUE) { break; }
        if (GetLastError() != ERROR_PIPE_BUSY) { return std::unexpected(windowsError("Connect")); }
        WaitNamedPipeW(name.c_str(), 25);
    } while (std::chrono::steady_clock::now() < deadline);
    Handle pipe(raw);
    if (!pipe.valid()) { return std::unexpected(windowsError("Connect timeout")); }
    if (!send(pipe.value, request, {}, deadline)) { return std::unexpected(windowsError("Write request")); }
    auto response = receive(pipe.value, {}, deadline);
    uint8_t acknowledgement = 1;
    if (response) { transfer(pipe.value, &acknowledgement, 1, true, {}, deadline); }
    return response;
#else
    return std::unexpected(DebugError{"Unsupported", "Named Pipe transport requires Windows"});
#endif
}

std::vector<uint32_t> debugProcesses()
{
    std::vector<uint32_t> result;
#ifdef _WIN32
    WIN32_FIND_DATAW data{};
    HANDLE search = FindFirstFileW(L"\\\\.\\pipe\\Metallic.Debug.*", &data);
    if (search != INVALID_HANDLE_VALUE) {
        do {
            const std::wstring name(data.cFileName);
            const auto position = name.rfind(L'.');
            if (position != std::wstring::npos) {
                try { result.push_back(static_cast<uint32_t>(std::stoul(name.substr(position + 1)))); } catch (...) {}
            }
        } while (FindNextFileW(search, &data));
        FindClose(search);
    }
#endif
    return result;
}

} // namespace metallic::debug
