"""Windows process resource samples for CPU-job experiments (never GPU memory)."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import json
import subprocess
import time


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


class _ProcessEntry(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", wintypes.LONG),
        ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260),
    ]


def _filetime_ns(value: wintypes.FILETIME) -> int:
    return ((value.dwHighDateTime << 32) | value.dwLowDateTime) * 100


class ProcessResources:
    """Own an OS query handle; querying cannot extend the child's running lifetime."""

    def __init__(self, process_id: int):
        if os.name != "nt":
            raise OSError("Production CPU-job process sampling currently requires Windows")
        self.pid = process_id
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.psapi = ctypes.WinDLL("psapi", use_last_error=True)
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
        self.kernel.GetProcessTimes.restype = wintypes.BOOL
        self.psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_ProcessMemoryCounters), wintypes.DWORD]
        self.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        self.kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
        self.kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        self.kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)]
        self.kernel.Process32FirstW.restype = wintypes.BOOL
        self.kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_ProcessEntry)]
        self.kernel.Process32NextW.restype = wintypes.BOOL
        # Query information + VM read, as required by GetProcessMemoryInfo.
        self.handle = self.kernel.OpenProcess(0x0400 | 0x0010, False, process_id)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _threads(self) -> tuple[int | None, str | None]:
        snapshot = self.kernel.CreateToolhelp32Snapshot(0x00000002, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            return None, f"Process snapshot failed: Windows error {ctypes.get_last_error()}"
        try:
            item = _ProcessEntry()
            item.dwSize = ctypes.sizeof(item)
            found = self.kernel.Process32FirstW(snapshot, ctypes.byref(item))
            while found:
                if item.th32ProcessID == self.pid:
                    return int(item.cntThreads), None
                found = self.kernel.Process32NextW(snapshot, ctypes.byref(item))
            return None, "Process was absent from the Toolhelp process snapshot"
        finally:
            self.kernel.CloseHandle(snapshot)

    def sample(self, origin_ns: int) -> dict:
        started = time.perf_counter_ns()
        creation, exit_time, kernel, user = (wintypes.FILETIME() for _ in range(4))
        missing = []
        if not self.kernel.GetProcessTimes(self.handle, ctypes.byref(creation), ctypes.byref(exit_time),
                                           ctypes.byref(kernel), ctypes.byref(user)):
            kernel_ns = user_ns = None
            lifetime_ns = None
            missing.append(f"GetProcessTimes failed: Windows error {ctypes.get_last_error()}")
        else:
            kernel_ns, user_ns = _filetime_ns(kernel), _filetime_ns(user)
            exit_ns = _filetime_ns(exit_time)
            lifetime_ns = exit_ns - _filetime_ns(creation) if exit_ns else None
        memory = _ProcessMemoryCounters()
        memory.cb = ctypes.sizeof(memory)
        if not self.psapi.GetProcessMemoryInfo(self.handle, ctypes.byref(memory), memory.cb):
            private = peak_commit = working_set = peak_working_set = None
            missing.append(f"GetProcessMemoryInfo failed: Windows error {ctypes.get_last_error()}")
        else:
            private, peak_commit = int(memory.PrivateUsage), int(memory.PeakPagefileUsage)
            working_set, peak_working_set = int(memory.WorkingSetSize), int(memory.PeakWorkingSetSize)
        threads, thread_error = self._threads()
        if thread_error:
            missing.append(thread_error)
        completed = time.perf_counter_ns()
        return {
            "elapsedNs": started - origin_ns, "processId": self.pid,
            "cpuNs": kernel_ns + user_ns if kernel_ns is not None else None,
            "kernelCpuNs": kernel_ns, "userCpuNs": user_ns,
            "processLifetimeNs": lifetime_ns,
            "privateBytes": private, "peakPagefileBytes": peak_commit,
            "workingSetBytes": working_set, "peakWorkingSetBytes": peak_working_set,
            "threadCount": threads, "queryWallNs": completed - started, "missing": missing,
        }


def summarize_process(samples: list[dict], process_wall_ns: int) -> dict:
    """High-water counters and sampled peaks are intentionally distinct."""
    def maximum(field):
        values = [row[field] for row in samples if row.get(field) is not None]
        return max(values) if values else None

    cpu_ns = maximum("cpuNs")
    lifetime_ns = maximum("processLifetimeNs")
    actual_wall_ns = lifetime_ns if lifetime_ns is not None else process_wall_ns
    successful_memory = [row for row in samples if row.get("privateBytes") is not None]
    last_memory = successful_memory[-1]["elapsedNs"] if successful_memory else None
    return {
        "samples": len(samples),
        "processWallNs": actual_wall_ns,
        "observedLaunchToReapWallNs": process_wall_ns,
        "processWallSource": "GetProcessTimes creation/exit FILETIME" if lifetime_ns is not None else "Observer launch-to-reap interval; includes polling delay",
        "processCpuNs": cpu_ns,
        "averageCpuCores": cpu_ns / actual_wall_ns if cpu_ns is not None and actual_wall_ns > 0 else None,
        "lifetimePeakPrivateBytes": maximum("peakPagefileBytes"),
        "lifetimePeakWorkingSetBytes": maximum("peakWorkingSetBytes"),
        "sampledPeakPrivateBytes": maximum("privateBytes"),
        "sampledPeakWorkingSetBytes": maximum("workingSetBytes"),
        "sampledPeakThreads": maximum("threadCount"),
        "lastMemorySampleBeforeExitMs": (process_wall_ns - last_memory) * 1e-6 if last_memory is not None else None,
        "samplerQueryWallNs": sum(row.get("queryWallNs", 0) for row in samples),
        "missing": sorted({message for row in samples for message in row.get("missing", [])}),
        "definitions": {
            "lifetimePeakPrivateBytes": "Observed Windows PeakPagefileUsage (process commit high-water), not GPU memory; high-water extends only through last successful sample.",
            "lifetimePeakWorkingSetBytes": "Observed Windows PeakWorkingSetSize; high-water extends only through last successful sample.",
            "sampledPeakThreads": "Maximum cntThreads from process snapshots, not an exact lifetime thread high-water.",
            "averageCpuCores": "GetProcessTimes kernel+user / process launch-to-exit wall; no logical-CPU normalization.",
            "sampling": "Sub-interval transients may escape current-value sampling; process lifetime OS peak counters retain earlier peaks until queried.",
        },
    }


def monitor_command(command, *, cwd, environment, console_path, resources_path,
                    timeout_seconds=1800, sample_ms=100) -> tuple[int, dict]:
    """Run one child with resource monitoring; the caller owns its workload protocol."""
    samples = []
    origin = time.perf_counter_ns()
    with console_path.open("w", encoding="utf-8") as console, \
            resources_path.open("w", encoding="utf-8", newline="\n") as resources:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=console,
                                   stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            with ProcessResources(process.pid) as monitor:
                next_sample = origin
                while True:
                    if (time.perf_counter_ns() - origin) * 1e-9 > timeout_seconds:
                        raise TimeoutError(f"Child process exceeded {timeout_seconds}s")
                    row = monitor.sample(origin)
                    samples.append(row)
                    resources.write(json.dumps(row, separators=(",", ":")) + "\n")
                    resources.flush()
                    if process.poll() is not None:
                        final = monitor.sample(origin)
                        samples.append(final)
                        resources.write(json.dumps(final, separators=(",", ":")) + "\n")
                        break
                    next_sample += int(sample_ms * 1e6)
                    time.sleep(max(0, (next_sample - time.perf_counter_ns()) * 1e-9))
            exit_code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=30)
            raise
    return exit_code, summarize_process(samples, time.perf_counter_ns() - origin)
