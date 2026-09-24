# Streamline Debug panel

Open **Window > Streamline Debug**, or set `METALLIC_STREAMLINE_DEBUG=1` before launching Metallic.

The panel reads a value-only snapshot under the Streamline mutex. It does not call SDK evaluation APIs, retain texture pointers, or introduce GPU synchronization. Snapshots reset with the Streamline device session.

- Overview: compiled SDK version, Vulkan initialization/device registration and descriptor-heap workaround.
- DLSS SR / RR: device support, last attempt age and frame, attempt/success counts, error text, mode, render/output extents, CPU evaluation wall time, camera/history inputs and supplied resource bindings/formats.
- Reflex: requested Off/On/Boost mode, frame interval (`0` disables its cap), suspension state and cached latency report. Option changes apply at the next frame begin. Render latency excludes display latency; reports refresh at the existing 60-frame cadence.

Feature support does not mean a pass is currently running. DLSS sections deliberately label all evaluation data as the last recorded attempt. Resource rows describe supplied inputs; failed validation may prevent tagging/evaluation. CPU evaluation time is not GPU execution time. Use Profiler for GPU timings and NVML Monitor for GPU/system memory information.

This is Metallic's own ImGui panel, inspired by [NVIDIA's Streamline ImGui debugging guide](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/Debugging%20-%20SL%20ImGUI%20%28Realtime%20Data%20Inspection%29.md). It does not load `sl.imgui` or change the deployed SDK binaries. NVIDIA's overlay requires non-production Streamline libraries. Its documented DLSS-G buffer visualizer is outside this panel because Metallic does not integrate DLSS Frame Generation.

## Validation

Build with `cmake --build build-dev --target Metallic`. For an automated panel/snapshot smoke check in PowerShell:

```powershell
$env:METALLIC_SMOKE_TEST_STREAMLINE_DEBUG = '1'
$env:METALLIC_SMOKE_TEST_FRAMES = '3'
$env:METALLIC_SMOKE_TEST_SAMPLE = 'pathtracing-sample-dlss-sr'
.\build-dev\Source\Metallic.exe --smoke-test
```

Repeat with `pathtracing-sample-dlss-rr` and `material-visualization-abeautiful-game` (session without Streamline initialization). The check verifies that the panel was drawn and that evaluated features publish successful, populated snapshots. Unsupported hardware cannot pass the SR/RR checks. An SDK-disabled build displays an unavailable message.