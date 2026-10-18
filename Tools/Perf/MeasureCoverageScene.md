# Coverage real-scene A/B

`MeasureCoverageScene.py` reuses the production `material_phase0_baseline` test, fixed OpenPBR LookDev mesh/HDRI, and the Coverage fixtures. Each process runs deferred-only and pathtrace-only graphs for control/simple/complex expressions. The reported full-graph graphics queue envelope includes graph waits, excludes CPU/presentation/loading/compiler/readback, and must not be described as editor wall-clock FPS.

Prepare paired Release executables with `PrepareCoverageSceneBinaries.py`. Legacy differs only in CPU Coverage bytecode transport; archive its exact patch/build identity. The runner temporarily installs the original legacy Coverage shader for the legacy executable and restores the initial native bytes even on ordinary failure. It refuses to overwrite an unexpected external shader edit. Abrupt interruption can be recovered using `recover --input <evidence>`; inspect the cooperative lock owner before manually removing a stale lock.

```powershell
python -B Tools/Perf/PrepareCoverageSceneBinaries.py --build-dir build-pass-stages-nrd --output build/coverage-scene-20261018/binaries-fixed --vsdevcmd E:/VisualStudio/Common7/Tools/VsDevCmd.bat
python -B Tools/Perf/MeasureCoverageScene.py run --legacy-exe build/coverage-scene-20261018/binaries-fixed/legacy/MetallicRHITests.exe --native-exe build/coverage-scene-20261018/binaries-fixed/native/MetallicRHITests.exe --legacy-shader build/coverage-native-20261017/baseline/Coverage.slang --build-evidence build/coverage-scene-20261018/binaries-fixed --cases LookDev/MaterialSystem/Coverage/Cases.json --output build/coverage-scene-20261018/ab --blocks 2 --receipts
python -B Tools/Perf/MeasureCoverageScene.py verify --input build/coverage-scene-20261018/ab
python -B Tools/Perf/MeasureCoverageScene.py report --input build/coverage-scene-20261018/ab --output build/coverage-scene-20261018/report
```

These paths describe the retained 2026-10-18 run. Select fresh output directories
for a new run; preparation and capture refuse existing evidence directories.
Reuse a compatible configured build and set TEMP/TMP to a writable local folder
when required by the environment. The common GPUScene Coverage classification
fix must be present in both builds. See
`Documentation/MaterialCoverageNativeSlang.md` for results and known regression
limitations.

Two ABBA blocks produce eight independent processes; each case has four paired process comparisons. Frames within a process are not independent samples. All timing frames are retained; no outlier rejection. The interval is Student t over paired percentage reductions. Zero crossing means inconclusive. HDR gate is fixed before capture: finite, exact same-arm repeats and exact cross-arm pixels. Differences remain visible with max absolute error and RMSE and disqualify performance acceptance.

Ordinary runs clear inherited METALLIC settings, reject injected Vulkan layers, disable validation and leave clocks unchanged. Required `--receipts` adds two separate diagnostic processes after timing; their timing data is excluded. Bound production Coverage raster-fragment/pathtrace shader module/entry/SPIR-V receipts must exist and custom workloads must differ between arms. Diagnostic recipe capture is not mixed into acceptance timestamps. Sources, shader variants, binaries/DLLs, asset dependencies and graph snapshots are hashed; raw logs, HDR and report JSON are sealed. Reports must be written outside the sealed evidence directory. A cooperative lock does not establish system-wide GPU exclusivity.

Generated charts show every process median, paired gains with CI, and all warmup/timing trajectories. Inspect the PNGs before delivery. Claims remain specific to the fixed mesh, environment, camera, resolution, sampling and expression; generic rendering gains require other workloads.
