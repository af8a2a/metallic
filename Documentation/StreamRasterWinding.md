# Reflected stream instance coverage

MiniZorah's courtyard floor contained missing faces with the camera at
`eye=(64.801514, 5.373773, 0.592118)`,
`center=(69.286797, -0.135649, -4.914468)`, FOV 60 degrees and reversed Z.
The issue was reproduced at 1074 x 510 after page requests had converged.

Disabling instance/meshlet HZB and frustum culling produced identical images.
Hardware-only rasterization retained the holes, as did reducing LOD pixel error
from 1.5 to 0.1. The affected geometry uses reflected instance transforms.
The stream rasterizer emitted the source triangle winding unchanged after a
negative-determinant transform, so its single-sided face rejection removed the
authored exterior surfaces.

`streamClusterTriangle` now reverses the emitted winding for reflected instances.
Hardware emission, triangle queueing, cluster binning and direct software
rasterization share this helper. Visibility IDs still identify the original
payload triangle, preserving material reconstruction and vertex attributes.
The normal-cone culling axis removes the cofactor determinant sign to agree with
the corrected raster winding. Material shading normals are unchanged.

The same camera now shows continuous floor coverage. HZB and frustum toggles
still produce identical images. The hardware/hybrid comparison differs at 526
of 547,740 pixels (0.096%, primarily triangle-edge ownership).

## Regression checks

```powershell
cmake --build build-release --target MetallicRhiTests MetallicGPUDrivenSample -j 8
build-release/tests/MetallicRhiTests.exe --rhi-no-validation --filter stream_reflected_winding
$env:METALLIC_TEST_MINIZORAH='1'
build-release/tests/MetallicRhiTests.exe --rhi-no-validation --filter minizorah_ground_coverage
```

The small fixture pairs ordinary and reflected single-sided triangles. It checks
symmetric front coverage, back-face rejection and identical visibility IDs with
hardware/software rasterization and normal-cone culling enabled/disabled.
The opt-in MiniZorah test records the reported camera, checks culling equivalence
and bounds hardware/hybrid differences. Captures and residency statistics are
written to ignored `rhi-test-output/Ground-*.png`, `.bin` and `GroundReport.json`.

The scene checks use `--rhi-no-validation` because this environment has a known
validation-layer/descriptor-heap driver crash during preview rendering; these
results do not establish a validation-layer-clean run.
