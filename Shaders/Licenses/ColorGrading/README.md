# Color grading attribution

The Slang modules in `Shaders/Modules/ColorGrading` and CPU table implementation
`Source/Runtime/Render/Core/ACESTables.cpp` adapt algorithms from Unreal Engine
5.7.4 (compatible changelist 47537391), Academy ACES 1.3/2.0 and OpenColorIO 2.4.1.
Their applicable licenses and notices remain in this directory.

Original UE entry points: `TonemapCommon.ush`, `PostProcessCombineLUTs.usf`,
`PostProcessCommon.ush`, `GammaCorrectionCommon.ush` and the `ACES/` shader library;
CPU `Renderer/Private/PostProcess/ACESUtils.cpp` and `Core/Private/ColorManagement/ColorSpace.cpp`.

The integration specializes the used forward display paths, removes engine glue
and unused functions, passes parameters explicitly and composes a native 3D LUT.
It neither embeds the original engine files nor requires an engine installation.

`UNREAL_LICENSE.md` is the original checkout notice. `ACES_v1.3_LICENSE.txt` is
the bundled Academy notice. `ACES2_LICENSE.md` comes from the ACES repository's
dev-branch LICENSE.md referenced by UE's ACES2 third-party notice. `OpenColorIO.txt`
retains the BSD notice accompanying the CPU table algorithms.
