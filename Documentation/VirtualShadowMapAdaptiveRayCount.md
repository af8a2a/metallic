# SMRT adaptive ray count

SMRT projection uses adaptive ray count by default on devices supporting the
existing compute subgroup ballot/arithmetic capability. The experimental
`benchmarkAdaptiveRayCount` property and its settings toggle have been removed.
Unsupported devices retain fixed-ray projection. This does not change Morton
mapping, Mark, the configured ray sequence, or page admission.

After the first ray, an integer subgroup reduction tests whether all participating
receivers missed. After at least two rays, it tests whether every participating
receiver has accumulated zero misses. A successful test exits only if rays remain;
visibility is normalized by the actual traced count. These are heuristic light/umbra
assumptions, not equivalence to the fixed-ray estimator. The existing subgroup
ballot/arithmetic capability gates both per-face and one-pass permutations.

One-pass membership and receiver/domain rejection determine the participating lanes
for each light. Per-face workgroup barriers remain outside the receiver helper.
ARC is fixed by the device capability, rather than a mutable history setting.
PCF and Hard do not execute ARC.

ARC introduced the following workload counters in the historical v2 schema.
The current layout, subsequent extensions, version fields and historical-package
reading rules are defined centrally in the [diagnostic ABI](VirtualShadowMapsGPUProfile.md#诊断-abi当前布局与历史读取):

- `projectionAdaptiveLitReceivers`: receiver/light evaluations exiting after one ray.
- `projectionAdaptiveUmbraReceivers`: evaluations exiting with zero accumulated misses
  after at least two rays.
- `projectionAdaptiveSavedRays`: omitted iterations relative to configured ray count.

Only exits with remaining rays are counted. A 2-ray configuration therefore records
no umbra savings. These counters count receiver/light evaluations, not waves. The
public statistics remain unchanged; private workload and Build offsets use the
shared size constants documented in the diagnostic ABI, not fixed historical lengths.

Historical 2026-11-01 Sponza validation and plots (workload v2 / Build v3) are in
`build/vsm-arc-20261101/Report.md`: four diagnostic processes (2-ray/8-ray A/B) and
one 2-ray ABBA block. The measured 2-ray Projection improvement was about 1.9%
(0.019 ms), with localized nonzero image differences. The 8-ray test verified
substantial ray savings and actual umbra exits, but did not measure ordinary 8-ray
performance. The user accepted the measured image differences and authorized
default adoption; this is not a new statistical performance confirmation.

Default-path validation is recorded separately in
`build/vsm-arc-default-20261101/Report.md`, using a graph without the experimental
property and the accepted 2-ray candidate as its image and shader reference.
