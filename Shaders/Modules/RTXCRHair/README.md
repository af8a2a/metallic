# RTXCR Hair native Slang

`import RTXCRHair;` exposes `Metallic.RTXCR` types and functions. The entry module
uses `__include` for its six implementing files. Production Hair shaders no longer
textually include the upstream Hair HLSL headers or require an RTXCR SDK search path.

This is a native Slang 2026 port of NVIDIA RTXCR-Material commit
`75fab0e13f75ce2f06075a08db3fe45ef8e3c21f`. `Upstream.json` records original file
SHA-256 hashes and the symbol mapping; `LICENSE` and the original source copyright
headers retain NVIDIA's MIT terms. The snapshot in `External/RTXCR-Material`
remains unchanged, and GPU differential tests compile it independently.

The port covers Chiang, Separate Chiang, Far Field BCSDF, interactions, sampling,
absorption models and their math helpers. Include guards/macros became modules,
namespaced types, constants and public declarations. Scattering formulas and the
upstream sample rejection threshold are preserved. Subsurface scattering and
geometry conversion remain in their existing SDKs.

For Metallic materials use `RTXCRFiber`, not direct vendor calls:

```slang
import RTXCRFiber;
import FiberMaterial;
import MaterialProgram;
// A parameter provider implements IRTXCRHairParameterProvider.load(instance).
// program.evaluate(context, instance) -> closure.prepare(wo, Radiance)
// -> prepared.evalProjected(wi) / prepared.sampleWeighted(random4).
```

Prepared Chiang caches the interaction, frame and local outgoing direction once.
Its sample weight already divides the projected scattering numerator by the
sampled PDF. Do not multiply either projected evaluation or weighted sampling by
a Surface NdotL. Importance transport and axial/numerically near-axial outgoing directions are
rejected by the Metallic adapter; axial views have no defined azimuthal frame.
Low-level vendor functions retain their normalized-frame/nondegenerate-input
preconditions. Prepared data contains no resource handles or texture samplers.

`material_fiber_native_and_stages` compares evaluation, sampled direction,
reported PDF, numerator, event and Far Field diffuse extension with upstream on
4096 cases. It also tests the high-level Program/Closure/Prepared path, resource
evaluation count for 1/8 lights, instance offsets, projected lighting, axial
rejection and unsupported transport. This validates the port against upstream;
it is not a claim that the upstream models or MIS approximation are exact.
