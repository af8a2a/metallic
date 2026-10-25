# Metallic ShaderMathHarness

Certified-error validation for quantized numerical models used in shader math
(OpenPBR BSDF optimization work). The harness is both a **proof tool** and a
**test tool**: it proves, under an explicitly stated floating point execution
model, that an approximation satisfies quantified error bounds over its whole
declared domain — and separately tests the bit-exact host simulation of that
model, up to full exhaustion of the legal binary32 inputs.

Phase-1 pilot: the small-`tau` transmittance polynomial
`exp(-tau) ~ 1 - tau + tau^2/2 - tau^3/6 + tau^4/24` on `tau in [0, 1/4]`,
evaluated as a binary32 Horner chain of four single-rounding FMAs.

## What a certificate here means — and what it does not

`PROVED_UNDER_MODEL` means: for every **real** input in the declared domain
(hence every legal binary32 input), the stated error bound holds for the
**stated operation graph**: binary32, one correctly rounded FMA per Horner
step (round-to-nearest-even), no reassociation, subnormal inputs preserved,
coefficients fixed to the declared bit patterns.

It does **not** mean:

- that any particular GPU/compiler executed this graph (see GPU validation,
  phase 2 — a Slang module and a bit-exact expectation manifest are emitted);
- that the polynomial is optimal, only that it meets the contract budget;
- that floating point monotonicity holds (the real polynomial's monotonicity
  is proved; the binary32 chain's is only observed, and the exhaustive sweep
  reports any adjacent-input violations it finds — the pilot has 2886, which
  is informational, not a contract requirement).

## Error chain and evidence levels

Following the decomposition `E_total <= E_approximation + E_coefficient +
E_evaluation`, the certification pass produces a rigorous interval for each
term over the whole continuous domain:

| Term | Compared quantities | Pilot result |
|---|---|---|
| Approximation | `P(c)` vs `exp(-x)` | sup in `[-7.45e-9, 7.8123e-6]` (true sup `1595/2048 - e^-1/4 = 7.8107e-6`; analytic Taylor bound `1/122880 = 8.138e-6`) |
| Coefficients | `P(c~)` vs `P(c)` | sup `7.322e-11` (true sup `15/206158430208 = 7.2766e-11`; document's triangle bound `17/206158430208 = 8.2468e-11`) |
| Evaluation | `fl(P)` vs `P(c~)` | gamma bound `3.06134e-7` with `gamma_4 = 1/(2^22-1)` (document: `< 3.062e-7`) |
| **Total** | `fl(P)` vs `exp(-x)` | **`<= 8.1184e-6` vs requirement `8.5e-6` — PROVED_UNDER_MODEL** |

Observed (exhaustive sweep of all 1,048,576,001 legal binary32 inputs in
[0, 1/4], ~3 s on 16 threads): max error `7.8480e-6` at `tau = 0x3E7FFEF3`,
re-verified against the rigorous interval oracle. Observed maxima are
cross-checked against the certified bounds; a violation fails the run rather
than hiding an engine bug.

Evidence levels used in reports: `PROVED_UNDER_MODEL`, `CERTIFIED_BOUND`,
`EXHAUSTIVE_MODEL_SWEEP`, `SAMPLED_MODEL_SWEEP`, `INCONCLUSIVE`, `FAILED`,
`NOT_RUN`. An exhaustive host sweep is not a proof about reals; a sampled GPU
run is never reported as proved.

Certified vs observed are always reported separately. "Inconclusive" means
the interval machinery could not decide within its subdivision budget — it is
not a counterexample; `FAILED` requires either a certified lower bound above
the requirement or a reproduced failing input.

## Soundness of the built-in interval engine

The harness ships its own rigorous numerics instead of depending on MPFR:

- All interval operations are computed in default round-to-nearest `double`
  and each endpoint is widened by two ulps toward -/+ infinity. This stays
  sound without switching hardware rounding modes as long as the binary is
  built without fast-math reassociation (repository default) and every
  compiled operation is a single IEEE-754 rounded step; compiler contraction
  of `a*b + c` into one FMA only removes a rounding the inflation already
  accounts for.
- `exp` references use a directed Taylor series with a certified remainder
  (arguments limited to `[-0.02, 0.26]`), which is why the pilot reference
  `exp(-tau)` works; broader functions should move to an MPFR-based oracle
  (phase 2 integration point: `ExpNegativeSmall` in `src/Numeric.cpp`).
- Domain-wide hulls use the **mean-value form**
  `g(x) in g(m) + g'(B) * (x - m)` with adaptive subdivision, which keeps
  cancellation-heavy differences (like `P(c~) - P(c)`, magnitude `1e-11`)
  provable at realistic box counts.
- The gamma evaluation bound assumes no subnormal intermediates; the binary32
  interval chain certifies every intermediate range first, and the verdict
  downgrades if that premise fails.
- `std::fma` is assumed to be a single-rounding operation (an IEEE-754
  requirement); the sweep cross-check `observed eval error <= gamma bound`
  doubles as a canary for that assumption.
- `/arch:AVX2` (default on MSVC, disable via
  `METALLIC_SHADER_MATH_HARNESS_AVX2=OFF`) only lets the compiler inline
  `std::fma` as `vfmadd`; correctness does not depend on it.

Known engine floor: near `tau = 0` the true error (~`tau^5/120`) drops below
the interval noise floor (~`1e-13`), so global one-sidedness of the error
sign is not certifiable by direct differences. The report states precisely
what is proved: non-negative on `[0.0625, 1/4]`, and `>= -7.45e-9` on the
whole domain. This distinction — proved / observed / undecidable — is the
point of the harness.

## Layout

```
shader-math-harness/
  contracts/AttenuationExpSmallTauV1.yaml   versioned numeric contract
  src/
    Numeric.{hpp,cpp}   intervals, rigorous exp, binary32 helpers, hashing
    Contract.{hpp,cpp}  strict YAML-subset loader (unknown keys rejected)
    Model.{hpp,cpp}     candidate graph: bit-exact FMA chain + interval forms
    Certify.{hpp,cpp}   MV-form hulls, gamma bound, range proofs, verdicts
    Sweep.{hpp,cpp}     multithreaded exhaustive/sampled binary32 sweep
    Export.{hpp,cpp}    Sollya/Gappa scripts, Slang module, GPU manifest
    Report.{hpp,cpp}    JSON report + console summary + consistency checks
    Main.cpp            CLI; embedded contract for --mode self-test
```

## Build and run

Standalone (no engine dependencies, MSVC or Clang/GCC):

```powershell
# from a VS x64 developer prompt
cmake -S shader-math-harness -B build-shader-math-harness -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-shader-math-harness
.\build-shader-math-harness\MetallicShaderMathHarness.exe              # full validation
.\build-shader-math-harness\MetallicShaderMathHarness.exe --mode self-test
ctest --test-dir build-shader-math-harness --output-on-failure
```

Or inside the repository build, off by default:

```powershell
cmake --preset metallic-dev -D METALLIC_BUILD_SHADER_MATH_HARNESS=ON
cmake --build --preset metallic-dev --target MetallicShaderMathHarness
```

Artifacts land in `shader-math-harness-output/` (gitignored): `report.json`,
`exports/*.sollya`, `exports/*.gappa`, `exports/AttenuationExpSmallTauV1.slang`,
and `exports/gpu-manifest.txt`.

Useful flags: `--sample 1000000` (sampled sweep instead of exhaustive),
`--max-exhaustive-inputs N` (auto-downgrade above N), `--threads N`,
`--worst-k N`, `--run-sollya` / `--run-gappa` (invoke external tools when on
PATH), `--quiet`. Exit code 0 only when all requirement verdicts are
`PROVED_UNDER_MODEL` and all consistency checks pass.

## Contract schema (phase 1)

See `contracts/AttenuationExpSmallTauV1.yaml`. The loader rejects unknown
keys, unsupported forms, and malformed numbers. Supported today:
`reference.expression = exp(-tau)`, `candidate.form = horner_degree_4` with
five coefficient bit patterns, optional `ideal_coefficients` (exact rationals
used to split approximation error from coefficient-quantization error),
`postprocess = clamp_0_1|none`, proof model `binary32_fma` + `nearest_even` +
`preserve` + `forbidden`, and requirements on absolute error, output range,
and the zero endpoint. `outside_domain.action` documents the intended
fallback but is not enforced by the harness yet.

## External provers and the GPU (phase 2)

- **Sollya**: the generated script asks `supnorm` for a certified supremum of
  the approximation error. Without Sollya on PATH the artifact is reported as
  NOT executed and nothing is claimed from it.
- **Gappa**: the script is a best-effort template using
  `float32<ne>(...)` rounding and exact dyadic coefficient literals
  (`-11184811 * 1b-26` etc.). It was generated without Gappa installed; its
  syntax and the goal must be validated before results are trusted.
- **GPU**: `exports/AttenuationExpSmallTauV1.slang` mirrors the proven
  operation graph (`module AttenuationExpSmallTauV1;`, per repository Slang
  module ownership). `exports/gpu-manifest.txt` maps input bits to the
  expected bit-exact model outputs (worst inputs, representational
  boundaries, deterministic stratified random). The phase-2 runner executes
  the module through the metallic RHI test infrastructure, compares raw
  output bits, and treats mismatches as compiler-semantics findings — a GPU
  `fma` may be split, fused, or reassociated relative to the proof model,
  and `GLSL.std.450 Fma` does not guarantee single-rounding semantics.

## Roadmap

1. RHI runner consuming the manifest (compiled-semantics verification).
2. MPFR-backed oracle for arbitrary reference expressions.
3. Multi-variable contracts and per-region budgets; sensitive-region splits
   (small denominators, near-total-reflection, roughness near zero).
4. FP16-coefficient and LUT candidates sharing the same acceptance logic.
5. Sollya/Gappa execution as first-class evidence when the tools are present.
