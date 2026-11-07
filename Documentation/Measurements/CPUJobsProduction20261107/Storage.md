# Production evidence storage

The production raw records are retained as a versioned GitHub evidence asset,
separate from ordinary source history. This change moves storage only: no
renderer, cooker, correctness test or performance measurement is rerun, and
none of the original records, hashes or measured conclusions is rewritten.

- Release: [pr-74-cpu-jobs-evidence-20261107](https://github.com/af8a2a/metallic/releases/tag/pr-74-cpu-jobs-evidence-20261107).
- Asset: [metallic-cpu-jobs-production-20261107.zip](https://github.com/af8a2a/metallic/releases/download/pr-74-cpu-jobs-evidence-20261107/metallic-cpu-jobs-production-20261107.zip).
- Size: **159,991,114 bytes**; SHA256:
  `e51c7a7c0ff634f951a1692c85f4b578ea4b3434ab72d539ca7244a2a8971f83`.
- [Data/Storage.json](Data/Storage.json) catalogs every outer member's original
  byte size and SHA256. [Data/Evidence.json](Data/Evidence.json) is the unchanged
  original manifest for the inner run archives and their complete records.

The asset restores the original `Data/<bundle>/` layout, including all original
`Metadata.json`, `Raw.jsonl`, `Summary.json` and `Runs.zip` bytes. The outer ZIP
uses stored entries; original inner `Runs.zip` files remain ZIP LZMA. Generated
meshstream binaries excluded by the original package stay excluded: their
validated hashes and original build paths remain in the unchanged manifest.
All cooked output hashes match the provided source streamasset.

Versioned evidence prerelease assets have **no automatic expiry**, unlike
short-lived CI artifacts. Retain this release while the PR/report is cited;
any removal requires a maintainer decision. The local ignored backup at
`build-issue73-production/evidence-release-20261107/OriginalData` is a second
copy, not the remote retention mechanism. A maintainer can still delete a
release; this is not an irrevocable storage guarantee.

## Lightweight source records

The repository keeps plots, exact as-run plotting scripts, audits, source and
executable provenance, original per-file hashes, and per-run
`CompactSummary.json` files. Compact summaries omit repeated geometry arrays,
per-scope distributions and other large duplicated fields; they explicitly
identify their schema and the original JSONL line/file hashes. They are not
substitutes for raw inputs to the plotting scripts. Metadata files retain
their original bytes, including the stopped original frame plan's status;
that preserved status is historical and does not mean a process is running.

The original 56-process frame plan was stopped at the user's request.
`FrameReduced` contains eight excluded prewarm records and only the completed
W1 ABBA processes 8–11 for paired conclusions. `FrameOriginal` preserves the
one-sided W2 process and interrupted attempt. Migration does not reclassify
those populations or add samples.

## Restore and replot without GPU work

Download the asset from the release page. For an authenticated/private-repository
download, GitHub CLI can use the same versioned asset:

```powershell
gh release download pr-74-cpu-jobs-evidence-20261107 --repo af8a2a/metallic --pattern metallic-cpu-jobs-production-20261107.zip --dir build-issue73-evidence-download
python -B Documentation/Measurements/CPUJobsProduction20261107/RestoreEvidence.py --archive build-issue73-evidence-download/metallic-cpu-jobs-production-20261107.zip --output build-issue73-evidence-restored --bundle FrameReduced --bundle ColdCook
python -B Documentation/Measurements/CPUJobsProduction20261107/FinalPlots/PlotProductionCPUJobs.py --production build-issue73-evidence-restored/Data/FrameReduced --production-cook build-issue73-evidence-restored/Data/ColdCook --output build-issue73-replot
```

The restore helper rejects a different asset size/hash, verifies all original
outer file hashes, and optionally expands requested inner run archives and
checks their original per-file hashes. It refuses to overwrite an existing
nonempty restoration directory. It never launches a renderer, test or benchmark.
Use `--bundle SteadySoak` or `--bundle CoreValidation` for their complete
diagnostic/image inputs. The original absolute input paths in plotting/audit
manifests identify where those records were measured, not required restore paths.

## History and provenance

The original heavy measurement commit was `e9ba6b90c26ee819d35e352b8c7819862efbefa9`.
To avoid retaining the large raw blobs in the PR's ordinary ancestry, the
measurement-only commit is rebuilt with this lightweight storage tree; earlier
measured code commits such as `53237e6df` stay unchanged. Subsequent review
fixes are replayed after that tree. The original heavy history is retained only
in local backup refs/bundles, not pushed as a remote branch or tag.

The evidence release tag points to the lightweight branch, **not** the old heavy
commit. Original binary/source SHA256s, original evidence metadata and exact
plotting snapshots remain the authority for the historical measurements. The
new code review fixes have only their separately reported targeted validation;
the historical measurements are not relabeled as timings of the revised code.
