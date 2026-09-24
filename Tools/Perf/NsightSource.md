# Shader Profiler source evidence queries

M1 currently provides a **mapping-driven offline importer**, not a validated
native Nsight CSV dialect or a working UI adapter. The historical raw CSV is
remote-only. Tests use synthetic contract data, never regenerated historical
measurements. GPU Trace metrics tables belong to a different importer.

Python standard library only; no renderer build, GPU or Nsight is needed:

```powershell
python -B Tools/Perf/NsightSource.py inspect path/to/export.csv --start 1 --count 12
python -B Tools/Perf/NsightSource.py import --raw path/to/export.csv --layout layout.json --context context.json --output build/source-run-01
python -B Tools/Perf/NsightSource.py verify build/source-run-01
python -B Tools/Perf/NsightSource.py shaders build/source-run-01
python -B Tools/Perf/NsightSource.py hotspots build/source-run-01 --module "exact module identifier" --entry "exact entry" --representation il
python -B Tools/Perf/NsightSource.py source build/source-run-01 --module "exact module identifier" --entry "exact entry" --record 9 --radius 3
python -B Tools/Perf/NsightSource.py compare-repeats build/source-run-01 build/source-run-02 build/source-run-03
```

The module and record above are placeholders: use values returned by your
bundle. Commands emit JSON. Exit 0 means the requested operation succeeded,
1 means repeat results differ, and 2 means invalid or unavailable evidence.
Import refuses to overwrite an existing output directory. Inspect supports
comma/tab/semicolon and explicit UTF-8/UTF-16 encodings; malformed CSV or decoding
errors fail without silently replacing bytes. Inspect defaults to UTF-8 with
optional BOM; use --encoding utf-16 for UTF-16 BOM input.

## Explicit layout contract, version 1

Inspect the raw export before writing a layout. It must contain:

- version: 1; raw_sha256: exact SHA256 returned by inspect.
- delimiter: comma by default (literal tab or semicolon also supported).
- encoding: utf-8-sig by default, or utf-16 / utf-16-le / utf-16-be.
- sections: nonoverlapping, explicitly bounded tables. Each has header_record,
  end_record (inclusive), header (exact ordered array, including duplicate names),
  representation (il, source, summary), columns and identity_columns.

CSV records are **1-based**, column indexes are **0-based**. Quoted multiline
fields count as one record; evidence also retains physical line spans.
Each section begins immediately after its header and ends at end_record.
Only truly empty records are skipped; footer and metadata rows must be outside
table bounds. Nonempty unmapped records are counted and their first 20 indexes
are reported. Their presence is a coverage gap, not an import failure.

columns maps module, entry, file, line, code, self_samples, inclusive_samples,
dependency_samples and live_registers to positional indexes. module and entry
columns are required; optional fields may be omitted. Header names are not
guessed or matched fuzzily. This first contract requires explicit module/entry
cells in each data row. Native exports using section markers or inferred entry
names need an evidence-checked dialect adapter once actual bytes are available;
do not edit the original export or invent an entry to satisfy this contract.

identity_columns is a nonempty list of indexes defining a unique logical row
within a module, entry and representation. Include instruction ID, file,
function/callsite or other context as necessary. A source line alone is often
insufficient. Equal rows with the same identity retain every evidence location
but count once. Conflicting duplicates fail. A mapping that omits relevant
identity dimensions cannot be independently diagnosed without native fixtures.

Sample values must be full nonnegative decimal integers or empty. Empty means
unknown, whereas 0 remains measured zero. K/M abbreviations, grouping separators,
percentages and fractional values are rejected rather than assigned guessed
units. IL self sample totals cover only the mapped rows. Missing values make
their completeness flag false; even a true flag does not prove export coverage.
Source self, inclusive and dependency samples are queryable but never summed
into IL self totals. No milliseconds or speedup are inferred.

Optional stalls is a list of objects with name and samples column indexes.
If present, stall_coverage must be top-k-lower-bound or complete, based on the
actual export. Names (including Not Selected) are retained verbatim, and missing
counts stay unknown. A top-three export must use top-k-lower-bound; this tool does
not relabel Not Selected as an execution stall or derive occupancy from registers.

A **synthetic example** (not an observed Nsight header) is:

```json
{
  "version": 1,
  "raw_sha256": "<64 lowercase hex characters from inspect>",
  "sections": [{
    "header_record": 1,
    "end_record": 3,
    "header": ["Module", "Entry", "Instruction", "Self", "Inclusive"],
    "representation": "il",
    "columns": {"module": 0, "entry": 1, "self_samples": 3, "inclusive_samples": 4},
    "identity_columns": [2]
  }]
}
```

## Context contract and trust boundary

context.json requires artifact_sha256 (SHA256 of the capture/trace),
nsight_version, export_id (different for each actual export), and selection:

```json
{
  "artifact_sha256": "<SHA256 of the actual trace/capture>",
  "nsight_version": "<observed version>",
  "export_id": "<unique actual export ID>",
  "selection": {
    "id": "<stable selection ID>",
    "requested_scope": "shader-in-window",
    "achieved_scope": "shader-in-window"
  }
}
```

Scope values are dispatch, shader-in-window, marker-range, whole-device.
Requested and achieved scope must match for this analysis bundle. A mismatch
fails; correct the actual selection or explicitly create a separate broader
case. Queue, dispatch, timeline bounds, grouping, filtering, pipeline/variant,
environment, source snapshot and selection screenshots/logs should be retained
as additional context fields. A declared artifact hash is not proof that the
capture was opened. The tool cannot inspect actual UI selection or verify
artifact identity, symbols, source correlation or a tool-module ↔ SPIR-V mapping.

Queries require a unique module/entry pair. An empty entry can be inventoried,
but hotspot/source attribution fails until its identity is established.
Source queries return only exported source rows in the same file, with gaps
unfilled. They never read similarly named files from the current working tree.
IL-to-source or dependency producer/consumer links are not inferred.

## Bundle and repeat checks

The new directory contains exact raw.csv, layout.json, context.json, derived
analysis.json and manifest.json. File hashes and the parser hash are pinned.
Every query verifies hashes and regenerates the analysis. After a parser change,
re-import original evidence into a new directory. These hashes detect accidental
changes; they are not a signed attestation.

compare-repeats requires at least three distinct declared export IDs. It compares
artifact hash, Nsight version, full selection, normalized rows and unmapped
record counts, excluding CSV record positions and duplicate mirror locations.
Equality is an offline consistency result only: it never sets
automation_verified=true or proves that the exports were independently made.
Native Nsight format support, source identity and three actual UI exports remain
separate M1 acceptance gates.

Run validation with:

```powershell
cmake -S tests/perf -B build/perf-tests
ctest --test-dir build/perf-tests -C Debug --output-on-failure
```
