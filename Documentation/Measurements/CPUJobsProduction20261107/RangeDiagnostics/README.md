# Range/TaskGraph correlation example

[Example.json](Example.json) is one real CPU-only diagnostic execution with
seven TaskGraph nodes, two range operations and five lane records. The output
checks every graph/execution/node/thread identity and the lane's enclosing
task interval before marking correctness true. A second Python pass checked
those associations, the async scheduled-to-start delay and the nested null
queue delay. These times are not performance acceptance samples.

The source is [RangeDiagnosticsExample.cpp](../../../../Tools/CPUJobs/RangeDiagnosticsExample.cpp).
Its manual CMake target is MetallicRangeDiagnosticsExample, excluded from the
default build. With a configured root tree, explicitly build that target and
run build-issue73-production/cpu-jobs/MetallicRangeDiagnosticsExample.exe with a fresh
JSON output path. A standalone Tools/CPUJobs configuration places the executable
directly in its binary directory. No GPU or external scene is needed.

[Metadata.json](Metadata.json) records the canonical example source and sample
SHA256s. The archived execution was rebuilt and run through the manual CMake
target in the Release production tree. Reproduction creates new operation IDs,
thread hashes and timestamps;
those values are expected to change while the association checks remain true.

The backend names are generic caller labels for this integer correctness demo,
not assertions about a renderer or ISPC backend. The actual production meshlet
adapter leaves range diagnostics disabled. This example proves the opt-in
data contract, not that every production caller emits these counters.
