"""Build paired scene-test executables with native IDs or the old Coverage transport.

Run with no concurrent source edits/builds. Only the CPU instruction transport is
temporarily changed; the scene runner selects the matching shader separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "Source/Runtime/Render/Material/MaterialValueProgram.cpp"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--vsdevcmd", type=Path)
    parser.add_argument("--recover", type=Path, help="Restore source from an interrupted output; leaves binaries unqualified")
    args = parser.parse_args()
    if args.recover:
        original = (args.recover / "MaterialValueProgram.native.cpp").read_bytes()
        legacy = (args.recover / "MaterialValueProgram.legacy.cpp").read_bytes()
        record = json.loads((args.recover / "BuildIdentity.json").read_text())
        if digest(original) != record["nativeSourceSha256"] or digest(legacy) != record["legacySourceSha256"]:
            raise ValueError("Recovery source hashes changed")
        if SOURCE.read_bytes() not in (original, legacy):
            raise ValueError("Concurrent source change; refusing recovery overwrite")
        SOURCE.write_bytes(original)
        print("Native source restored; rebuild MetallicRHITests before using the configured tree executable")
        return
    if not all((args.build_dir, args.output, args.vsdevcmd)):
        parser.error("--build-dir, --output and --vsdevcmd are required for a build")
    build, output = args.build_dir.resolve(), args.output.resolve()
    if output.exists():
        raise ValueError("Use a new output directory")
    output.mkdir(parents=True)
    original = SOURCE.read_bytes()
    text = original.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    anchor = "        for (size_t i = 0; i < coverageSources.size(); ++i) {"
    if text.count(anchor) != 1:
        raise ValueError("Coverage transport source changed; inspect before updating patch")
    insertion = """        // Experiment-only legacy transport; restored after this executable build.
        for (const auto& [source, slice] : coveragePrograms) {
            require(result->inputBytes_.size() <= UINT32_MAX - slice.instructions.size() * sizeof(MaterialCoverageInstruction),
                "Coverage byte offsets exceed uint32");
            coverageIds[source] = static_cast<uint32_t>(result->inputBytes_.size());
            const auto* first = reinterpret_cast<const std::byte*>(slice.instructions.data());
            result->inputBytes_.insert(result->inputBytes_.end(), first, first + slice.instructions.size() * sizeof(MaterialCoverageInstruction));
        }
""".replace("\n", newline)
    legacy = text.replace(anchor, insertion + anchor).encode("utf-8")
    (output / "MaterialValueProgram.native.cpp").write_bytes(original)
    (output / "MaterialValueProgram.legacy.cpp").write_bytes(legacy)
    shutil.copyfile(__file__, output / Path(__file__).name)
    record = {"source": str(SOURCE), "nativeSourceSha256": digest(original),
              "legacySourceSha256": digest(legacy), "sourceRestored": False, "builds": {}}
    def other_sources():
        files = {p for tree in (ROOT / "Source", ROOT / "tests") for pattern in ("*.cpp", "*.h", "CMakeLists.txt") for p in tree.rglob(pattern)}
        files.discard(SOURCE)
        files.update((ROOT / "CMakeLists.txt", build / "CMakeCache.txt"))
        return {str(p): digest(p.read_bytes()) for p in sorted(files)}
    record["sharedSources"] = other_sources()
    def persist():
        (output / "BuildIdentity.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    persist()
    def build_arm(arm):
        if other_sources() != record["sharedSources"]:
            raise RuntimeError("Shared build inputs changed")
        script = output / (arm + ".cmd")
        script.write_text(f'@echo off\ncall "{args.vsdevcmd.resolve()}" -arch=x64 -host_arch=x64 >nul\n'
                          f'if errorlevel 1 exit /b %errorlevel%\ncmake --build "{build}" --target MetallicRHITests -j 8\n', encoding="utf-8")
        with (output / (arm + "-build.log")).open("w", encoding="utf-8") as log:
            result = subprocess.run(["cmd", "/d", "/c", str(script)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record["builds"][arm] = {"exitCode": result.returncode}
        persist()
        if result.returncode:
            raise RuntimeError(f"{arm} build failed; inspect retained log")
        if other_sources() != record["sharedSources"]:
            raise RuntimeError("Shared build inputs changed during compilation")
        destination = output / arm
        destination.mkdir()
        files = [build / "tests/MetallicRHITests.exe", *sorted((build / "tests").glob("*.dll"))]
        for path in files:
            shutil.copy2(path, destination / path.name)
        record["builds"][arm]["binaries"] = {p.name: digest(p.read_bytes()) for p in destination.iterdir()}
        persist()
    SOURCE.write_bytes(legacy)
    try:
        build_arm("legacy")
    finally:
        if SOURCE.read_bytes() != legacy:
            raise RuntimeError(f"Concurrent source change; refusing overwrite. Original retained at {output}")
        SOURCE.write_bytes(original)
        record["sourceRestored"] = True
        persist()
    build_arm("native")
    if SOURCE.read_bytes() != original:
        raise RuntimeError("Native source changed during build")
    print(output / "BuildIdentity.json")


if __name__ == "__main__":
    main()
