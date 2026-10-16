"""Isolate Slang link-time capability specialization on the real Guides shader.

Preparation snapshots sources without changing production shaders. Runs separate
source-to-module compilation from module-to-SPIR-V linking. Feature removal is
an explicit experimental scene restriction, never an assertion of general scene
equivalence. Compare matching static and link-time feature masks before claiming
that link-time specialization removes more code.
"""
import argparse
from collections import Counter
import ctypes
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import ProfileComputeShaderFrontend as profile

ROOT = Path(__file__).resolve().parents[2]
MECHANISMS = ("original", "preprocessor", "static", "constant", "type-static", "type")
FEATURES = {
    "general": dict(translucency=True, metallic=True, fiber=True),
    "no-translucency": dict(translucency=False, metallic=True, fiber=True),
    "no-metallic": dict(translucency=True, metallic=False, fiber=True),
    "no-fiber": dict(translucency=True, metallic=True, fiber=False),
    "opaque-dielectric": dict(translucency=False, metallic=False, fiber=False),
}
CAPABILITIES = ("SheenAndCoat", "Dispersion", "Translucency", "Metallic", "Fiber")
SURFACE_SOURCE = Path("Modules/OpenPBRSurface.slang")
VERTEX_SOURCE = Path("Modules/OpenPBRPathTracing/OpenPBRPathVertex.slang")
FEATURE_METHODS = """    public static bool enableSheenAndCoat()
    {
#if METALLIC_CUSTOM_MATERIALS
        return true;
#else
        return false;
#endif
    }
    public static bool enableDispersion() { return false; }
    public static bool enableTranslucency() { return true; }
    public static bool enableMetallic() { return true; }"""
FIBER_BRANCH = "    if (materialDomain(transport) == kMaterialDomainFiber) {"
COVERAGE_BRANCH = "    if (transport.textureParams.w > 1.5 && push.debugView == kOpenPBRDebugViewFinal) {"


def validate_shader_layout(shader_root):
    """Fail on moved/changed gates before snapshotting an invalid experiment."""
    for relative, owner, namespace, anchors in (
            (SURFACE_SOURCE, "module OpenPBRSurface;", "namespace Metallic.SceneOpenPBR",
             ("public struct SceneOpenPBRContext : IOpenPBRContext", FEATURE_METHODS,
              "// The closure module receives")),
            (VERTEX_SOURCE, "implementing OpenPBRPathTracing;", "namespace Metallic.OpenPBRPathTracing",
             (FIBER_BRANCH, COVERAGE_BRANCH))):
        path = shader_root / relative
        if not path.is_file():
            raise ValueError(f"Preparation source layout changed: expected owned shader {path}")
        source = path.read_text(encoding="utf-8")
        owners = re.findall(r"^\s*(?:module|implementing)\s+[\w.]+\s*;", source, re.M)
        namespaces = re.findall(r"^\s*" + re.escape(namespace) + r"\s*\{", source, re.M)
        if len(owners) != 1 or owners[0].strip() != owner or len(namespaces) != 1:
            raise ValueError(f"Preparation source layout changed: ownership or namespace mismatch in {path}")
        for anchor in anchors:
            if source.count(anchor) != 1:
                raise ValueError(f"Preparation source layout changed: expected one feature gate in {path}: {anchor}")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def fresh_directory(path):
    path = path.resolve()
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory must be fresh: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def feature_values(name):
    return dict(SheenAndCoat=False, Dispersion=False,
                Translucency=FEATURES[name]["translucency"],
                Metallic=FEATURES[name]["metallic"], Fiber=FEATURES[name]["fiber"])


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"Expected exactly one shader edit anchor: {before}")
    return text.replace(before, after)


def compiler_manifest(compiler):
    compiler = Path(compiler).resolve()
    sdk = compiler.parent.parent
    files = [compiler, *sorted(compiler.parent.glob("*.dll")),
             sdk / "include/slang.h", sdk / "include/slang-tag-version.h"]
    return [dict(path=str(path), sha256=profile.sha(path)) for path in files]


def rewrite_input_path(token, overlay, source_roots):
    """Rewrite only paths in explicitly snapshotted shader/include roots."""
    try:
        path = Path(token)
        if path.is_absolute():
            resolved = path.resolve()
            if any(resolved.is_relative_to(root) for root in source_roots):
                return str(overlay / resolved.relative_to(ROOT))
    except (ValueError, OSError):
        pass
    return token


def rewrite_command_inputs(command, overlay, source_roots):
    rewritten = list(command)
    rewritten[1] = rewrite_input_path(rewritten[1], overlay, source_roots)
    for index, token in enumerate(command[:-1]):
        if token == "-I":
            rewritten[index + 1] = rewrite_input_path(command[index + 1], overlay, source_roots)
    return rewritten


def prepare(args):
    shader_root = ROOT / "Shaders"
    validate_shader_layout(shader_root)
    output = fresh_directory(args.output_dir)
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    samples = [sample for sample in report["samples"]
               if sample["request"]["index"] == args.request_index]
    if len(samples) != 1 or samples[0]["status"] != "passed":
        raise ValueError("Need exactly one successful fixed Guides request")
    sample = samples[0]
    if sample["request"]["module"] != "Features/PathTracing/OpenPBRRayQueryPathTraceGuides":
        raise ValueError("This experiment currently supports the real OpenPBR Guides entry only")
    if sample["effective_slang_optimization"] != "O1" or sample["request"]["entry"] != "openPbrRayQueryPathTraceGuidesMain":
        raise ValueError("The fixed experiment requires the mapped/O1 Guides entry")
    defines = dict(sample["request"]["defines"])
    if defines.get("METALLIC_CUSTOM_MATERIALS") != "0":
        raise ValueError("The experiment must preserve the measured CUSTOM_MATERIALS=0 feature baseline")
    cache_path, reference = profile.bench_bytes(report, sample)
    (output / "Reference.spv").write_bytes(reference)
    compiler = (args.slang_root / "bin/slangc.exe").resolve()
    base_command = profile.diagnostic_command(compiler, report, sample, output / "unused.spv")
    overlay_root = output / "overlays"
    source_files = []
    additional_roots = [Path(name).resolve() for name in sample["request"]["additional_search_paths"]]
    if any(not root.is_relative_to(ROOT) for root in additional_roots):
        raise ValueError("Additional shader search roots must be snapshotable repository paths")
    input_roots = [shader_root, *additional_roots]
    if any(output.is_relative_to(root) for root in input_roots):
        raise ValueError("Preparation output must not be inside a snapshotted input root")
    for path in sorted(shader_root.rglob("*")):
        if path.is_file():
            source_files.append(dict(path=str(path.relative_to(ROOT)), sha256=profile.sha(path)))
    for root in additional_roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.is_relative_to(shader_root):
                source_files.append(dict(path=str(path.relative_to(ROOT)), sha256=profile.sha(path)))
    auxiliary = []
    for name in sample["dependencies"]:
        path = Path(name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError(f"Dependency lies outside the repository: {path}")
        relative = path.relative_to(ROOT)
        if relative.parts[0] != "Shaders":
            auxiliary.append((path, relative))
    # These textual includes are not consistently present in old cache manifests.
    for name in ("Source/Runtime/Render/Core/NamedResourceParameters.h",
                 "Source/Runtime/Render/Core/ColorSpaceMatrices.h",
                 "Source/Runtime/Render/Environment/AtmosphereKernelParameters.h"):
        item = (ROOT / name, Path(name))
        if item not in auxiliary:
            auxiliary.append(item)
    source_files.extend(dict(path=str(relative), sha256=profile.sha(path))
                        for path, relative in auxiliary)
    interface = ("#language slang 2026\nmodule LinkTimeOpenPBRCapabilities;\n\n"
                 "namespace Metallic.SceneOpenPBR\n{\n")
    interface += "public interface ISceneOpenPBRCapabilities\n{\n"
    interface += "".join(f"    public static bool enable{name}();\n" for name in CAPABILITIES)
    interface += "};\n} // namespace Metallic.SceneOpenPBR\n"
    plans = []
    for mechanism in MECHANISMS:
        overlay = overlay_root / mechanism
        shutil.copytree(shader_root, overlay / "Shaders")
        for root in additional_roots:
            if not root.is_relative_to(shader_root):
                shutil.copytree(root, overlay / root.relative_to(ROOT), dirs_exist_ok=True)
        for path, relative in auxiliary:
            destination = overlay / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
        surface_path = overlay / "Shaders" / SURFACE_SOURCE
        vertex_path = overlay / "Shaders" / VERTEX_SOURCE
        if mechanism == "preprocessor":
            vertex = vertex_path.read_text(encoding="utf-8")
            vertex = replace_once(vertex,
                FIBER_BRANCH,
                "#if METALLIC_EXPERIMENT_ENABLE_FIBER\n" + FIBER_BRANCH)
            vertex = replace_once(vertex,
                COVERAGE_BRANCH,
                "#endif\n" + COVERAGE_BRANCH)
            vertex_path.write_text(vertex, encoding="utf-8")
        elif mechanism != "original":
            surface = surface_path.read_text(encoding="utf-8")
            if mechanism == "static":
                expressions = dict(SheenAndCoat="false", Dispersion="false")
                expressions.update({name: "METALLIC_EXPERIMENT_ENABLE_" + name.upper()
                                    for name in CAPABILITIES[2:]})
            elif mechanism == "constant":
                declarations = "\n".join(f"extern static const bool kSceneEnable{name};"
                                         for name in CAPABILITIES)
                surface = replace_once(surface, "// The closure module receives", declarations + "\n\n// The closure module receives")
                expressions = {name: f"kSceneEnable{name}" for name in CAPABILITIES}
            else:
                surface = replace_once(surface, "module OpenPBRSurface;",
                    "module OpenPBRSurface;\nimport LinkTimeOpenPBRCapabilities;")
                declarations = "extern struct SceneOpenPBRCapabilities : ISceneOpenPBRCapabilities;\n"
                if mechanism == "type-static":
                    declarations = "public struct SelectedOpenPBRCapabilities : ISceneOpenPBRCapabilities\n{\n"
                    for name in CAPABILITIES:
                        value = "false" if name in CAPABILITIES[:2] else "METALLIC_EXPERIMENT_ENABLE_" + name.upper()
                        declarations += f"    public static bool enable{name}() {{ return {value}; }}\n"
                    declarations += "};\ntypealias SceneOpenPBRCapabilities = SelectedOpenPBRCapabilities;\n"
                surface = replace_once(surface, "// The closure module receives", declarations + "\n// The closure module receives")
                expressions = {name: f"SceneOpenPBRCapabilities.enable{name}()" for name in CAPABILITIES}
                (overlay / "Shaders/Modules/LinkTimeOpenPBRCapabilities.slang").write_text(interface, encoding="utf-8")
            after = "\n".join(f"    public static bool enable{name}() {{ return {expressions[name]}; }}"
                              for name in CAPABILITIES)
            surface_path.write_text(replace_once(surface, FEATURE_METHODS, after), encoding="utf-8")
            vertex = vertex_path.read_text(encoding="utf-8")
            vertex_path.write_text(replace_once(vertex,
                FIBER_BRANCH,
                "    if (SceneOpenPBRContext.enableFiber() && materialDomain(transport) == kMaterialDomainFiber) {"), encoding="utf-8")
        feature_names = ("general",) if mechanism == "original" else (("general", "no-fiber") if mechanism == "preprocessor" else FEATURES)
        for feature_name in feature_names:
            values = feature_values(feature_name)
            label = mechanism + "-" + feature_name
            provider = None
            if mechanism in ("constant", "type"):
                provider = output / "providers" / (label + ".slang")
                provider.parent.mkdir(parents=True, exist_ok=True)
                owner = "LinkTimeOpenPBR" + "".join(part.title() for part in label.split("-")) + "Provider"
                content = f"#language slang 2026\nmodule {owner};\n\n"
                if mechanism == "type":
                    content += "import LinkTimeOpenPBRCapabilities;\n\n"
                content += "namespace Metallic.SceneOpenPBR\n{\n"
                if mechanism == "constant":
                    content += "\n".join(f"export static const bool kSceneEnable{name} = {str(value).lower()};"
                                        for name, value in values.items()) + "\n"
                else:
                    content += "public struct SelectedOpenPBRCapabilities : ISceneOpenPBRCapabilities\n{\n"
                    content += "".join(f"    public static bool enable{name}() {{ return {str(value).lower()}; }}\n"
                                       for name, value in values.items())
                    content += "};\nexport struct SceneOpenPBRCapabilities : ISceneOpenPBRCapabilities = SelectedOpenPBRCapabilities;\n"
                content += "} // namespace Metallic.SceneOpenPBR\n"
                provider.write_text(content, encoding="utf-8")
            command = rewrite_command_inputs(base_command, overlay, input_roots)
            if mechanism in ("preprocessor", "static", "type-static"):
                for name in CAPABILITIES[2:]:
                    command += ["-D", f"METALLIC_EXPERIMENT_ENABLE_{name.upper()}={int(values[name])}"]
            plans.append(dict(label=label, mechanism=mechanism, feature_mask=values,
                              provider=str(provider) if provider else None, direct_command=command))
    preparation = dict(schema_version=1, status="prepared", report=str(report_path),
                       report_sha256=profile.sha(report_path), shader_cache=str(cache_path),
                       reference_fnv1a64=profile.fnv(reference), reference_sha256=hashlib.sha256(reference).hexdigest(),
                       source_files=source_files, plans=plans, compiler=str(compiler),
                       compiler_sha256=profile.sha(compiler),
                       sdk_files=compiler_manifest(compiler), helper_sha256=profile.sha(__file__),
                       scope="Current real mapped/O1 Guides workload. Source overlays only. Feature-off experiments require explicit scene capability validation.",
                       caution="Feature restrictions require validated material capabilities across all reachable hits and numerical GPU qualification. Slang link-time values are not Vulkan specialization constants.")
    write_json(output / "Preparation.json", preparation)
    write_json(output / "Commands.json", dict(schema_version=1, plans=plans))
    profile.seal(output)
    print(json.dumps(dict(status="prepared", preparation=str(output), variants=len(plans)), indent=2))


def verify_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads((path / "Manifest.json").read_text(encoding="utf-8"))
    listed = set()
    for artifact in manifest["artifacts"]:
        filename = (path / artifact["path"]).resolve()
        if not filename.is_relative_to(path) or filename in listed or profile.sha(filename) != artifact["sha256"]:
            raise ValueError("Artifact checksum or path mismatch")
        listed.add(filename)
    actual = {filename.resolve() for filename in path.rglob("*")
              if filename.is_file() and filename.name != "Manifest.json"}
    if listed != actual:
        raise ValueError("Artifact manifest inventory mismatch")


def verify_preparation(path):
    path = Path(path).resolve()
    verify_manifest(path)
    preparation = json.loads((path / "Preparation.json").read_text(encoding="utf-8"))
    if profile.sha(preparation["compiler"]) != preparation["compiler_sha256"]:
        raise ValueError("Pinned compiler changed after preparation")
    for artifact in preparation.get("sdk_files", []):
        if profile.sha(artifact["path"]) != artifact["sha256"]:
            raise ValueError("Pinned compiler SDK changed after preparation")
    return preparation


def command_parts(plan):
    command = plan["direct_command"]
    source = command[1]
    flags = []
    module_name = None
    entry = None
    index = 2
    while index < len(command):
        token = command[index]
        if token in ("-module-name", "-entry", "-target", "-o"):
            value = command[index + 1]
            if token == "-module-name":
                module_name = value
            elif token == "-entry":
                entry = value
            index += 2
        elif token == "-emit-spirv-directly":
            index += 1
        else:
            flags.append(token)
            index += 1
    return command[0], source, flags, module_name, entry


def invoke(command, directory, name, timeout):
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=timeout)
    seconds = time.perf_counter() - started
    (directory / (name + ".stdout.log")).write_bytes(result.stdout)
    (directory / (name + ".stderr.log")).write_bytes(result.stderr)
    record = dict(command=command, exit_code=result.returncode, wall_seconds=seconds,
                  stdout=name + ".stdout.log", stderr=name + ".stderr.log")
    if result.returncode:
        record["status"] = "failed"
    else:
        record["status"] = "passed"
    return record


def run(args):
    preparation_path = args.preparation.resolve()
    preparation = verify_preparation(preparation_path)
    output = fresh_directory(args.output_dir)
    wanted = set(args.variants or [plan["label"] for plan in preparation["plans"]])
    plans = [plan for plan in preparation["plans"] if plan["label"] in wanted]
    if wanted != {plan["label"] for plan in plans}:
        raise ValueError("Unknown requested variant")
    summary = dict(schema_version=1, status="running", preparation=str(preparation_path),
                   preparation_manifest_sha256=profile.sha(preparation_path / "Manifest.json"),
                   runs=[], repetitions=args.repetitions, variants=[plan["label"] for plan in plans],
                   mode="source-and-link", helper_sha256=profile.sha(__file__),
                   sdk_files=compiler_manifest(preparation["compiler"]),
                   scope="Sequential independent compiler processes. Initial consumer/provider compilation is recorded separately from linking frozen IR. Imported source modules may still be loaded during linking; phases overlap.")
    write_json(output / "summary.json", summary)
    old_error_mode = ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002) if sys.platform == "win32" else None
    try:
        with profile.experiment_lock():
            for repetition in range(args.repetitions):
                ordered = plans if repetition % 2 == 0 else list(reversed(plans))
                for plan in ordered:
                    directory = output / f"block-{repetition:02}-{plan['label']}"
                    directory.mkdir()
                    compiler, source, flags, module_name, entry = command_parts(plan)
                    stages = []
                    if plan["mechanism"] == "original":
                        direct = list(plan["direct_command"])
                        direct[direct.index("-o") + 1] = str(directory / "Direct.spv")
                        stages.append(invoke(direct, directory, "source-direct", args.timeout))
                    consumer_module = directory / "Consumer.slang-module"
                    provider_module = directory / "Provider.slang-module"
                    stage = invoke([compiler, "-target", "spirv", *flags, "-no-codegen", "-emit-ir", source, "-module-name", module_name,
                                    "-o", str(consumer_module)], directory, "consumer-precompile", args.timeout)
                    stages.append(stage)
                    if stage["status"] == "passed" and plan["provider"]:
                        stage = invoke([compiler, "-target", "spirv", *flags, "-no-codegen", "-emit-ir", plan["provider"], "-o", str(provider_module)],
                                       directory, "provider-precompile", args.timeout)
                        stages.append(stage)
                    destination = directory / "Guides.spv"
                    if all(stage["status"] == "passed" for stage in stages):
                        modules = [str(provider_module)] if plan["provider"] else []
                        stages.append(invoke([compiler, "-target", "spirv", *flags, *modules, str(consumer_module),
                                              "-emit-spirv-directly", "-entry", entry,
                                              "-o", str(destination)], directory, "link", args.timeout))
                    record = dict(label=plan["label"], mechanism=plan["mechanism"],
                                  feature_mask=plan["feature_mask"], repetition=repetition,
                                  directory=directory.name, stages=stages,
                                  status="passed" if all(stage["status"] == "passed" for stage in stages) and destination.exists() else "failed")
                    if record["status"] == "passed":
                        raw = destination.read_bytes()
                        record.update(spirv_bytes=len(raw), spirv_fnv1a64=profile.fnv(raw),
                                      spirv_sha256=hashlib.sha256(raw).hexdigest(),
                                      matches_current_bench_byte_for_byte=raw == (preparation_path / "Reference.spv").read_bytes())
                        if plan["mechanism"] == "original":
                            direct_matches = (directory / "Direct.spv").read_bytes() == (preparation_path / "Reference.spv").read_bytes()
                            record["source_direct_matches_current_bench_byte_for_byte"] = direct_matches
                            if not direct_matches or not record["matches_current_bench_byte_for_byte"]:
                                record["status"] = "baseline_bytecode_mismatch"
                    summary["runs"].append(record)
                    write_json(output / "summary.json", summary)
                    print(json.dumps(dict(label=record["label"], repetition=repetition,
                                          status=record["status"], spirv_bytes=record.get("spirv_bytes"))), flush=True)
            summary["status"] = "passed" if all(record["status"] == "passed" for record in summary["runs"]) else "failed"
            write_json(output / "summary.json", summary)
            profile.seal(output)
    finally:
        if old_error_mode is not None:
            ctypes.windll.kernel32.SetErrorMode(old_error_mode)
    return 0 if summary["status"] == "passed" else 1


def relink(args):
    """Regenerate target code from checksum-verified, previously serialized IR.

    The consumer/provider modules are reused byte-for-byte. Their imported
    modules remain in the frozen source overlay; this is not a full dependency
    IR cache. A changed final optimization level is recorded independently from
    the optimization that produced the frozen consumer IR.
    """
    input_path = args.input_dir.resolve()
    initial = verify_run(input_path)
    preparation_path = Path(initial["preparation"]).resolve()
    preparation = verify_preparation(preparation_path)
    output = fresh_directory(args.output_dir)
    wanted = set(args.variants or initial["variants"])
    plans = [plan for plan in preparation["plans"] if plan["label"] in wanted]
    if wanted != {plan["label"] for plan in plans} or not wanted <= set(initial["variants"]):
        raise ValueError("The initial evidence must contain every requested variant")
    if any(plan["mechanism"] == "original" for plan in plans) and args.target_optimization != "O1":
        raise ValueError("The exact Bench original control requires O1 final target generation")
    summary = dict(schema_version=1, status="running", preparation=str(preparation_path),
                   preparation_manifest_sha256=profile.sha(preparation_path / "Manifest.json"),
                   runs=[], repetitions=args.repetitions, variants=[plan["label"] for plan in plans],
                   mode="relink", final_target_optimization=args.target_optimization,
                   initial_ir_evidence=str(input_path), initial_ir_manifest_sha256=profile.sha(input_path / "Manifest.json"),
                   helper_sha256=profile.sha(__file__), sdk_files=compiler_manifest(preparation["compiler"]),
                   scope="Frozen consumer/provider IR reuse only; imported modules remain frozen source. Initial precompile costs are excluded from each relink wall time and recorded separately. Changing final target optimization is a separate experiment.")
    write_json(output / "summary.json", summary)
    old_error_mode = ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002) if sys.platform == "win32" else None
    try:
        with profile.experiment_lock():
            for repetition in range(args.repetitions):
                ordered = plans if repetition % 2 == 0 else list(reversed(plans))
                for plan in ordered:
                    original = next(record for record in initial["runs"]
                                    if record["label"] == plan["label"] and record["repetition"] == 0)
                    original_directory = input_path / original["directory"]
                    directory = output / f"block-{repetition:02}-{plan['label']}"
                    directory.mkdir()
                    reused = []
                    for name in (["Consumer.slang-module", "Provider.slang-module"] if plan["provider"] else ["Consumer.slang-module"]):
                        source = original_directory / name
                        data = source.read_bytes()
                        if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"SLmc":
                            raise ValueError("Frozen IR does not contain a Slang module container")
                        destination = directory / name
                        destination.write_bytes(data)
                        reused.append(dict(source=str(source), destination=name,
                                           sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
                    compiler, source, flags, module_name, entry = command_parts(plan)
                    flags = ["-" + args.target_optimization if flag == "-O1" else flag for flag in flags]
                    modules = [str(directory / "Provider.slang-module")] if plan["provider"] else []
                    destination = directory / "Guides.spv"
                    stage = invoke([compiler, "-target", "spirv", *flags, *modules,
                                    str(directory / "Consumer.slang-module"), "-emit-spirv-directly", "-entry", entry,
                                    "-o", str(destination)], directory, "relink", args.timeout)
                    record = dict(label=plan["label"], mechanism=plan["mechanism"],
                                  feature_mask=plan["feature_mask"], repetition=repetition,
                                  directory=directory.name, stages=[stage], reused_ir=reused,
                                  initial_precompile_wall_seconds=sum(item["wall_seconds"] for item in original["stages"]
                                                                     if item["stdout"].startswith(("consumer-precompile", "provider-precompile"))),
                                  status=stage["status"])
                    if record["status"] == "passed":
                        raw = destination.read_bytes()
                        record.update(spirv_bytes=len(raw), spirv_fnv1a64=profile.fnv(raw),
                                      spirv_sha256=hashlib.sha256(raw).hexdigest(),
                                      matches_current_bench_byte_for_byte=raw == (preparation_path / "Reference.spv").read_bytes())
                        if plan["mechanism"] == "original":
                            shutil.copy2(original_directory / "Direct.spv", directory / "Direct.spv")
                            if not record["matches_current_bench_byte_for_byte"]:
                                record["status"] = "baseline_bytecode_mismatch"
                    summary["runs"].append(record)
                    write_json(output / "summary.json", summary)
                    print(json.dumps(dict(label=record["label"], repetition=repetition,
                                          target_optimization=args.target_optimization,
                                          status=record["status"], spirv_bytes=record.get("spirv_bytes"))), flush=True)
            summary["status"] = "passed" if all(record["status"] == "passed" for record in summary["runs"]) else "failed"
            write_json(output / "summary.json", summary)
            profile.seal(output)
    finally:
        if old_error_mode is not None:
            ctypes.windll.kernel32.SetErrorMode(old_error_mode)
    return 0 if summary["status"] == "passed" else 1


def verify_run(path):
    path = Path(path).resolve()
    verify_manifest(path)
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "passed" or not summary["runs"]:
        raise ValueError("Failed or incomplete link-time experiment")
    variants = summary["variants"]
    if not variants or len(variants) != len(set(variants)) or summary["repetitions"] < 1:
        raise ValueError("Invalid experiment schedule")
    expected = [(repetition, label) for repetition in range(summary["repetitions"])
                for label in (variants if repetition % 2 == 0 else list(reversed(variants)))]
    if [(record["repetition"], record["label"]) for record in summary["runs"]] != expected:
        raise ValueError("Missing, reordered or duplicated experiment run")
    preparation = Path(summary["preparation"]).resolve()
    verify_manifest(preparation)
    if profile.sha(preparation / "Manifest.json") != summary["preparation_manifest_sha256"]:
        raise ValueError("Preparation manifest changed after the experiment")
    reference = (preparation / "Reference.spv").read_bytes()
    preparation_data = json.loads((preparation / "Preparation.json").read_text(encoding="utf-8"))
    known_plans = {plan["label"]: plan for plan in preparation_data["plans"]}
    if any(label not in known_plans for label in variants):
        raise ValueError("Unknown variant in the experiment schedule")
    if summary.get("mode") == "relink":
        origin = Path(summary["initial_ir_evidence"]).resolve()
        if origin == path:
            raise ValueError("Relink evidence cannot reuse itself")
        verify_manifest(origin)
        if profile.sha(origin / "Manifest.json") != summary["initial_ir_manifest_sha256"]:
            raise ValueError("Initial frozen IR evidence changed")
    for record in summary["runs"]:
        directory = (path / record["directory"]).resolve()
        canonical = f"block-{record['repetition']:02}-{record['label']}"
        if not directory.is_relative_to(path) or record["directory"] != canonical or record["status"] != "passed":
            raise ValueError("Invalid experiment run path or status")
        plan = known_plans[record["label"]]
        if record["mechanism"] != plan["mechanism"] or record["feature_mask"] != plan["feature_mask"]:
            raise ValueError("Recorded mechanism or feature mask differs from the sealed plan")
        if not record["stages"] or any(stage["status"] != "passed" or stage["exit_code"] != 0
                                       for stage in record["stages"]):
            raise ValueError("Unsuccessful compiler stage")
        raw = (directory / "Guides.spv").read_bytes()
        if len(raw) != record["spirv_bytes"] or hashlib.sha256(raw).hexdigest() != record["spirv_sha256"] or profile.fnv(raw) != record["spirv_fnv1a64"]:
            raise ValueError("Recorded SPIR-V identity mismatch")
        instruction_counts(raw)
        if summary.get("mode") == "relink":
            expected_names = {"Consumer.slang-module", "Provider.slang-module"} if plan["provider"] else {"Consumer.slang-module"}
            if {artifact["destination"] for artifact in record["reused_ir"]} != expected_names or len(record["reused_ir"]) != len(expected_names):
                raise ValueError("Incomplete or duplicated reused IR modules")
            for artifact in record["reused_ir"]:
                original_module = Path(artifact["source"]).resolve()
                reused_module = directory / artifact["destination"]
                expected_original = origin / f"block-00-{record['label']}" / artifact["destination"]
                if original_module != expected_original.resolve() or not original_module.is_relative_to(origin) or profile.sha(original_module) != artifact["sha256"] or profile.sha(reused_module) != artifact["sha256"] or reused_module.stat().st_size != artifact["bytes"]:
                    raise ValueError("Reused IR identity mismatch")
        if record["label"] == "original-general":
            if raw != reference or (directory / "Direct.spv").read_bytes() != reference:
                raise ValueError("Original direct or serialized bytecode differs from the pinned Bench reference")
    return summary


def instruction_counts(data):
    if len(data) < 20 or len(data) % 4 or data[:4] != b"\x03\x02\x23\x07":
        raise ValueError("Invalid SPIR-V header")
    words = struct.unpack("<" + "I" * (len(data) // 4), data)
    cursor = 5
    counts = Counter()
    while cursor < len(words):
        size = words[cursor] >> 16
        if not size or cursor + size > len(words):
            raise ValueError("Invalid SPIR-V instruction")
        counts[words[cursor] & 65535] += 1
        cursor += size
    return dict(instructions=sum(counts.values()), functions=counts[54],
                function_calls=counts[57], basic_blocks=counts[248],
                loop_merges=counts[246], phi=counts[245],
                specialization_constant_instructions=sum(counts[opcode] for opcode in range(48, 53)),
                opcode_counts={str(opcode): count for opcode, count in sorted(counts.items())})


def analyze(args):
    input_path = args.input_dir.resolve()
    summary = verify_run(input_path)
    output = fresh_directory(args.output_dir)
    records = []
    for record in summary["runs"]:
        source = input_path / record["directory"] / "Guides.spv"
        raw = source.read_bytes()
        normalized = output / (record["directory"] + ".compact.spv")
        subprocess.run([str(args.spirv_opt), "--compact-ids", str(source), "-o", str(normalized)],
                       cwd=ROOT, check=True, capture_output=True, timeout=120)
        validation = subprocess.run([str(args.spirv_val), "--target-env", "vulkan1.4",
                                     "--scalar-block-layout", str(source)],
                                    cwd=ROOT, capture_output=True, timeout=120)
        (output / (record["directory"] + ".validation.log")).write_bytes(validation.stdout + validation.stderr)
        if validation.returncode:
            raise ValueError(f"SPIR-V validation failed: {record['label']}")
        records.append(dict(label=record["label"], repetition=record["repetition"],
                            feature_mask=record["feature_mask"], source=str(source), spirv_bytes=len(raw),
                            sha256=profile.sha(source), compact_ids_sha256=profile.sha(normalized),
                            counts=instruction_counts(raw)))
    comparisons = []
    for name in FEATURES:
        matching = [record for record in records if record["label"].endswith("-" + name)]
        for left_index, left in enumerate(matching):
            for right in matching[left_index + 1:]:
                if left["repetition"] != right["repetition"]:
                    continue
                comparisons.append(dict(left=left["label"], right=right["label"], repetition=left["repetition"],
                                        raw_byte_equal=left["sha256"] == right["sha256"],
                                        compact_ids_byte_equal=left["compact_ids_sha256"] == right["compact_ids_sha256"],
                                        opcode_counts_equal=left["counts"]["opcode_counts"] == right["counts"]["opcode_counts"]))
    write_json(output / "Analysis.json", dict(schema_version=1, status="passed", source=str(input_path),
                                               records=records, comparisons=comparisons,
                                               caution="Compact IDs are renumbering only. Opcode equality does not establish numerical or scene equivalence. No Vulkan specialization constants are expected."))
    profile.seal(output)
    print(json.dumps(dict(status="passed", analyzed=len(records), comparisons=len(comparisons)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--report", type=Path, required=True)
    prepare_parser.add_argument("--request-index", type=int, default=146)
    prepare_parser.add_argument("--slang-root", type=Path, default=ROOT / "External/slang")
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--preparation", type=Path, required=True)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--variants", nargs="+")
    run_parser.add_argument("--repetitions", type=int, default=1)
    run_parser.add_argument("--timeout", type=float, default=900)
    relink_parser = commands.add_parser("relink")
    relink_parser.add_argument("--input-dir", type=Path, required=True)
    relink_parser.add_argument("--output-dir", type=Path, required=True)
    relink_parser.add_argument("--variants", nargs="+")
    relink_parser.add_argument("--repetitions", type=int, default=1)
    relink_parser.add_argument("--timeout", type=float, default=900)
    relink_parser.add_argument("--target-optimization", choices=("O1", "O2"), default="O1")
    analyze_parser = commands.add_parser("analyze")
    analyze_parser.add_argument("--input-dir", type=Path, required=True)
    analyze_parser.add_argument("--output-dir", type=Path, required=True)
    analyze_parser.add_argument("--spirv-opt", type=Path, default=Path("C:/VulkanSDK/1.4.363.0/Bin/spirv-opt.exe"))
    analyze_parser.add_argument("--spirv-val", type=Path, default=Path("C:/VulkanSDK/1.4.363.0/Bin/spirv-val.exe"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command in ("run", "relink"):
        if args.repetitions < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
            parser.error("Positive repetitions and timeout are required")
        return run(args) if args.command == "run" else relink(args)
    else:
        analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
