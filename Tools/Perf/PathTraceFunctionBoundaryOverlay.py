"""Plan a single noinline boundary on the current full-feature Guides source.

Only the annotation immediately before openPbrShadowTransmittance changes.
This pure generator neither writes production sources nor launches compilation.
Original line endings and every other source byte are preserved.
"""

import hashlib
from pathlib import Path
import re


ARMS = ("original-general", "shadow-noinline-general")
OWNER_SOURCE = "Modules/OpenPBRPathTracing.slang"
TARGET_SOURCE = "Modules/OpenPBRPathTracing/OpenPBRPathTrace.slang"
ENTRY_SOURCE = "Features/PathTracing/OpenPBRRayQueryPathTraceGuides.slang"
SOURCE_FILES = (OWNER_SOURCE, TARGET_SOURCE, ENTRY_SOURCE)
TARGET_FUNCTION = "openPbrShadowTransmittance"
ANNOTATION = "[noinline]"
SIGNATURE = """public float3 openPbrShadowTransmittance(
    float3 origin,
    float3 direction,
    float maxDistance,
    OpenPBRInteriorStack interiorStack,
    ScenePathTracePush push)"""
ANCHOR = SIGNATURE + "\n{"


def sha256(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def validate_owned(source, owner):
    declarations = re.findall(r"^\s*(?:module|implementing)\s+[\w.]+\s*;", source, re.M)
    if len(declarations) != 1 or declarations[0].strip() != owner:
        raise ValueError("Function-boundary shader ownership changed: " + owner)


def plan_overlay(shader_root: Path, arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("Unknown function-boundary arm: " + str(arm))
    shader_root = Path(shader_root)
    sources, hashes = {}, {}
    for relative in SOURCE_FILES:
        raw = (shader_root / relative).read_bytes()
        sources[relative], hashes[relative] = raw.decode("utf-8"), sha256(raw)
    for relative, owner in ((OWNER_SOURCE, "module OpenPBRPathTracing;"),
                            (TARGET_SOURCE, "implementing OpenPBRPathTracing;"),
                            (ENTRY_SOURCE, "module OpenPBRRayQueryPathTraceGuides;")):
        validate_owned(sources[relative], owner)
    owner = sources[OWNER_SOURCE]
    if owner.count('__include "OpenPBRPathTracing/OpenPBRPathTrace.slang";') != 1:
        raise ValueError("Target implementation is no longer included exactly once by its owner")
    entry = sources[ENTRY_SOURCE]
    for anchor in ('import OpenPBRPathTracing;', '[shader("compute")]', '[numthreads(8, 8, 1)]',
                   'void openPbrRayQueryPathTraceGuidesMain(uint3 id : SV_DispatchThreadID)',
                   '    Metallic.OpenPBRPathTracing::dispatchOpenPbrRayQueryPathTraceGuides(id);'):
        if entry.count(anchor) != 1:
            raise ValueError("The fixed Guides entrypoint changed: " + anchor)
    source = sources[TARGET_SOURCE]
    normalized = source.replace("\r\n", "\n")
    if "\r" in normalized or normalized.count(ANCHOR) != 1:
        raise ValueError("The exact shadow function definition or its line endings changed")
    newline = "\r\n" if "\r\n" in source else "\n"
    if newline == "\r\n" and "\n" in source.replace("\r\n", ""):
        raise ValueError("Mixed target shader line endings")
    actual_anchor = ANCHOR.replace("\n", newline)
    index = source.index(actual_anchor)
    previous = source[:index].rstrip().splitlines()[-1]
    if previous.strip().startswith("["):
        raise ValueError("Target function already has an attribute")
    changed = source[:index] + ANNOTATION + newline + source[index:]
    if changed.replace(ANNOTATION + newline + actual_anchor, actual_anchor, 1) != source:
        raise ValueError("Changes escaped the single function annotation")
    validate_owned(changed, "implementing OpenPBRPathTracing;")
    files = {} if arm == ARMS[0] else {TARGET_SOURCE: changed}
    return dict(schema_version=1, arm=arm, label=arm, feature="general",
        mechanism="original" if arm == ARMS[0] else "selective-noinline",
        files=files, changed_files=list(files), source_hashes=hashes,
        generated_hashes={relative: sha256(text) for relative, text in files.items()},
        provider_file=None, provider_module=None, provider_source=None, provider_imports=[],
        transformation=dict(target_file=TARGET_SOURCE, target_function=TARGET_FUNCTION,
            annotation=ANNOTATION, signature=SIGNATURE, original_sha256=hashes[TARGET_SOURCE],
            annotated_sha256=sha256(changed), line_ending="CRLF" if newline == "\r\n" else "LF",
            only_annotation_changes=True),
        preserves_report_macros=True, preserves_resource_layout=True, preserves_rng_and_texture_order=True,
        retains_fiber=True, retains_conditional_slab=True, requires_no_fiber=False, extra_compiler_flags=[],
        caution="Source annotation is not proof of a retained SPIR-V boundary or native/GPU benefit; inspect raw and cleaned function controls and calls.")
