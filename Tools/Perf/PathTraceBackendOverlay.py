"""Plan a coarse material-backend Type experiment on frozen Guides shaders.

This module only reads inputs and returns generated source text. The caller
copies the complete shader/include graph first, applies ``files`` beneath its
frozen Shaders directory, audits ownership, then compiles consumer and provider
with the same options. No production shader is edited by this helper.

No-Fiber removes only the found-vertex Fiber dispatch and independent Fiber
guide evaluation. It requires scene-wide eligibility across all reachable and
streamed materials. Shadow traversal can still reference Fiber code; neither
global dead-code elimination nor numerical equivalence is established here.
"""

import hashlib
from pathlib import Path
import re


ARMS = (
    "original-general",
    "backend-concrete-general",
    "backend-link-general",
    "backend-concrete-no-fiber",
    "backend-link-no-fiber",
)
ARM_ALIASES = {
    "original": "original-general",
    "concrete-general": "backend-concrete-general",
    "link-general": "backend-link-general",
    "concrete-no-fiber": "backend-concrete-no-fiber",
    "link-no-fiber": "backend-link-no-fiber",
}
OWNER_SOURCE = "Modules/OpenPBRPathTracing.slang"
VERTEX_SOURCE = "Modules/OpenPBRPathTracing/OpenPBRPathVertex.slang"
SURFACE_SOURCE = "Modules/OpenPBRSurface.slang"
GUIDES_SOURCE = "Modules/OpenPBRPathTracing/OpenPBRGuides.slang"
ENTRY_SOURCE = "Features/PathTracing/OpenPBRRayQueryPathTraceGuides.slang"
BACKEND_SOURCE = "Modules/OpenPBRPathTracing/PathTraceMaterialBackendExperiment.slang"
SOURCE_FILES = (OWNER_SOURCE, VERTEX_SOURCE, SURFACE_SOURCE, GUIDES_SOURCE, ENTRY_SOURCE)
FIBER_BRANCH = "    if (materialDomain(transport) == kMaterialDomainFiber) {"
COVERAGE_BRANCH = "    if (transport.textureParams.w > 1.5 && push.debugView == kOpenPBRDebugViewFinal) {"
TRANSPORT_LOAD = "    SceneMaterialData transport = loadSceneMaterial(hit.materialIndex);"
GUIDE_CALL = "    SceneSurfaceSummary summary = evaluateSceneSurfaceGuides(hit, push, rayCone, ray.direction, seed);"
OWNER_INCLUDE = '__include "OpenPBRPathTracing/OpenPBRPathContinuation.slang";'
VERTEX_SIGNATURE = """public OpenPBRPathControl shadeOpenPBRPathVertex(
    bool found, HitInfo hit, ScenePathTracePush push,
    inout Ray ray, inout RayCone rayCone, inout uint seed, inout uint depth,
    inout float3 radiance, inout float3 throughput,
    inout float previousBsdfPdf, inout bool previousScatterDelta,
    inout OpenPBRInteriorStack interiorStack,
    inout float specularHitDistance, inout bool captureSpecularHitDistance,
    inout uint alphaLayers, inout float3 primaryEndpoint,
    float3 primaryOrigin, inout bool primarySurface
#if (SHARC_UPDATE || SHARC_QUERY)
    , inout OpenPBRCacheContext cache
#endif
)"""
GUIDE_SIGNATURE = """public SceneSurfaceSummary evaluateSceneSurfaceGuides(inout HitInfo hit, ScenePathTracePush push,
    RayCone rayCone, float3 rayDirection, inout uint seed)"""
VERTEX_ARGUMENTS = """found, hit, push, ray, rayCone, seed, depth,
            radiance, throughput, previousBsdfPdf, previousScatterDelta,
            interiorStack, specularHitDistance, captureSpecularHitDistance,
            alphaLayers, primaryEndpoint, primaryOrigin, primarySurface
#if (SHARC_UPDATE || SHARC_QUERY)
            , cache
#endif"""
GUIDE_ARGUMENTS = "hit, push, rayCone, rayDirection, seed"


def sha256(value):
    """Hash raw bytes, or generated UTF-8 text with its explicit LF endings."""
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("Expected exactly one shader overlay anchor: " + before)
    return source.replace(before, after)


def brace_end(source, opening):
    """Return the exclusive closing-brace offset, ignoring comments/literals."""
    if opening >= len(source) or source[opening] != "{":
        raise ValueError("Expected an opening shader brace")
    depth = 0
    index = opening
    while index < len(source):
        if source.startswith("//", index):
            newline = source.find("\n", index + 2)
            index = len(source) if newline < 0 else newline + 1
            continue
        if source.startswith("/*", index):
            closing = source.find("*/", index + 2)
            if closing < 0:
                raise ValueError("Unterminated shader block comment")
            index = closing + 2
            continue
        char = source[index]
        if char in ('"', "'"):
            quote = char
            index += 1
            while index < len(source):
                if source[index] == "\\":
                    index += 2
                elif source[index] == quote:
                    index += 1
                    break
                else:
                    index += 1
            else:
                raise ValueError("Unterminated shader string literal")
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    raise ValueError("Unbalanced shader function or branch")


def function_parts(source, signature):
    """Require the exact function declaration and return its body offsets."""
    if source.count(signature) != 1:
        raise ValueError("Expected exactly one unchanged function signature: " + signature.splitlines()[0])
    start = source.index(signature)
    opening = start + len(signature)
    if source[opening:opening + 2] != "\n{":
        raise ValueError("Expected the established Allman function body")
    opening += 1
    ending = brace_end(source, opening)
    return start, opening, ending, source[opening + 1:ending - 1]


def remove_fiber_branch(body, next_anchor):
    """Delete the complete standalone branch; preserve every remaining byte."""
    if body.count(FIBER_BRANCH) != 1:
        raise ValueError("Expected exactly one material-domain Fiber branch")
    start = body.index(FIBER_BRANCH)
    ending = brace_end(body, start + len(FIBER_BRANCH) - 1)
    if not body.startswith("\n" + next_anchor, ending):
        raise ValueError("Fiber dispatch is no longer followed by the expected unchanged path")
    removed = body[start:ending + 1]
    return body[:start] + body[ending + 1:], removed


def validate_owned(source, owner, namespace=None):
    declarations = re.findall(r"^\s*(?:module|implementing)\s+[\w.]+\s*;", source, re.M)
    if len(declarations) != 1 or declarations[0].strip() != owner:
        raise ValueError("Shader overlay ownership changed: " + owner)
    if namespace and len(re.findall(r"^" + re.escape(namespace) + r"\s*\{", source, re.M)) != 1:
        raise ValueError("Shader overlay namespace changed: " + namespace)


def load_sources(shader_root):
    sources = {}
    hashes = {}
    for relative in SOURCE_FILES:
        path = shader_root / relative
        if not path.is_file():
            raise ValueError("Expected frozen shader input: " + str(path))
        raw = path.read_bytes()
        sources[relative] = raw.decode("utf-8").replace("\r\n", "\n")
        if "\r" in sources[relative]:
            raise ValueError("Unsupported shader line ending: " + str(path))
        hashes[relative] = sha256(raw)
    for relative, owner, namespace in (
            (OWNER_SOURCE, "module OpenPBRPathTracing;", None),
            (VERTEX_SOURCE, "implementing OpenPBRPathTracing;", "namespace Metallic.OpenPBRPathTracing"),
            (SURFACE_SOURCE, "module OpenPBRSurface;", "namespace Metallic.SceneOpenPBR"),
            (GUIDES_SOURCE, "implementing OpenPBRPathTracing;", "namespace Metallic.OpenPBRPathTracing"),
            (ENTRY_SOURCE, "module OpenPBRRayQueryPathTraceGuides;", None)):
        validate_owned(sources[relative], owner, namespace)
    entry = sources[ENTRY_SOURCE]
    for anchor in ('import OpenPBRPathTracing;', '[shader("compute")]', '[numthreads(8, 8, 1)]',
                   'void openPbrRayQueryPathTraceGuidesMain(uint3 id : SV_DispatchThreadID)',
                   '    Metallic.OpenPBRPathTracing::dispatchOpenPbrRayQueryPathTraceGuides(id);'):
        if entry.count(anchor) != 1:
            raise ValueError("Real Guides entrypoint layout changed: " + anchor)
    if sources[OWNER_SOURCE].count(OWNER_INCLUDE) != 1 or sources[GUIDES_SOURCE].count(GUIDE_CALL) != 1:
        raise ValueError("Shared owner include or independent guide call changed")
    if (shader_root / BACKEND_SOURCE).exists():
        raise ValueError("Shader input already contains a backend experiment")
    return sources, hashes


def vertex_edits(source):
    start, opening, ending, body = function_parts(source, VERTEX_SIGNATURE)
    if body.count(TRANSPORT_LOAD) != 1:
        raise ValueError("Found-vertex full material load changed")
    boundary = body.index(TRANSPORT_LOAD)
    prefix, general = body[:boundary], body[boundary:]
    no_fiber, removed = remove_fiber_branch(general, COVERAGE_BRANCH)
    # These anchors define this experiment's transport, alpha and Slab boundary.
    for anchor in ("    rayCone = propagateRayCone(rayCone, hit.t);",
                   "    if (!openPbrDebugFlag(push, kOpenPBRDebugDisableVolumeAttenuation)) {",
                   "    if (luminanceMax(throughput) <= 0.000001) {"):
        if prefix.count(anchor) != 1:
            raise ValueError("Common vertex prefix changed: " + anchor)
    for anchor in (COVERAGE_BRANCH, "        if (rand(seed) >= alpha) {", "            --depth;",
                   "#if METALLIC_CUSTOM_MATERIALS", "makeSceneSlabMaterialProgram<SceneSingleSlabModel>",
                   "makeSceneSlabMaterialProgram<SceneDualSlabModel>", "    return shadeOpenPBRSurfacePathVertex(hit, push,"):
        if general.count(anchor) != 1 or no_fiber.count(anchor) != 1:
            raise ValueError("Found-vertex coverage/Slab/OpenPBR layout changed: " + anchor)
    if general.count("shadeSceneFiber(") != 1 or "shadeSceneFiber(" in no_fiber or "kMaterialDomainFiber" in no_fiber:
        raise ValueError("Unexpected Fiber calls outside the isolated vertex branch")
    helpers = []
    for name, implementation in (("shadeOpenPBRFoundMaterialGeneral", general),
                                 ("shadeOpenPBRFoundMaterialNoFiber", no_fiber)):
        signature = VERTEX_SIGNATURE.replace("shadeOpenPBRPathVertex", name)
        helpers.append(signature + "\n{\n" + implementation + "}\n\n")
    call = "    return ScenePathTraceMaterialBackend.shadeFoundMaterial(\n        "
    call += VERTEX_ARGUMENTS.replace("            ", "        ") + "\n    );\n"
    changed = source[:start] + "".join(helpers) + source[start:opening + 1] + prefix + call + source[ending - 1:]
    return changed, dict(common_prefix_sha256=sha256(prefix), general_dispatch_sha256=sha256(general),
                         no_fiber_dispatch_sha256=sha256(no_fiber), removed_vertex_fiber_sha256=sha256(removed))


def surface_edits(source):
    _, _, ending, body = function_parts(source, GUIDE_SIGNATURE)
    next_anchor = "    let context = sceneSurfaceMaterialContextForRay(hit, rayCone, rayDirection);"
    no_fiber, removed = remove_fiber_branch(body, next_anchor)
    for anchor in ("    let transport = loadSceneMaterial(hit.materialIndex);", next_anchor,
                   "#if METALLIC_CUSTOM_MATERIALS", "makeSceneSlabMaterialProgram<SceneSingleSlabModel>",
                   "makeSceneSlabMaterialProgram<SceneDualSlabModel>",
                   "makeOpenPBRMaterialProgram(hit, push, rayCone, rayDirection, seed, true, false)"):
        if body.count(anchor) != 1 or no_fiber.count(anchor) != 1:
            raise ValueError("Independent guide material policy changed: " + anchor)
    if body.count("summarizeFiberGuides(") != 1 or "summarizeFiberGuides(" in no_fiber or "kMaterialDomainFiber" in no_fiber:
        raise ValueError("Unexpected Fiber evaluation outside the isolated guide branch")
    signature = GUIDE_SIGNATURE.replace("evaluateSceneSurfaceGuides", "evaluateSceneSurfaceGuidesNoFiber")
    helper = "\n\n" + signature + "\n{" + no_fiber + "}"
    return source[:ending] + helper + source[ending:], dict(
        general_guides_sha256=sha256(body), no_fiber_guides_sha256=sha256(no_fiber),
        removed_guide_fiber_sha256=sha256(removed))


def method_signature(signature, old_name, new_name):
    lines = signature.replace("public ", "public static ", 1).replace(old_name, new_name, 1).splitlines()
    return "\n".join(line if line.startswith("#") else "    " + line for line in lines)


def backend_source(mechanism, feature):
    vertex = method_signature(VERTEX_SIGNATURE, "shadeOpenPBRPathVertex", "shadeFoundMaterial")
    guides = method_signature(GUIDE_SIGNATURE, "evaluateSceneSurfaceGuides", "evaluateGuides")
    source = """#language slang 2026
implementing OpenPBRPathTracing;

import Core;
import Material;
import SceneSurface;
import SceneSurfaceMaterial;
import ScenePathTypes;
import ScenePathTracing;
import OpenPBRSurface;
import SHARC;

namespace Metallic.OpenPBRPathTracing
{
using Metallic;
using Metallic.Material;
using Metallic.SceneSurface;
using Metallic.ScenePathTracing;

// A complete material dispatch boundary, rather than boolean policy fields.
public interface IScenePathTraceMaterialBackend
{
"""
    source += vertex + ";\n" + guides + ";\n};\n\n"
    for name, vertex_helper, guide_helper in (
            ("GeneralScenePathTraceMaterialBackend", "shadeOpenPBRFoundMaterialGeneral", "evaluateSceneSurfaceGuides"),
            ("NoFiberScenePathTraceMaterialBackend", "shadeOpenPBRFoundMaterialNoFiber", "evaluateSceneSurfaceGuidesNoFiber")):
        source += "public struct " + name + " : IScenePathTraceMaterialBackend\n{\n"
        source += vertex + "\n    {\n        return " + vertex_helper + "(\n            " + VERTEX_ARGUMENTS + "\n        );\n    }\n"
        source += guides + "\n    {\n        return Metallic.SceneOpenPBR::" + guide_helper + "(" + GUIDE_ARGUMENTS + ");\n    }\n};\n\n"
    if mechanism == "concrete":
        selected = "General" if feature == "general" else "NoFiber"
        source += "typealias ScenePathTraceMaterialBackend = " + selected + "ScenePathTraceMaterialBackend;\n"
    else:
        source += "extern struct ScenePathTraceMaterialBackend : IScenePathTraceMaterialBackend;\n"
    return source + "} // namespace Metallic.OpenPBRPathTracing\n"


def provider_source(feature):
    selected = "General" if feature == "general" else "NoFiber"
    module = "PathTraceMaterialBackendLink" + selected + "Provider"
    source = "#language slang 2026\nmodule " + module + ";\nimport OpenPBRPathTracing;\n\n"
    source += "namespace Metallic.OpenPBRPathTracing\n{\n"
    source += "export struct ScenePathTraceMaterialBackend : IScenePathTraceMaterialBackend = "
    source += selected + "ScenePathTraceMaterialBackend;\n} // namespace Metallic.OpenPBRPathTracing\n"
    return "Modules/" + module + ".slang", module, source


def plan_overlay(shader_root: Path, arm: str) -> dict:
    """Return owned sources and provenance without writing or compiling them.

    ``shader_root`` is the frozen Shaders directory. ``files`` paths are POSIX
    paths relative to it. Five canonical labels are in ARMS; the documented
    short names in ARM_ALIASES normalize to those same labels. Existing report
    macros remain unchanged. The initial harness restricts CUSTOM_MATERIALS=0,
    SHARC off; conditional Slab and SHARC code remains in the generated sources.
    """
    arm = ARM_ALIASES.get(arm, arm)
    if arm not in ARMS:
        raise ValueError("Unknown path-trace backend arm: " + str(arm))
    shader_root = Path(shader_root)
    sources, hashes = load_sources(shader_root)
    # Validate both boundaries even for the unchanged baseline, so all arms
    # share exactly the same known source layout and fail together on drift.
    vertex, vertex_metadata = vertex_edits(sources[VERTEX_SOURCE])
    surface, guide_metadata = surface_edits(sources[SURFACE_SOURCE])
    feature = "no-fiber" if arm.endswith("no-fiber") else "general"
    mechanism = "original" if arm == "original-general" else arm.split("-")[1]
    files = {}
    provider_file = provider_module = provider_text = None
    if mechanism != "original":
        files = {
            OWNER_SOURCE: replace_once(sources[OWNER_SOURCE], OWNER_INCLUDE,
                OWNER_INCLUDE + '\n__include "OpenPBRPathTracing/PathTraceMaterialBackendExperiment.slang";'),
            VERTEX_SOURCE: vertex,
            SURFACE_SOURCE: surface,
            GUIDES_SOURCE: replace_once(sources[GUIDES_SOURCE], GUIDE_CALL,
                "    SceneSurfaceSummary summary = ScenePathTraceMaterialBackend.evaluateGuides(hit, push, rayCone, ray.direction, seed);"),
            BACKEND_SOURCE: backend_source(mechanism, feature),
        }
        if mechanism == "link":
            provider_file, provider_module, provider_text = provider_source(feature)
            if (shader_root / provider_file).exists():
                raise ValueError("Shader input already contains an experiment provider")
            files[provider_file] = provider_text
    for relative, source in files.items():
        owner = "module OpenPBRSurface;" if relative == SURFACE_SOURCE else "implementing OpenPBRPathTracing;"
        if relative == OWNER_SOURCE:
            owner = "module OpenPBRPathTracing;"
        elif relative == provider_file:
            owner = "module " + provider_module + ";"
        validate_owned(source, owner)
    return dict(schema_version=1, arm=arm, mechanism=mechanism, feature=feature,
        files=files, provider_file=provider_file, provider_module=provider_module,
        provider_source=provider_text, provider_imports=["OpenPBRPathTracing"] if provider_file else [],
        feature_mask=dict(SheenAndCoat=False, Dispersion=False, Translucency=True, Metallic=True,
                          Fiber=feature == "general"),
        source_hashes=hashes, generated_hashes={relative: sha256(source) for relative, source in files.items()},
        transformation=dict(**vertex_metadata, **guide_metadata),
        preserves_report_macros=True, preserves_resource_layout=True, retains_conditional_slab=True,
        extra_compiler_flags=[],
        caution="Unmeasured experimental implementation selection. No-Fiber requires scene-wide eligibility; shadow Fiber reachability and GPU equivalence remain unproven.")
