"""Plan full-feature path-tracing material field reads on frozen shader sources.

This helper only reads inputs and returns source texts and provenance metadata.
It never edits production shaders, launches a compiler, or selects a reduced
material backend. Callers freeze the complete shader/include graph, apply the
returned ``files`` under that frozen Shaders directory, audit module ownership,
and compile the original and candidate with identical request/options.

The existing SceneMaterialData load has no record bounds/default contract. The
overlay preserves its valid complete-record/index precondition and does not add
GetDimensions checks or reinterpret optional handles. It removes the initial
720-byte temporary from found-vertex and guide classification, then reads Fiber
or BLEND coverage fields only inside their existing branches.
"""

import hashlib
from pathlib import Path
import re


ARMS = ("original-general", "material-narrow-general", "material-member-general")
OWNER_SOURCE = "Modules/SceneSurface.slang"
VERTEX_SOURCE = "Modules/OpenPBRPathTracing/OpenPBRPathVertex.slang"
SURFACE_SOURCE = "Modules/OpenPBRSurface.slang"
GUIDES_SOURCE = "Modules/OpenPBRPathTracing/OpenPBRGuides.slang"
READ_SOURCE = "Modules/SceneSurfaceMaterialReads.slang"
MATERIAL_SOURCE = "Modules/Material/SceneMaterial.slang"
RUNTIME_SOURCE = "Modules/Material/MaterialRuntime.slang"
COVERAGE_SOURCE = "Modules/Material/Coverage.slang"
FIBER_SOURCE = "Modules/RTXCRMaterialParameters.slang"
SOURCE_FILES = (OWNER_SOURCE, VERTEX_SOURCE, SURFACE_SOURCE, GUIDES_SOURCE,
                MATERIAL_SOURCE, RUNTIME_SOURCE, COVERAGE_SOURCE, FIBER_SOURCE)
CPU_ABI_HEADER = "Source/Runtime/Render/Material/SceneMaterialData.h"
OFFSETS = dict(base_alpha=12, cutoff=40, program_id=56, alpha_mode=60,
               has_program=92, fiber=112, base_color_texture=176)
OWNER_ANCHOR = 'import StreamTextureSampling;\nimport WorkingColor;'
VERTEX_LOAD = "    SceneMaterialData transport = loadSceneMaterial(hit.materialIndex);"
GUIDE_LOAD = "    let transport = loadSceneMaterial(hit.materialIndex);"
DOMAIN_BRANCH = "    if (materialDomain(transport) == kMaterialDomainFiber) {"
COVERAGE_BRANCH = "    if (transport.textureParams.w > 1.5 && push.debugView == kOpenPBRDebugViewFinal) {"
GUIDE_SIGNATURE = """public SceneSurfaceSummary evaluateSceneSurfaceGuides(inout HitInfo hit, ScenePathTracePush push,
    RayCone rayCone, float3 rayDirection, inout uint seed)"""


def sha256(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError("Expected exactly one material-read overlay anchor: " + before)
    return source.replace(before, after)


def validate_owned(source, owner):
    declarations = re.findall(r"^\s*(?:module|implementing)\s+[\w.]+\s*;", source, re.M)
    if len(declarations) != 1 or declarations[0].strip() != owner:
        raise ValueError("Material-read shader ownership changed: " + owner)


def struct_body(source, name):
    match = re.search(r"public struct " + re.escape(name) + r"\s*\{([^{}]*)\};", source, re.S)
    if match is None:
        raise ValueError("Expected the established material struct: " + name)
    return match.group(1)


def member_declarations(source, name):
    body = re.sub(r"//[^\n]*", "", struct_body(source, name))
    fields = re.findall(r"public\s+(\w+)\s+(\w+)\s*;", body)
    remainder = re.sub(r"public\s+\w+\s+\w+\s*;", "", body)
    if remainder.strip():
        raise ValueError("Unsupported material declaration/layout: " + name)
    return fields


def validate_layout(sources):
    # Exact authored order is part of this experiment's byte addressing. CPU
    # sizeof/offsetof assertions are frozen separately by the preparation tool.
    prefix = ("baseColor", "emissive", "params", "textureParams", "glassParams",
              "attenuationColor", "diffuseTransmission", "rtxcrHairBaseColor",
              "rtxcrHairParams0", "rtxcrHairParams1", "rtxcrHairDiffuseTint")
    textures = ("baseColorTexture", "metallicRoughnessTexture", "normalTexture",
                "occlusionTexture", "emissiveTexture", "transmissionTexture",
                "thicknessTexture", "diffuseTransmissionTexture", "diffuseTransmissionColorTexture")
    expected = [("float4", name) for name in prefix]
    expected += [("MaterialTextureInfo", name) for name in textures]
    expected += [("float4", "specular"), ("MaterialTextureInfo", "specularTexture"),
                 ("MaterialTextureInfo", "specularColorTexture")]
    if member_declarations(sources[MATERIAL_SOURCE], "SceneMaterialData") != expected:
        raise ValueError("SceneMaterialData wire layout changed")
    if member_declarations(sources[MATERIAL_SOURCE], "MaterialTextureInfo") != [
            ("uint", "textureIndex"), ("uint", "texCoord"), ("uint2", "alignmentPadding"),
            ("float4", "transform0"), ("float4", "transform1")]:
        raise ValueError("MaterialTextureInfo wire layout changed")
    if member_declarations(sources[FIBER_SOURCE], "RTXCRHairMaterialParameters") != [
            ("float4", "baseColor"), ("float4", "roughnessIOR"), ("float4", "melanin"),
            ("float4", "diffuseTint")]:
        raise ValueError("Fiber parameter layout changed")
    for anchor in ("return uint(parameters.textureParams.z);",
                   "return materialProgramId(parameters) == kMaterialProgramRTXCRChiang ? kMaterialDomainFiber : kMaterialDomainSurface;"):
        if sources[RUNTIME_SOURCE].count(anchor) != 1:
            raise ValueError("Established material classification changed: " + anchor)
    for anchor in ("result.baseAlpha = data.baseColor.a;", "result.cutoff = data.params.z;",
                   "result.alphaMode = data.textureParams.w;", "result.hasProgram = data.attenuationColor.w > 0.5;",
                   "result.baseColorTexture = data.baseColorTexture;"):
        if sources[COVERAGE_SOURCE].count(anchor) != 1:
            raise ValueError("Established coverage extraction changed: " + anchor)
    for anchor in ("result.baseColor = data.rtxcrHairBaseColor;", "result.roughnessIOR = data.rtxcrHairParams0;",
                   "result.melanin = data.rtxcrHairParams1;", "result.diffuseTint = data.rtxcrHairDiffuseTint;"):
        if sources[FIBER_SOURCE].count(anchor) != 1:
            raise ValueError("Established Fiber extraction changed: " + anchor)


def read_source():
    return """#language slang 2026
implementing SceneSurface;

import Core;
import Material;
import ScenePathTypes;
import RTXCRMaterialParameters;

namespace Metallic.SceneSurface
{
using Metallic;
using Metallic.Material;

// Experimental read-only views into the existing 720-byte upload record.
// Like loadSceneMaterial, these require a valid complete record and index.
// No new bounds/default policy or optional-handle interpretation is introduced.
public ByteAddressBuffer sceneMaterialReadBuffer()
{
#if SHARC_UPDATE
    return resolveUniform(getResourceParameters<SceneResourceParameters>().materials);
#else
    return resolveUniform(getResourceParameterField<SceneResourceParameters,
        ResourceHandle<ByteAddressBuffer>>(kSceneMaterialsOffset));
#endif
}

public struct SceneMaterialReadHeader
{
    public float programId;
    public float alphaMode;
};

public SceneMaterialReadHeader loadSceneMaterialReadHeader(uint materialIndex)
{
    float2 fields = sceneMaterialReadBuffer().Load<float2>(materialIndex * kSceneMaterialRecordBytes + 56u);
    SceneMaterialReadHeader header;
    header.programId = fields.x;
    header.alphaMode = fields.y;
    return header;
}

public float loadSceneMaterialReadProgramId(uint materialIndex)
{
    return sceneMaterialReadBuffer().Load<float>(materialIndex * kSceneMaterialRecordBytes + 56u);
}

public uint sceneMaterialReadDomain(float programId)
{
    // Stored program IDs are floats. Preserve numeric uint conversion, not bits.
    return uint(programId) == kMaterialProgramRTXCRChiang ? kMaterialDomainFiber : kMaterialDomainSurface;
}

public RTXCRHairMaterialParameters loadSceneMaterialReadFiber(uint materialIndex)
{
    return sceneMaterialReadBuffer().Load<RTXCRHairMaterialParameters>(
        materialIndex * kSceneMaterialRecordBytes + 112u);
}

public MaterialCoverageParameters loadSceneMaterialReadCoverage(uint materialIndex, float alphaMode)
{
    let buffer = sceneMaterialReadBuffer();
    uint base = materialIndex * kSceneMaterialRecordBytes;
    MaterialCoverageParameters result;
    result.baseAlpha = buffer.Load<float>(base + 12u);
    result.cutoff = buffer.Load<float>(base + 40u);
    result.alphaMode = alphaMode;
    result.hasProgram = buffer.Load<float>(base + 92u) > 0.5;
    result.baseColorTexture = buffer.Load<MaterialTextureInfo>(base + 176u);
    return result;
}
} // namespace Metallic.SceneSurface
"""


def member_source():
    """Keep the same public API, but let Slang lower typed member projections.

    The compatibility buffer accessor remains available and unused by these
    readers. None of the member reader bodies uses Load, raw byte offsets or a
    complete SceneMaterialData value; the compiler decides its actual IR form.
    """
    source = read_source()
    source = replace_once(source,
        "// Experimental read-only views into the existing 720-byte upload record.",
        "// Experimental typed member projections from the existing upload record.")
    source = replace_once(source,
        "    float2 fields = sceneMaterialReadBuffer().Load<float2>(materialIndex * kSceneMaterialRecordBytes + 56u);",
        "    float2 fields = sceneMaterials()[materialIndex].textureParams.zw;")
    source = replace_once(source,
        "    return sceneMaterialReadBuffer().Load<float>(materialIndex * kSceneMaterialRecordBytes + 56u);",
        "    return sceneMaterials()[materialIndex].textureParams.z;")
    source = replace_once(source,
        "    return sceneMaterialReadBuffer().Load<RTXCRHairMaterialParameters>(\n"
        "        materialIndex * kSceneMaterialRecordBytes + 112u);",
        "    RTXCRHairMaterialParameters result;\n"
        "    result.baseColor = sceneMaterials()[materialIndex].rtxcrHairBaseColor;\n"
        "    result.roughnessIOR = sceneMaterials()[materialIndex].rtxcrHairParams0;\n"
        "    result.melanin = sceneMaterials()[materialIndex].rtxcrHairParams1;\n"
        "    result.diffuseTint = sceneMaterials()[materialIndex].rtxcrHairDiffuseTint;\n"
        "    return result;")
    source = replace_once(source,
        "    let buffer = sceneMaterialReadBuffer();\n"
        "    uint base = materialIndex * kSceneMaterialRecordBytes;\n",
        "")
    for before, after in (
            ("    result.baseAlpha = buffer.Load<float>(base + 12u);",
             "    result.baseAlpha = sceneMaterials()[materialIndex].baseColor.a;"),
            ("    result.cutoff = buffer.Load<float>(base + 40u);",
             "    result.cutoff = sceneMaterials()[materialIndex].params.z;"),
            ("    result.hasProgram = buffer.Load<float>(base + 92u) > 0.5;",
             "    result.hasProgram = sceneMaterials()[materialIndex].attenuationColor.w > 0.5;"),
            ("    result.baseColorTexture = buffer.Load<MaterialTextureInfo>(base + 176u);",
             "    result.baseColorTexture = sceneMaterials()[materialIndex].baseColorTexture;")):
        source = replace_once(source, before, after)
    return source


def plan_overlay(shader_root: Path, arm: str) -> dict:
    """Return deterministic owned texts; ``shader_root`` is the Shaders directory."""
    if arm not in ARMS:
        raise ValueError("Unknown material-read arm: " + str(arm))
    shader_root = Path(shader_root)
    if (shader_root / READ_SOURCE).exists():
        raise ValueError("Shader input already contains the material-read experiment")
    sources, hashes = {}, {}
    for relative in SOURCE_FILES:
        raw = (shader_root / relative).read_bytes()
        text = raw.decode("utf-8").replace("\r\n", "\n")
        if "\r" in text:
            raise ValueError("Unsupported shader line ending: " + relative)
        sources[relative], hashes[relative] = text, sha256(raw)
    for relative, owner in ((OWNER_SOURCE, "module SceneSurface;"),
                            (VERTEX_SOURCE, "implementing OpenPBRPathTracing;"),
                            (SURFACE_SOURCE, "module OpenPBRSurface;"),
                            (GUIDES_SOURCE, "implementing OpenPBRPathTracing;"),
                            (MATERIAL_SOURCE, "implementing Material;"),
                            (RUNTIME_SOURCE, "implementing Material;"),
                            (COVERAGE_SOURCE, "implementing Material;"),
                            (FIBER_SOURCE, "module RTXCRMaterialParameters;")):
        validate_owned(sources[relative], owner)
    validate_layout(sources)
    vertex = replace_once(sources[VERTEX_SOURCE], VERTEX_LOAD,
                          "    let materialHeader = loadSceneMaterialReadHeader(hit.materialIndex);")
    vertex = replace_once(vertex, DOMAIN_BRANCH,
                          "    if (sceneMaterialReadDomain(materialHeader.programId) == kMaterialDomainFiber) {")
    vertex = replace_once(vertex, "shadeSceneFiber(sceneFiberParameters(transport), hit, stableNormal, push, depth,",
                          "shadeSceneFiber(loadSceneMaterialReadFiber(hit.materialIndex), hit, stableNormal, push, depth,")
    vertex = replace_once(vertex, COVERAGE_BRANCH,
                          "    if (materialHeader.alphaMode > 1.5 && push.debugView == kOpenPBRDebugViewFinal) {")
    vertex = replace_once(vertex, "evaluateSceneCoverage(hit.materialIndex, sceneMaterialCoverageParameters(transport), hit.texcoord, push).opacity;",
                          "evaluateSceneCoverage(hit.materialIndex, loadSceneMaterialReadCoverage(hit.materialIndex, materialHeader.alphaMode), hit.texcoord, push).opacity;")
    surface = replace_once(sources[SURFACE_SOURCE], GUIDE_SIGNATURE + "\n{\n" + GUIDE_LOAD,
                           GUIDE_SIGNATURE + "\n{\n    float materialProgram = loadSceneMaterialReadProgramId(hit.materialIndex);")
    surface = replace_once(surface, DOMAIN_BRANCH,
                           "    if (sceneMaterialReadDomain(materialProgram) == kMaterialDomainFiber) {")
    surface = replace_once(surface, "return summarizeFiberGuides(sceneFiberParameters(transport));",
                           "return summarizeFiberGuides(loadSceneMaterialReadFiber(hit.materialIndex));")
    owner = replace_once(sources[OWNER_SOURCE], OWNER_ANCHOR,
                         OWNER_ANCHOR + '\n\n__include "SceneSurfaceMaterialReads.slang";')
    implementation = member_source() if arm == "material-member-general" else read_source()
    files = {} if arm == ARMS[0] else {OWNER_SOURCE: owner, VERTEX_SOURCE: vertex,
                                     SURFACE_SOURCE: surface, READ_SOURCE: implementation}
    for relative, text in files.items():
        expected_owner = "implementing SceneSurface;" if relative == READ_SOURCE else (
            "implementing OpenPBRPathTracing;" if relative == VERTEX_SOURCE else (
                "module SceneSurface;" if relative == OWNER_SOURCE else "module OpenPBRSurface;"))
        validate_owned(text, expected_owner)
    plan = dict(schema_version=1, arm=arm, label=arm,
        mechanism="original" if arm == ARMS[0] else "material-field-reads",
        feature="general", feature_mask=dict(Fiber=True, Slab=True, OpenPBR=True),
        files=files, changed_files=list(files), source_hashes=hashes,
        generated_hashes={relative: sha256(text) for relative, text in files.items()},
        provider_file=None, provider_module=None, provider_source=None, provider_imports=[],
        record_contract=dict(record_bytes=720, texture_info_bytes=48, fiber_parameter_bytes=64,
            cpu_abi_header=CPU_ABI_HEADER, offsets=OFFSETS.copy(),
            precondition="Valid resolved materials buffer, complete 720-byte record, valid index and non-overflowing index * 720.",
            bounds_policy="Preserve existing caller guards; no new GetDimensions, length or sentinel gate.",
            missing_policy="Existing loadSceneMaterial has no missing-record fallback; do not introduce CPU defaults or new zero values.",
            handle_policy="Same uniform handle resolver and SHARC_UPDATE/ordinary packet getter; preserve all handle bits."),
        field_reads=[dict(site="found-vertex header", offset=56, bytes=8, condition="at original initial transport load"),
            dict(site="guides program", offset=56, bytes=4, condition="at original initial transport load"),
            dict(site="Fiber parameters", offset=112, bytes=64, condition="inside each existing Fiber branch"),
            dict(site="BLEND baseAlpha", offset=12, bytes=4, condition="inside original final-view BLEND branch"),
            dict(site="BLEND cutoff", offset=40, bytes=4, condition="inside original final-view BLEND branch"),
            dict(site="BLEND hasProgram", offset=92, bytes=4, condition="inside original final-view BLEND branch"),
            dict(site="BLEND baseColorTexture", offset=176, bytes=48, condition="inside original final-view BLEND branch")],
        transformation=dict(vertex_before_sha256=sha256(sources[VERTEX_SOURCE]), vertex_after_sha256=sha256(vertex),
            surface_before_sha256=sha256(sources[SURFACE_SOURCE]), surface_after_sha256=sha256(surface),
            guides_unchanged_sha256=sha256(sources[GUIDES_SOURCE])),
        preserves_report_macros=True, preserves_resource_layout=True, retains_conditional_slab=True,
        preserves_rng_and_texture_order=True, requires_no_fiber=False, extra_compiler_flags=[],
        caution="Uncompiled field-read overlay; all material branches retained. No frontend/native/GPU benefit or invalid-record equivalence is established.")
    if arm == "material-member-general":
        plan.update(mechanism="material-member-reads", read_representation="typed-member-projection",
            caution="Uncompiled typed member overlay; source no longer passes a full record, but actual IR copies and frontend/native/GPU benefit remain unproven.")
        plan["record_contract"]["handle_policy"] = (
            "Use the existing typed sceneMaterials() getter; preserve handle bits and caller guards. "
            "The compatibility byte-buffer accessor is unused by member readers.")
        plan["record_contract"]["addressing_policy"] = (
            "No manual byte offsets or index * recordBytes in member readers; Slang lowers typed record indexing.")
        expressions = (("textureParams.zw",), ("textureParams.z",),
            ("rtxcrHairBaseColor", "rtxcrHairParams0", "rtxcrHairParams1", "rtxcrHairDiffuseTint"),
            ("baseColor.a",), ("params.z",), ("attenuationColor.w",), ("baseColorTexture",))
        for field, members in zip(plan["field_reads"], expressions):
            field["member_expressions"] = ["sceneMaterials()[materialIndex]." + member for member in members]
            field["byte_count_scope"] = "Selected authored field widths; no claim about actual IR or device memory reads."
    return plan
