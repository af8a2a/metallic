"""Rebuild the MaterialX default OpenPBR LookDev assets (Python standard library only).

The mesh streams and HDRI are unmodified MaterialX resources. The glTF material
and camera reproduce the Web Viewer's default material assignment and framing.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import urllib.request
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
REVISION = "0a6f5bde987a4ff7cbee08aa0f6fe64c3a5c3deb"
BASE_URL = f"https://raw.githubusercontent.com/AcademySoftwareFoundation/MaterialX/{REVISION}/"
SOURCES = {
    "shaderball.glb": ("resources/Geometry/shaderball.glb",
        "a47a52c37f6be60c963c766a6279d302d9940456b9d02972cc8a1bf9026a859a"),
    "san_giuseppe_bridge_split.hdr": ("resources/Lights/san_giuseppe_bridge_split.hdr",
        "72f55f98ed32d060e4a32d5f0e9c59c0297c9c669ee4e8a62a0b09154cc5ac0a"),
    "open_pbr_default.mtlx": ("resources/Materials/Examples/OpenPbr/open_pbr_default.mtlx",
        "0b111f05685d2c82e17d63713c6d3dbdb729b6f68a015ecb5cc828811d765b76"),
    "san_giuseppe_bridge_split.mtlx": ("resources/Lights/san_giuseppe_bridge_split.mtlx",
        "5f63a6fc71ccddfb6a8788523320ce80599f60164e6012a7113041024847c95b"),
    "LICENSE": ("LICENSE", None),
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=4) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path,
        help="Use previously downloaded reference files instead of accessing the network")
    args = parser.parse_args()
    assets = {}
    for name, (relative, checksum) in SOURCES.items():
        if args.source_dir:
            data = (args.source_dir / name).read_bytes()
        else:
            with urllib.request.urlopen(BASE_URL + relative, timeout=60) as response:
                data = response.read()
        actual = hashlib.sha256(data).hexdigest()
        if checksum and actual != checksum:
            raise ValueError(f"Reference checksum mismatch: {name}: {actual}")
        assets[name] = data

    glb = assets["shaderball.glb"]
    magic, version, length = struct.unpack_from("<4sII", glb)
    if magic != b"glTF" or version != 2 or length != len(glb):
        raise ValueError("Expected a glTF 2.0 binary")
    json_size, json_type = struct.unpack_from("<I4s", glb, 12)
    bin_start = 20 + json_size
    bin_size, bin_type = struct.unpack_from("<I4s", glb, bin_start)
    if json_type != b"JSON" or bin_type != b"BIN\0":
        raise ValueError("Unexpected reference GLB chunks")
    model = json.loads(glb[20:bin_start])
    material = ET.fromstring(assets["open_pbr_default.mtlx"]).find("open_pbr_surface")
    inputs = {item.attrib["name"]: item.attrib["value"] for item in material.findall("input")}
    color = [float(value) for value in inputs["base_color"].split(",")]
    model["materials"] = [{
        "name": "OpenPBR Default (MaterialX)",
        "pbrMetallicRoughness": {"baseColorFactor": color + [1.0],
            "metallicFactor": float(inputs["base_metalness"]),
            "roughnessFactor": float(inputs["specular_roughness"])},
        "extensions": {"KHR_materials_ior": {"ior": float(inputs["specular_ior"])}},
    }]
    model["extensionsUsed"] = ["KHR_materials_ior"]
    # FB_ngon_encoding has no effect on these triangle streams.
    for mesh in model["meshes"]:
        for primitive in mesh["primitives"]:
            primitive.pop("extensions", None)
    model["buffers"][0]["uri"] = "Shaderball.bin"
    model["asset"]["generator"] = "Metallic Tools/BuildOpenPbrLookDev.py"
    model["asset"]["copyright"] = "Copyright Contributors to the MaterialX Project; Apache-2.0"

    positions = [model["accessors"][p["attributes"]["POSITION"]]
        for mesh in model["meshes"] for p in mesh["primitives"]]
    low = [min(a["min"][i] for a in positions) for i in range(3)]
    high = [max(a["max"][i] for a in positions) for i in range(3)]
    center = [(a + b) * 0.5 for a, b in zip(low, high)]
    radius = math.sqrt(sum((b - a)**2 for a, b in zip(low, high))) * 0.5
    eye = [0.0, center[1], 2.0 * radius]
    yaw = math.atan2(eye[0] - center[0], eye[2] - center[2])
    model["cameras"] = [{"name": "MaterialX Reference", "type": "perspective",
        "perspective": {"yfov": math.radians(60.0), "znear": 0.05, "zfar": 100.0}}]
    model["scenes"][0]["nodes"].append(len(model["nodes"]))
    model["nodes"].append({"name": "MaterialX Reference Camera", "camera": 0,
        "translation": eye, "rotation": [0.0, math.sin(yaw * 0.5), 0.0, math.cos(yaw * 0.5)]})

    light = ET.fromstring(assets["san_giuseppe_bridge_split.mtlx"]).find("directional_light")
    light_inputs = {item.attrib["name"]: item.attrib["value"] for item in light.findall("input")}
    direction = [float(value) for value in light_inputs["direction"].split(",")]
    # MaterialX Web Viewer applies RY(+90) to emitted-light directions. Its
    # atan2(x,-z) environment projection after RY(+90) equals Metallic's atan2(z,x).
    direction = [direction[2], direction[1], -direction[0]]
    environment = {"enabled": True, "path": "san_giuseppe_bridge_split.hdr",
        "intensity": 1.0, "rotationDegrees": 0.0, "visible": True}
    lighting = {"exposureEV100": 0.0, "autoExposure": {"enabled": False, "compensation": 0.0},
        "lights": [{"name": "MaterialX Split Sun", "type": "directional", "enabled": True,
            "direction": direction, "color": [float(v) for v in light_inputs["color"].split(",")],
            "intensity": float(light_inputs["intensity"])}]}
    scene = {"version": 3, "source": "OpenPbrDefault.gltf", "sceneIndex": 0, "nodes": [],
        "world": {"environment": environment, "lighting": lighting}}
    camera = {"eye": eye, "center": center, "up": [0.0, 1.0, 0.0], "fovDegrees": 60.0,
        "projection": "perspective", "znear": 0.05, "zfar": 100.0, "reversedZ": True}
    scene_path = "Asset/LookDev/OpenPbrDefault/OpenPbrDefault.gltf"
    graph = {"version": 1, "name": "OpenPBR Default LookDev", "nodes": [
        {"id": 1, "name": "PathTrace", "type": "ScenePathTracePass",
            "position": {"x": 80.0, "y": 120.0},
            "properties": {"path": scene_path, "camera": camera, "samples": 4, "bsdf": "openpbr",
                "maxDepth": 12, "accumulate": True, "outputLinear": True}},
        {"id": 2, "name": "AutoExposure", "type": "AutoExposurePass",
            "position": {"x": 440.0, "y": 120.0}, "properties": {"toneCurve": "none"}},
        {"id": 3, "name": "FinalBlit", "type": "FinalBlitPass",
            "position": {"x": 800.0, "y": 120.0}, "properties": {}}],
        "edges": [{"id": 1, "src": "PathTrace.color", "dst": "AutoExposure.source"},
            {"id": 2, "src": "AutoExposure.color", "dst": "FinalBlit.source"}], "outputs": []}

    output = ROOT / "Asset/LookDev/OpenPbrDefault"
    output.mkdir(parents=True, exist_ok=True)
    (output / "Shaderball.bin").write_bytes(glb[bin_start + 8:bin_start + 8 + bin_size])
    for name in SOURCES:
        if name != "shaderball.glb":
            (output / name).write_bytes(assets[name])
    write_json(output / "OpenPbrDefault.gltf", model)
    write_json(output / "OpenPbrDefault.metallic_scene.json", scene)
    write_json(ROOT / "Pipelines/Samples/openpbr_lookdev.metallic_graph.json", graph)
    comparison = {"version": 1, "name": "LookDev Shading Comparison", "nodes": [
        {"id": index + 1, "name": name, "type": "ScenePathTracePass",
            "position": {"x": 80.0, "y": 80.0 + index * 360.0},
            "properties": {**graph["nodes"][0]["properties"], "bsdf": bsdf,
                "cameraSyncGroup": "LookDevComparison"}}
        for index, (name, bsdf) in enumerate((("OpenPBR", "openpbr"), ("Standard", "standard")))],
        "edges": [{"id": 1, "src": "OpenPBR.color", "dst": "Slider.sourceA"},
            {"id": 2, "src": "Standard.color", "dst": "Slider.sourceB"},
            {"id": 3, "src": "Slider.color", "dst": "AutoExposure.source"},
            {"id": 4, "src": "AutoExposure.color", "dst": "FinalBlit.source"}], "outputs": []}
    comparison["nodes"] += [
        {"id": 3, "name": "Slider", "type": "SliderDebugPass",
            "position": {"x": 440.0, "y": 220.0},
            "properties": {"splitPosition": 0.5, "orientation": "vertical", "swapSides": False}},
        {"id": 4, "name": "AutoExposure", "type": "AutoExposurePass",
            "position": {"x": 800.0, "y": 220.0}, "properties": {"toneCurve": "none"}},
        {"id": 5, "name": "FinalBlit", "type": "FinalBlitPass",
            "position": {"x": 1160.0, "y": 220.0}, "properties": {}}]
    write_json(ROOT / "Pipelines/Samples/lookdev_shading_compare.metallic_graph.json", comparison)
    write_json(output / "Reference.json", {"materialxRevision": REVISION,
        "viewer": "https://academysoftwarefoundation.github.io/MaterialX/?file=Materials/Examples/OpenPbr/open_pbr_default.mtlx",
        "sources": {name: {"url": BASE_URL + relative,
            "sha256": hashlib.sha256(assets[name]).hexdigest()} for name, (relative, _) in SOURCES.items()},
        "camera": camera, "materialInputs": inputs, "environmentRotationDegrees": 0.0,
        "display": "scene-linear Rec.709 -> fixed EV100 0 -> sRGB (no tone curve)",
        "capture": {"width": 768, "height": 768, "samplesPerPixel": 1024, "maxDepth": 12}})
    print(f"Created {output}; camera eye={eye}, center={center}")


if __name__ == "__main__":
    main()
