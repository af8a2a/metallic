"""Create fixed PT query differential fixtures and production-pass timing cases."""
import argparse
import copy
import json
from pathlib import Path
import struct

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
GUIDES = ["albedo", "specularAlbedo", "normalRoughness", "motionVectors",
          "linearDepth", "specularHitDistance", "depth"]


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def room(folder, mixed):
    """Opaque room with MASK, BLEND, normal mapped and nested dielectric surfaces."""
    folder.mkdir()
    data = bytearray()
    views, accessors, primitives = [], [], []

    def accessor(values, kind, components):
        offset = len(data)
        for v in values:
            data.extend(struct.pack("<" + "f" * components, *v))
        views.append(dict(buffer=0, byteOffset=offset, byteLength=len(data)-offset))
        item = dict(bufferView=len(views)-1, componentType=5126, count=len(values), type=kind)
        if kind == "VEC3":
            item.update(min=[min(v[c] for v in values) for c in range(3)],
                        max=[max(v[c] for v in values) for c in range(3)])
        accessors.append(item)
        return len(accessors)-1

    def quad(a, b, c, d, n, material):
        vertices = [a, b, c, a, c, d]
        attrs = dict(POSITION=accessor(vertices, "VEC3", 3),
                     NORMAL=accessor([n]*6, "VEC3", 3),
                     TEXCOORD_0=accessor([(0,0),(1,0),(1,1),(0,0),(1,1),(0,1)], "VEC2", 2))
        primitives.append(dict(attributes=attrs, material=material))

    def box(lo, hi, material):
        x,y,z = lo; X,Y,Z = hi
        quad((x,y,Z),(X,y,Z),(X,Y,Z),(x,Y,Z),(0,0,1),material)
        quad((X,y,z),(x,y,z),(x,Y,z),(X,Y,z),(0,0,-1),material)
        quad((X,y,Z),(X,y,z),(X,Y,z),(X,Y,Z),(1,0,0),material)
        quad((x,y,z),(x,y,Z),(x,Y,Z),(x,Y,z),(-1,0,0),material)
        quad((x,Y,Z),(X,Y,Z),(X,Y,z),(x,Y,z),(0,1,0),material)
        quad((x,y,z),(X,y,z),(X,y,Z),(x,y,Z),(0,-1,0),material)

    materials = [dict(pbrMetallicRoughness=dict(baseColorFactor=[.6,.5,.4,1],
                    metallicFactor=0, roughnessFactor=.5), doubleSided=True)]
    quad((-2,-1,-2),(2,-1,-2),(2,2,-2),(-2,2,-2),(0,0,1),0)
    quad((-2,-1,2),(2,-1,2),(2,-1,-2),(-2,-1,-2),(0,1,0),0)
    quad((-2,2,-2),(2,2,-2),(2,2,2),(-2,2,2),(0,-1,0),0)
    quad((-2,-1,-2),(-2,2,-2),(-2,2,2),(-2,-1,2),(1,0,0),0)
    quad((2,-1,2),(2,2,2),(2,2,-2),(2,-1,-2),(-1,0,0),0)
    images, textures = [], []
    if mixed:
        image = Image.new("RGBA", (8,8))
        image.putdata([(220,180,120,255 if (x+y)%2 else 0) for y in range(8) for x in range(8)])
        image.save(folder / "Coverage.png")
        Image.new("RGB", (4,4), (165,105,245)).save(folder / "Normal.png")
        Image.new("RGB", (4,4), (190,220,150)).save(folder / "Transmission.png")
        images = [dict(uri=n) for n in ["Coverage.png", "Normal.png", "Transmission.png"]]
        textures = [dict(source=i) for i in range(3)]
        materials[0]["normalTexture"] = dict(index=1, scale=.7)
        for alpha in ("MASK", "BLEND"):
            m = copy.deepcopy(materials[0])
            m.update(alphaMode=alpha, alphaCutoff=.5)
            m["pbrMetallicRoughness"].update(baseColorTexture=dict(index=0), baseColorFactor=[.5,.8,.3,.6])
            materials.append(m)
        for ior in (1.5, 1.33):
            materials.append(dict(pbrMetallicRoughness=dict(baseColorFactor=[.85,.95,1,1], metallicFactor=0,
                roughnessFactor=.1), doubleSided=True, extensions={
                "KHR_materials_transmission": dict(transmissionFactor=1, transmissionTexture=dict(index=2)),
                "KHR_materials_ior": dict(ior=ior),
                "KHR_materials_volume": dict(thicknessFactor=.3, attenuationDistance=.7,
                                             attenuationColor=[.8,.9,.95])}))
        quad((-1.7,-.8,.7),(-.7,-.8,.7),(-.7,1.2,.7),(-1.7,1.2,.7),(0,0,1),1)
        quad((.8,-.8,1.0),(1.7,-.8,1.0),(1.7,1.2,1.0),(.8,1.2,1.0),(0,0,1),2)
        quad((.7,-.8,.5),(1.6,-.8,.5),(1.6,1.2,.5),(.7,1.2,.5),(0,0,1),2)
        box((-.55,-.8,-.3),(.55,.8,.8),3)
        box((-.35,-.6,-.1),(.35,.6,.6),4)
    else:
        box((-.7,-1,-.2),(.7,.8,.8),0)
    (folder / "Scene.bin").write_bytes(data)
    model = dict(asset=dict(version="2.0"), scene=0, scenes=[dict(nodes=[0])], nodes=[dict(mesh=0)],
                 meshes=[dict(primitives=primitives)], materials=materials, bufferViews=views,
                 accessors=accessors, buffers=[dict(uri="Scene.bin", byteLength=len(data))])
    if mixed:
        model.update(images=images, textures=textures,
                     extensionsUsed=["KHR_materials_transmission", "KHR_materials_ior", "KHR_materials_volume"])
    save(folder / "Scene.gltf", model)
    return folder / "Scene.gltf"


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    opaque, mixed = room(output / "Opaque", False), room(output / "Mixed", True)
    env = ROOT / "External/RTXCR-Assets/EnvironmentMaps/studio_small_09_1k.hdr"
    if not env.is_file():
        raise FileNotFoundError(env)
    camera = dict(eye=[0,.3,4.5], center=[0,.3,0], up=[0,1,0], fovDegrees=55,
                  projection="perspective", znear=.001, zfar=100)

    def case(name, scene, samples=2, size=128, guides=True, cache=False, miss=False, legacy=False, perf=False):
        cam = copy.deepcopy(camera)
        if miss:
            cam["center"] = [0,.3,8]
        props = dict(path=str(scene), camera=cam, samples=samples, maxDepth=12, bsdf="gltf" if legacy else "openpbr",
                     accumulate=False, outputLinear=True, exportDenoiserGuides=guides,
                     cacheMode="sharc" if cache else "off")
        if cache:
            props.update({"sharc.entriesLog2":16, "sharc.updateStride":2})
        graph = dict(version=1, name=name, nodes=[dict(id=1, name="PathTrace", type="ScenePathTracePass", properties=props)],
                     edges=[], outputs=["PathTrace.color"] + (["PathTrace."+n for n in GUIDES] if guides else []))
        save(output / (name + ".json"), graph)
        spec = dict(id=name, graph=name+".json", scene=None, output="PathTrace.color", width=size, height=size,
                    environment=dict(path=str(env), intensity=1., rotationDegrees=25.),
                    requiredTiming=["PathTrace", "SHaRC query" if cache else "Path trace shading"])
        if not perf:
            spec["useRenderView"] = True
            spec["captureFrames"] = [8,9,10,11,12,13,14,15]
            spec["captureOutputs"] = ["PathTrace."+n for n in GUIDES] if guides else []
            spec["guideCameraStep"] = dict(node="PathTrace",delta=[.02,0,0])
            moved = copy.deepcopy(cam); moved["eye"][0] += .18
            spec["cameraFrames"] = [dict(frame=11,node="PathTrace",camera=moved)]
        return spec

    correct = [case("Opaque1",opaque,1), case("Opaque2",opaque), case("Mixed1",mixed,1),
               case("Mixed4",mixed,4), case("Sky",mixed,2,miss=True), case("MixedNoGuides",mixed,guides=False),
               case("MixedSharc",mixed,cache=True), case("LegacyShadow",mixed,guides=False,legacy=True)]
    perf = [case("OpaquePerf",opaque,size=512,perf=True), case("MixedPerf",mixed,size=512,perf=True)]
    original = json.loads((ROOT / "Pipelines/Samples/pathtracing_abeautiful_game_openpbr_dlss_rr.metallic_graph.json").read_text())
    node = next(n for n in original["nodes"] if n["name"]=="PathTrace")
    # Query-implementation A/B keeps its original workload as sample defaults evolve.
    node["properties"].update(samples=2, maxDepth=12, cacheMode="off")
    graph = dict(version=1,name="ABeautifulGamePerf", nodes=[node], edges=[],outputs=["PathTrace.color"])
    save(output / "ABeautifulGamePerf.json",graph)
    game = ROOT / node["properties"]["path"]
    perf.append(dict(id="ABeautifulGamePerf",graph="ABeautifulGamePerf.json",scene=None,output="PathTrace.color",
                     width=512,height=512,requiredTiming=["PathTrace","Path trace shading"],
                     environment=dict(path=str(env),intensity=1.,rotationDegrees=25.)))
    # Retain the production camera and the frame that exposed floating-point
    # drift when primary hit reconstruction was hoisted out of the path loop.
    gameCorrect = copy.deepcopy(perf[-1])
    gameCorrect.update(id="ABeautifulGame", frameCount=97, captureFrames=[93,94,95,96],
                       captureOutputs=["PathTrace."+n for n in GUIDES])
    correct.append(gameCorrect)
    assets = [str(p) for p in output.rglob("*") if p.is_file()] + [str(env),str(game)]
    model = json.loads(game.read_text())
    assets += [str(game.parent / i["uri"]) for i in model.get("buffers",[])+model.get("images",[])
               if i.get("uri") and not i["uri"].startswith("data:")]
    sidecar = game.with_suffix(".metallic_scene.json")
    if sidecar.is_file(): assets.append(str(sidecar))
    save(output / "Correctness.json",dict(version=2,frames=16,warmupFrames=4,timingFrames=4,cases=correct,assetFiles=assets))
    save(output / "Performance.json",dict(version=2,frames=97,warmupFrames=32,timingFrames=64,cases=perf,assetFiles=assets))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    prepare(parser.parse_args().output.resolve())
