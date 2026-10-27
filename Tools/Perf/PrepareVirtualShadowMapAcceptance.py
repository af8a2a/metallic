"""Prepare explicit VSM production replays; authored samples and assets stay unchanged."""
import argparse
from pathlib import Path
from VirtualShadowMapAcceptance import ROOT, read, save


def prepare(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.frames < 32 or args.frames > 16384:
        raise ValueError("Use 32..16384 frames")
    look_camera = read(ROOT / "Pipelines/Samples/virtual_shadow_map_p4.metallic_graph.json")["view"]["camera"]
    catalog = [
        ("LookDev", "realtime_lighting_vsm", "Asset/LookDev/OpenPbrDefault/OpenPbrVSMQuality.metallic_scene.json", look_camera, False),
        ("Sponza", "realtime_lighting_vsm", "Asset/Sponza/glTF/Sponza.gltf",
         dict(eye=[8, 2, 0], center=[0, 2, 0], up=[0, 1, 0], fovDegrees=60, znear=.05, zfar=200, projection="perspective"), False),
        ("MiniZorah", "gpu_driven_realtime_vsm", "Asset/MiniZorah/zorah_main_public.v2.gltf", None, True),
        ("ZorahFull", "gpu_driven_zorah_full_vsm", "Asset/ZorahFull/zorah_textured_public.v1.gltf", None, True)]
    cases = []
    for identifier, sample, scene, camera, metadata in catalog:
        if args.case and identifier not in args.case:
            continue
        if not (ROOT / scene).is_file():
            raise FileNotFoundError(ROOT / scene)
        graph = read(ROOT / f"Pipelines/Samples/{sample}.metallic_graph.json")
        for node in graph["nodes"]:
            if node["name"] in ("VBuffer", "Deferred"):
                node["properties"]["path"] = scene
        if camera is not None:
            graph["view"]["camera"] = camera
        if args.core_only:
            removed = {"DLSSSR", "DLSSNR"}
            graph["nodes"] = [n for n in graph["nodes"] if n["name"] not in removed]
            graph["edges"] = [e for e in graph["edges"] if e["src"].split(".")[0] not in removed
                              and e["dst"].split(".")[0] not in removed]
            next_id = max(e["id"] for e in graph["edges"]) + 1
            graph["edges"] += [dict(id=next_id, src="Deferred.color", dst="AutoExposure.source"),
                               dict(id=next_id + 1, src="AutoExposure.color", dst="FinalBlit.source")]
        save(output / f"{identifier}.json", graph)
        case = dict(id=identifier, graph=f"{identifier}.json", scene=scene, metadataOnly=metadata,
                    enableStreamline=not args.core_only, synchronizationValidation=args.core_only and args.mode == "diagnostic",
                    requireNoRayCapabilities=args.core_only, temporalJitter=args.mode == "timing",
                    width=768, height=512,
                    output="FinalBlit.color", comparisonOutput="Deferred.color", camera=graph["view"]["camera"],
                    lighting=dict(sun=dict(direction=[-.711269, -.479014, -.514434], illuminance=3., angularRadius=.00465)),
                    environment=dict(enabled=True, path="Asset/ABeautifulGame/environment.hdr", intensity=1., visible=True),
                    exposureEV100=0, requireShadow=True)
        if args.mode == "diagnostic":
            case.update(captureFrames=[0, args.frames - 1], comparisonFrames=[16, args.frames - 1])
        if args.motion:
            case["cameraMotion"] = dict(axis=0, amplitude=args.motion, period=128)
        cases.append(case)
        save(output / f"{identifier}Cases.json", dict(version=1, mode=args.mode, frames=args.frames, warmupFrames=16, cases=[case]))
    save(output / "Catalog.json", {"caseManifests": [f"{case['id']}Cases.json" for case in cases],
                                   "execution": "Run each case manifest in a separate process"})
    for case in cases:
        print(output / f"{case['id']}Cases.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=["diagnostic", "timing"], default="diagnostic")
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--case", action="append", choices=["LookDev", "Sponza", "MiniZorah", "ZorahFull"])
    parser.add_argument("--motion", type=float, default=0, help="World-space sinusoidal camera translation amplitude")
    parser.add_argument("--core-only", action="store_true", help="Remove DLSS for a strict no-RT-capability synchronization-validation run")
    prepare(parser.parse_args())
