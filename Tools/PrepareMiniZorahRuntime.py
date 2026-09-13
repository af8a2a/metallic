"""Generate the MiniZorah first-frame profile from the validated M1 manifest."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def next_power_of_two(value: int) -> int:
    return 1 << (max(1, value) - 1).bit_length()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "Asset/MeshletCache/MiniZorahCook/MiniZorah.manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "Pipelines/Samples/gpu_driven_minizorah.metallic_graph.json")
    parser.add_argument("--page-mib", type=int, default=1024)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or manifest["payloadValidation"] != "all-pages":
        raise ValueError("A completed, fully validated M1 cache is required")
    page_bytes = args.page_mib * 1024 * 1024
    if not manifest["minimumResidentBytesIncludingOneStreamPage"] <= page_bytes < 2**32:
        raise ValueError("Page budget cannot hold the terminal cut, or exceeds 32-bit device offsets")
    roots = manifest["terminalPageCount"]
    properties = {
        "path": Path(manifest["source"]).resolve().relative_to(ROOT).as_posix(),
        "streamAssetPath": Path(manifest["asset"]).resolve().relative_to(ROOT).as_posix(),
        "streamAssetOnly": True, "autoBuildStreamAsset": False, "enableClusterRtx": False,
        "maxResidentBytes": page_bytes,
        # Variable-size pages are bounded by bytes. Dividing by the largest
        # page would cap the page count too early while leaving bytes unused.
        "maxResidentPages": 0,
        "maxLockedFallbackPages": next_power_of_two(roots),
        "maxActiveGroups": next_power_of_two(manifest["terminalInstanceGroups"] * 4),
        "maxPageUploadsPerFrame": 256, "maxPageLoadsInFlight": 1024, "pageLoadConcurrency": 4,
        "maxGpuPageRequests": 65536, "maxGpuPageUnloadRequests": 65536,
        "maxTraversalWorkers": 1024, "maxTraversalWorkItems": 1048576,
        "autoLod": True, "lodPixelError": 1.5, "debugColorMode": "shaded",
        "instanceFrustumCull": True, "instanceHzbCull": True,
        "clusterFrustumCull": True, "clusterNormalConeCull": False,
        "camera": {"eye": [55.34291, 6.4527273, 0.32432523],
                   "center": [46.38051, 5.7994967, 0.5308178], "up": [0, 1, 0],
                   "fovDegrees": 60, "znear": 0.020000003, "zfar": 29999.998, "reversedZ": True}}
    graph = {"name": "GPUDrivenMiniZorah", "version": 1,
             "nodes": [{"id": 1, "name": "GPUDriven", "type": "GPUDrivenStreamAssetPass",
                        "position": {"x": 320, "y": 280}, "properties": properties},
                       {"id": 2, "name": "FinalBlit", "type": "FinalBlitPass",
                        "position": {"x": 1000, "y": 280}, "properties": {}}],
             "edges": [{"id": 1, "src": "GPUDriven.color", "dst": "FinalBlit.source"}], "outputs": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, indent=4) + "\n", encoding="utf-8")
    print(f"Prepared {args.output}: {roots} roots, {args.page_mib} MiB page pool")


if __name__ == "__main__":
    main()
