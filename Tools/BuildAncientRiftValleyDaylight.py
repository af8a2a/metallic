#!/usr/bin/env python3
"""Apply authored daylight to the existing Ancient Rift Valley scene document."""

import argparse
import json
from pathlib import Path

from BuildTerrainRegions import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--lighting", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scene_path = args.scene.resolve()
    output_path = args.output.resolve()
    if scene_path.parent != output_path.parent:
        parser.error("the daylight document must be beside the source document to preserve relative asset paths")
    if scene_path == output_path:
        parser.error("the daylight document must use a different filename from the source document")
    try:
        scene = json.loads(scene_path.read_text(encoding="utf-8"))
        lighting = json.loads(args.lighting.read_text(encoding="utf-8"))["world"]
        scene["world"] = {**scene["world"], **lighting}
        write_json(output_path, scene)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
