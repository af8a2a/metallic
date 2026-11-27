#!/usr/bin/env python3
"""Export the sample rock weight alongside a Houdini height EXR with hoiiotool."""

import argparse
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


def read_image_windows(tool, path):
    result = subprocess.run([tool, "--info:format=xml:verbose=1", str(path)],
                            check=True, stdout=subprocess.PIPE)
    spec = ET.fromstring(result.stdout)
    data = tuple(int(spec.findtext(key)) for key in ("width", "height", "x", "y"))
    display = tuple(int(spec.findtext(key)) for key in ("full_width", "full_height", "full_x", "full_y"))
    return data, display


def export_rock_weight(tool, height_path, output):
    data, display = read_image_windows(tool, height_path)
    width, height, x, y = data
    if width < 2 or height < 2:
        raise ValueError("the height EXR must contain at least 2×2 samples")
    full_width, full_height, full_x, full_y = display
    output.parent.mkdir(parents=True, exist_ok=True)
    size = f"{width}x{height}"

    # Each column follows the height samples, with terrain X spanning -128 to +128.
    # t goes from one on the rock plateau to zero through the grass transition.
    subprocess.run([
        tool, "--pattern", "fill:left=-128:right=128", size, "1",
        # Bend the north end east outside the primary seam so a reversed Z is visible.
        "--pattern", "fill:top=-128:bottom=128", size, "1",
        "--addc", "75", "--mulc", str(-8 / 53), "--clamp:min=0:max=8", "--addc", "56",
        "--sub", "--abs", "--mulc", "-1", "--addc", "5", "--mulc", "0.4", "--clamp:min=0:max=1",
        # Convert t to smoothstep t*t*(3-2*t).
        "--dup", "--dup", "--mul", "--swap", "--mulc", "-2", "--addc", "3", "--mul",
        "--chnames", "R", "--origin", f"{x:+d}{y:+d}",
        "--fullsize", f"{full_width}x{full_height}{full_x:+d}{full_y:+d}",
        "--nosoftwareattrib", "-d", "float", "-o", str(output),
    ], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("height", type=Path, help="Houdini height field OpenEXR")
    parser.add_argument("--output", type=Path, required=True, help="Output R-channel rock weight OpenEXR")
    parser.add_argument("--oiiotool", default="hoiiotool", help="Path to Houdini's hoiiotool executable")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".exr" or args.output.resolve() == args.height.resolve():
        parser.error("--output must be a separate OpenEXR file")
    try:
        export_rock_weight(args.oiiotool, args.height, args.output)
    except (ET.ParseError, OSError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Terrain weight export failed: {error}\n")


if __name__ == "__main__":
    main()
