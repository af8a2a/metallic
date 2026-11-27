"""Import the UE 5.7.4 Windows cloud material's platform mip-zero texels."""

import argparse
import hashlib
from pathlib import Path
import struct
import tempfile


CLOUD_TEXTURES = (
    ("T_CloudPattern", "Pattern.rgba16f", (256, 256, 1),
     "adfe34abac1e5a93c9170b368919f09886b18ca9a4563fc27d6e5ca89fe3ef2f"),
    ("T_Profile_08", "Profile.rgba16f", (256, 256, 1),
     "f9b1085bb9c309f422a6642d5e28c098e3b37aa1a4bd8397b9a407864e475e5f"),
    ("VT_PerlinWorley_Balanced", "Noise.rgba8srgb", (128, 128, 128),
     "f390df7192eed9e39235f5850f3eb5e9f3b20cf7ed16d40a91a11478b5db9c9a"),
)
OUTPUT = Path(__file__).resolve().parents[1] / "Asset/Atmosphere/CloudShape"


def platform_pixels(source, name, extent):
    width, height, depth = extent
    volume = depth > 1
    size = width * height * depth * (4 if volume else 8)
    raw = source / (name + ".mip0")
    bgra = volume
    if raw.is_file():
        pixels = raw.read_bytes()
        if len(pixels) != size:
            raise ValueError(f"{raw} does not match the expected platform mip size")
    else:
        path = source / (name + ".dds")
        dds = path.read_bytes()
        if len(dds) < 148 or dds[:4] != b"DDS " or dds[84:88] != b"DX10":
            raise ValueError(f"{path} requires a DDS DX10 header")
        h, w, _, d = struct.unpack_from("<4I", dds, 12)
        dxgi, dimension, _, layers, _ = struct.unpack_from("<5I", dds, 128)
        if (w, h, max(d, 1)) != extent or dimension != (4 if volume else 3) or layers != 1:
            raise ValueError(f"{path} does not match the platform texture dimensions")
        if dxgi not in ((28, 29, 87, 91) if volume else (10,)):
            raise ValueError(f"{path} has unexpected DXGI format {dxgi}")
        bgra = dxgi in (87, 91)
        pixels = dds[148:148 + size]
    if bgra:
        rgba = bytearray(pixels)
        rgba[0::4], rgba[2::4] = pixels[2::4], pixels[0::4]
        return bytes(rgba)
    return pixels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="folder of platform .mip0 or DX10 .dds files")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    decoded = {}
    try:
        for asset, filename, extent, digest in CLOUD_TEXTURES:
            pixels = platform_pixels(args.source, asset, extent)
            if hashlib.sha256(pixels).hexdigest() != digest:
                raise ValueError(f"{asset} differs from the UE 5.7.4 Windows reference")
            decoded[filename] = pixels
        args.output.mkdir(parents=True, exist_ok=True)
        for filename, pixels in decoded.items():
            stream = tempfile.NamedTemporaryFile(dir=args.output, delete=False)
            staging = Path(stream.name)
            try:
                with stream:
                    stream.write(pixels)
                staging.replace(args.output / filename)
            finally:
                staging.unlink(missing_ok=True)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Cloud shape import failed: {error}\n")


if __name__ == "__main__":
    main()
