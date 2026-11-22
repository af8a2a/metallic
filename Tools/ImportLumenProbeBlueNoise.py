"""Import UE 5.7.4's scalar and vector Lumen blue-noise textures from a local installation or PNG exports."""

import argparse
import ctypes
from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
import tempfile


ASSETS = Path("Content/EngineMaterials")
OODLE_DLL = Path("Binaries/DotNET/UnrealBuildTool/oo2core_9_win64.dll")
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "Asset/Lumen"
WIDTH, HEIGHT = 128, 8192


@dataclass(frozen=True)
class BlueNoiseTexture:
    package: str
    output: str
    package_sha256: str
    pixels_sha256: str


VEC2 = BlueNoiseTexture(
    "FastBlueNoise_vec2_128x128x64.uasset", "FastBlueNoiseVec2.rg8",
    "2ec202c29d53cf9955449bd2a26c7cc7be88c78e36c43c167ac6b20eff630169",
    "b9b16cc692b3cd12b7d6f19fcff14d9e7354ef801e6d4cc46d0bf2e2cf8cd33e")
SCALAR = BlueNoiseTexture(
    "FastBlueNoise_scalar_128x128x64.uasset", "FastBlueNoiseScalar.r8",
    "b7942616faf1b08633b3d3d8c7544f7aa789853351cf0b555f121dcaeeaec643",
    "b135dd62c73cdee1920885d28baf2b71643a0577934a77cda6d584c35162de0b")


def decode_oodle(package, header, exponent, blocks, raw_size, packed_size, oodle_dll):
    sizes = struct.unpack_from(f">{blocks}I", package, header + 64)
    cursor = header + 64 + 4 * blocks
    if packed_size != cursor - header + sum(sizes):
        raise ValueError("Invalid compressed texture block sizes.")

    if not oodle_dll.is_file():
        raise ValueError(f"UE's Win64 Oodle decoder was not found at {oodle_dll}. "
                         "Use --oodle with its location, or pass a PNG export.")
    oodle = ctypes.CDLL(str(oodle_dll.resolve()))
    decode = oodle.OodleLZ_Decompress
    ptr, size, integer = ctypes.c_void_p, ctypes.c_int64, ctypes.c_int
    decode.argtypes = [ptr, size, ptr, size, integer, integer, integer, ptr, size, ptr, ptr, ptr, size, integer]
    decode.restype = size
    data = bytearray()
    for packed in sizes:
        block = package[cursor:cursor + packed]
        cursor += packed
        expected = min(1 << exponent, raw_size - len(data))
        if packed == expected:
            data.extend(block)
        else:
            unpacked = ctypes.create_string_buffer(expected)
            if decode(block, packed, unpacked, expected, 1, 0, 0, None, 0, None, None, None, 0, 3) != expected:
                raise ValueError("Oodle could not decompress the texture source.")
            data.extend(unpacked.raw)

    return data


def import_package(source, texture, oodle_dll):
    package = source.read_bytes()
    if hashlib.sha256(package).hexdigest() != texture.package_sha256:
        raise ValueError("This package differs from the pinned UE 5.7.4 asset. "
                         "Export its source as a PNG to verify the pixels independently.")
    # The pinned packages contain a single FCompressedBuffer with a 64-byte
    # header in network byte order (UE Core/Private/Compression/CompressedBuffer.cpp).
    header = package.index(bytes.fromhex("b7756362"))
    method, _, _, exponent, blocks, raw_size, packed_size = struct.unpack_from(">BBBBIQQ", package, header + 8)
    if texture == SCALAR:
        if method != 0 or raw_size != WIDTH * HEIGHT or packed_size != 64 + raw_size:
            raise ValueError("Expected an uncompressed 128 x 8192 G8 texture source.")
        pixels = bytearray(package[header + 64:header + packed_size])
        row_size = WIDTH
    else:
        if method != 3 or raw_size != WIDTH * HEIGHT * 4:
            raise ValueError("Expected an Oodle-compressed 128 x 8192 BGRA8 texture source.")
        data = decode_oodle(package, header, exponent, blocks, raw_size, packed_size, oodle_dll)
        pixels = bytearray(WIDTH * HEIGHT * 2)
        pixels[0::2] = data[2::4]
        pixels[1::2] = data[1::4]
        row_size = WIDTH * 2
    # UE ImageCore/Private/ImageCoreDelta.cpp resets vertical byte deltas every 32768 pixels.
    rows_per_cut = 32768 // WIDTH
    for y in range(HEIGHT):
        if y % rows_per_cut:
            start = y * row_size
            for offset in range(start, start + row_size):
                pixels[offset] = (pixels[offset] + pixels[offset - row_size]) & 255
    return pixels


def import_png(source, texture):
    try:
        from PIL import Image
    except ImportError as error:
        raise ValueError("PNG input requires Pillow. UE installation input uses only Python's standard library.") from error
    with Image.open(source) as image:
        if image.format != "PNG" or image.size != (WIDTH, HEIGHT):
            raise ValueError("Expected a PNG of size 128 x 8192.")
        if texture == SCALAR and image.mode == "L":
            return image.tobytes()
        if image.mode not in ("RGB", "RGBA"):
            raise ValueError("Expected an 8-bit RGB or RGBA PNG, or an 8-bit grayscale PNG for scalar data.")
        if texture == SCALAR:
            return image.getchannel("R").tobytes()
        rg = bytearray(WIDTH * HEIGHT * 2)
        rg[0::2] = image.getchannel("R").tobytes()
        rg[1::2] = image.getchannel("G").tobytes()
        return rg


def import_texture(source, texture, oodle_dll):
    if source.suffix.lower() == ".png":
        pixels = import_png(source, texture)
    elif source.suffix.lower() == ".uasset":
        pixels = import_package(source, texture, oodle_dll)
    else:
        raise ValueError("Pass a UE 5.7.4 installation directory, a .uasset, or a PNG export.")
    if hashlib.sha256(pixels).hexdigest() != texture.pixels_sha256:
        raise ValueError(f"Decoded pixels differ from UE 5.7.4's {texture.package}.")
    output = OUTPUT_DIR / texture.output
    output_file = tempfile.NamedTemporaryFile(dir=output.parent, delete=False)
    temporary = Path(output_file.name)
    try:
        with output_file:
            output_file.write(pixels)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Imported {output} ({len(pixels)} bytes, SHA256 {texture.pixels_sha256})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="UE root, Engine directory, the original .uasset, or a source PNG export")
    parser.add_argument("--oodle", type=Path, help="UE Win64 Oodle decoder DLL for a standalone .uasset")
    parser.add_argument("--scalar", action="store_true", help="import only the scalar texture (also for renamed PNGs or packages)")
    args = parser.parse_args()
    try:
        source = args.source.resolve()
        if source.is_dir():
            engine = source / "Engine" if (source / "Engine").is_dir() else source
            for texture in ((SCALAR,) if args.scalar else (VEC2, SCALAR)):
                import_texture(engine / ASSETS / texture.package, texture, args.oodle or engine / OODLE_DLL)
        else:
            texture = SCALAR if args.scalar or source.stem == Path(SCALAR.package).stem else VEC2
            import_texture(source, texture, args.oodle or source.parent.parent.parent / OODLE_DLL)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Import failed: {error}\n")


if __name__ == "__main__":
    main()
