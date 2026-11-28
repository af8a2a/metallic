"""Regression checks for Houdini terrain layer assets and weight alignment."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    import Imath
    import numpy as np
    import OpenEXR
    from PIL import Image
except ModuleNotFoundError as error:
    if error.name not in {"Imath", "numpy", "OpenEXR", "PIL"}:
        raise
    print("Terrain layer tests require Tools/requirements-terrain.txt", file=sys.stderr)
    sys.exit(77)

TOOLS = Path(__file__).resolve().parents[2] / "Tools"
sys.path.insert(0, str(TOOLS))
from BuildTerrainLayerTextures import build_layer_material
from BuildTerrainMesh import TerrainBuildError
from BuildTerrainVSMSample import build_sample
from ImportTerrainLayers import import_terrain_layers
from TerrainLayerAsset import load_terrain_layer_asset, read_layer_weights


class TerrainLayerAssetTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "mountain.layers.json"
        self.weight_map = self.root / "weights.exr"
        textures = self.root / "textures"
        textures.mkdir()
        Image.new("RGB", (1, 1), (120, 150, 50)).save(textures / "color.png")
        Image.new("RGB", (1, 1), (128, 128, 255)).save(textures / "normal.png")
        Image.new("L", (1, 1), 168).save(textures / "roughness.png")
        texture_paths = {
            "baseColor": "textures/color.png",
            "normal": "textures/normal.png",
            "roughness": "textures/roughness.png",
        }
        self.document = {
            "type": "Metallic.TerrainLayers",
            "version": 1,
            "weightMap": "weights.exr",
            "defaultLayer": "bedrock",
            "layers": [
                {"name": "grass", "channel": "B", "worldTileSize": [3, 5], **texture_paths},
                {"name": "bedrock", "channel": "A", "worldTileSize": 8, **texture_paths},
                {"name": "sand", "channel": "R", "worldTileSize": 6, **texture_paths},
            ],
        }
        self.terrain = {"sourceDataWindow": [-3, 7, -2, 8], "flipZ": False}
        self.planes = {
            "R": [0.125, 0.25, 0, 0],
            "G": [1, 1, 1, 1],
            "B": [0.375, 0.5, 0, 0.5],
            "A": [0.5, 0.25, 0, 0],
        }
        self.write_exr(self.planes)

    def write_exr(self, planes, data_window=None, path=None):
        min_x, min_y, max_x, max_y = data_window or self.terrain["sourceDataWindow"]
        header = OpenEXR.Header(max_x - min_x + 1, max_y - min_y + 1)
        header["dataWindow"] = Imath.Box2i(Imath.V2i(min_x, min_y), Imath.V2i(max_x, max_y))
        header["displayWindow"] = header["dataWindow"]
        header["channels"] = {
            name: Imath.Channel(Imath.PixelType(Imath.PixelType.HALF if name == "A" else Imath.PixelType.FLOAT))
            for name in planes
        }
        image = OpenEXR.OutputFile(str(path or self.weight_map), header)
        try:
            image.writePixels({
                name: np.array(values, dtype=np.float16 if name == "A" else np.float32).tobytes()
                for name, values in planes.items()
            })
        finally:
            image.close()

    def load(self, document=None):
        self.path.write_text(json.dumps(document if document is not None else self.document), encoding="utf-8")
        return load_terrain_layer_asset(self.path)

    def test_scale_paths_and_stable_layer_channel_mapping(self):
        asset = self.load()
        self.assertEqual(self.weight_map.resolve(), asset.weight_map)
        self.assertEqual("bedrock", asset.default_layer)
        self.assertEqual(("grass", "bedrock", "sand"), tuple(layer.name for layer in asset.layers))
        self.assertEqual(("B", "A", "R"), tuple(layer.channel for layer in asset.layers))
        self.assertEqual(((3.0, 5.0), (8.0, 8.0), (6.0, 6.0)), tuple(layer.world_tile_size for layer in asset.layers))
        self.assertEqual(self.root.resolve() / "textures" / "color.png", asset.layers[0].base_color)
        reordered_document = deepcopy(self.document)
        reordered_document["layers"][0]["name"] = "草地"
        reordered_document["layers"].reverse()
        reordered_asset = self.load(reordered_document)
        self.assertEqual(("sand", "bedrock", "草地"), tuple(layer.name for layer in reordered_asset.layers))
        np.testing.assert_array_equal(read_layer_weights(reordered_asset, self.terrain),
                                      read_layer_weights(asset, self.terrain))

    def test_linear_weights_use_alpha_default_and_follow_height_flip_z(self):
        asset = self.load()
        expected = np.array([
            [[0.125, 0, 0.375, 0.5], [0.25, 0, 0.5, 0.25]],
            [[0, 0, 0, 1], [0, 0, 1, 0]],
        ], dtype=np.float32)
        weights = read_layer_weights(asset, self.terrain)
        self.assertEqual(np.float32, weights.dtype)
        np.testing.assert_array_equal(weights, expected)
        self.terrain["flipZ"] = True
        np.testing.assert_array_equal(read_layer_weights(asset, self.terrain), expected[::-1])

    def test_equal_size_exr_with_shifted_data_window_is_rejected(self):
        self.write_exr(self.planes, [-2, 7, -1, 8])
        with self.assertRaisesRegex(TerrainBuildError, "data windows must match"):
            read_layer_weights(self.load(), self.terrain)

    def test_missing_channel_and_out_of_range_weights_are_rejected(self):
        asset = self.load()
        for replacement, message in (
            ({"a": self.planes["A"]}, "missing channel 'A'"),
            ({"A": [1.5, 0, 0, 0]}, "finite and in"),
            ({"A": [float("nan"), 0, 0, 0]}, "finite and in"),
        ):
            with self.subTest(message=message, replacement=replacement):
                planes = dict(self.planes)
                del planes["A"]
                planes.update(replacement)
                self.write_exr(planes)
                with self.assertRaisesRegex(TerrainBuildError, message):
                    read_layer_weights(asset, self.terrain)

    def test_invalid_layer_metadata_is_rejected(self):
        for fields, message in (
            ({"channel": "A"}, "repeats a layer name or RGBA channel"),
            ({"name": "BEDROCK"}, "repeats a layer name or RGBA channel"),
            ({"worldTileSize": [1, 0]}, "positive finite number or a pair"),
            ({"worldTileSize": 10**400}, "positive finite number or a pair"),
            ({"baseColor": "D:/asset/sand.png"}, "must be relative"),
        ):
            with self.subTest(fields=fields):
                document = deepcopy(self.document)
                document["layers"][0].update(fields)
                with self.assertRaisesRegex(TerrainBuildError, message):
                    self.load(document)
        document = {**self.document, "defaultLayer": "Bedrock"}
        with self.assertRaisesRegex(TerrainBuildError, "must exactly name a declared layer"):
            self.load(document)

    def test_import_baked_terrain_writes_linear_weight_texture_and_world_transforms(self):
        height_path = self.root / "height.exr"
        terrain_path = self.root / "terrain.gltf"
        output = self.root / "baked" / "material" / "terrain.layers.json"
        self.write_exr({"Y": [0, 1, 2, 3]}, path=height_path)
        self.load()
        bake = subprocess.run([
            sys.executable, "-B", str(TOOLS / "BuildTerrainMesh.py"), str(height_path),
            "--output", str(terrain_path), "--horizontal-size", "30", "--flip-z",
        ], capture_output=True, text=True)
        self.assertEqual(0, bake.returncode, bake.stdout + bake.stderr)

        import_terrain_layers(terrain_path, self.path, output)
        manifest_bytes = output.read_bytes()
        manifest = json.loads(manifest_bytes)
        self.assertEqual("Metallic.ImportedTerrainLayers", manifest["type"])
        self.assertEqual(self.terrain["sourceDataWindow"], manifest["terrain"]["sourceDataWindow"])
        self.assertEqual([2, 2], manifest["terrain"]["sourceResolution"])
        self.assertTrue(manifest["terrain"]["flipZ"])
        self.assertEqual("bedrock", manifest["defaultLayer"])
        layer = manifest["layers"][0]
        self.assertEqual("grass", layer["name"])
        self.assertEqual([10, 0, -5, 0, 6, -3], layer["uvTransform"])
        for role, name, color_space in (
            ("baseColor", "color.png", "srgb"),
            ("normal", "normal.png", "linear"),
            ("roughness", "roughness.png", "linear"),
        ):
            resource_path = Path(layer[role]["path"])
            self.assertFalse(resource_path.is_absolute())
            self.assertEqual((self.root / "textures" / name).resolve(), (output.parent / resource_path).resolve())
            self.assertEqual(color_space, layer[role]["colorSpace"])

        weight_map = manifest["weightMap"]
        weight_path = output.parent / weight_map["path"]
        self.assertEqual("linear", weight_map["colorSpace"])
        self.assertEqual("RGBA8", weight_map["format"])
        with Image.open(weight_path) as image:
            self.assertEqual("RGBA", image.mode)
            self.assertEqual((4, 4), image.size)
            self.assertFalse({"srgb", "gamma", "icc_profile", "chromaticity"} & image.info.keys())
            rgba = np.asarray(image)
        np.testing.assert_array_equal(rgba.sum(axis=-1), np.full((4, 4), 255))
        sx, _, ox, _, sz, oz = weight_map["uvTransform"]
        for edge, expected in ((0, [0, 0, 0, 1]), (1, [0.25, 0, 0.5, 0.25])):
            sample_x = (sx * edge + ox) * rgba.shape[1] - 0.5
            sample_z = (sz * edge + oz) * rgba.shape[0] - 0.5
            self.assertEqual((edge, edge), (sample_x, sample_z))
            np.testing.assert_allclose(rgba[int(sample_z), int(sample_x)] / 255, expected, atol=1 / 255, rtol=0)

        png_bytes = weight_path.read_bytes()
        import_terrain_layers(terrain_path, self.path, output)
        self.assertEqual(manifest_bytes, output.read_bytes())
        self.assertEqual(png_bytes, weight_path.read_bytes())

    def test_legacy_weight_map_preserves_the_rock_and_grass_entry(self):
        self.write_exr({"R": [0, 0.25, 0.5, 1]})
        terrain = {**self.terrain, "horizontalSize": 30, "flipZ": True}
        _, metadata = build_layer_material(self.root / "legacy.gltf", self.weight_map, terrain)
        self.assertEqual(("R", "1-R"), (metadata["rockWeight"], metadata["grassWeight"]))
        with Image.open(self.root / "legacy_weight.png") as image:
            self.assertEqual("L", image.mode)
            np.testing.assert_array_equal(np.asarray(image), [
                [128, 255, 255, 128], [0, 64, 64, 0], [0, 64, 64, 0], [128, 255, 255, 128],
            ])
        material = json.loads((self.root / "legacy.material").read_text(encoding="utf-8"))
        weight_map = material["resources"]["occlusionTexture"]
        self.assertEqual("asset://legacy_weight.png", weight_map["uri"])
        self.assertEqual([0.25, 0, 0.125, 0, 0.25, 0.125], weight_map["transform"])

    def test_sample_binds_four_layers_with_private_linear_and_color_resources(self):
        height_path = self.root / "height.exr"
        terrain_path = self.root / "terrain.gltf"
        output = self.root / "baked" / "terrain.gltf"
        self.write_exr({"Y": [0, 1, 2, 3]}, path=height_path)
        self.planes["G"] = [1, 1, 0, 1]
        self.write_exr(self.planes)
        template = self.document["layers"][0]
        self.document["layers"].append({**template, "name": "clay", "channel": "G"})
        self.load()
        bake = subprocess.run([
            sys.executable, "-B", str(TOOLS / "BuildTerrainMesh.py"), str(height_path),
            "--output", str(terrain_path), "--horizontal-size", "30", "--flip-z",
        ], capture_output=True, text=True)
        self.assertEqual(0, bake.returncode, bake.stdout + bake.stderr)
        lighting = self.root / "lighting.json"
        lighting.write_text(json.dumps({"world": {"environment": {"path": "environment.hdr"}}}), encoding="utf-8")
        build_sample(terrain_path, lighting,
                     TOOLS.parent / "Pipelines/Samples/gpu_driven_terrain_vsm.metallic_graph.json", output,
                     layer_asset=self.path)

        sidecar = json.loads(output.with_suffix(".metallic_scene.json").read_text(encoding="utf-8"))
        binding, = sidecar["materials"]
        self.assertEqual(0, binding["materialIndex"])
        self.assertEqual({"uri": "asset://terrain.material", "root": "."}, binding["materialAsset"])
        definition = json.loads(output.with_suffix(".materialdef").read_text(encoding="utf-8"))
        self.assertEqual("OpenPBR.TerrainLayers", definition["implementation"])
        material = json.loads(output.with_suffix(".material").read_text(encoding="utf-8"))
        self.assertNotIn("resources", material)
        terrain_layers = material["terrainLayers"]
        self.assertEqual("bedrock", terrain_layers["defaultLayer"])
        self.assertEqual(["B", "A", "R", "G"], [layer["channel"] for layer in terrain_layers["layers"]])
        weights = terrain_layers["weights"]
        self.assertEqual({"uri": "asset://terrain_layer_weights.png", "transform": [0.25, 0, 0.125, 0, 0.25, 0.125]},
                         weights)
        with Image.open(output.parent / weights["uri"].removeprefix("asset://")) as image:
            self.assertEqual("RGBA", image.mode)
            rgba = np.asarray(image)
            np.testing.assert_array_equal(rgba.sum(axis=-1), np.full((4, 4), 255))
            np.testing.assert_array_equal(rgba[0, 0], [0, 0, 0, 255])
        first = terrain_layers["layers"][0]
        for role, texture in (("baseColor", "color.png"), ("normal", "normal.png"), ("roughness", "roughness.png")):
            resource = first[role]
            self.assertEqual([10, 0, -5, 0, 6, -3], resource["transform"])
            self.assertEqual("srgb" if role == "baseColor" else None, resource.get("colorSpace"))
            texture_uri = resource["uri"].removeprefix("asset://")
            self.assertEqual(Path(texture_uri).name, texture_uri)
            self.assertEqual((self.root / "textures" / texture).read_bytes(), (output.parent / texture_uri).read_bytes())


if __name__ == "__main__":
    unittest.main()
