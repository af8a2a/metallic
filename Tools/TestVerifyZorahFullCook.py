"""Small metadata and evidence regressions for VerifyZorahFullCook.py."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from VerifyZorahFullCook import geometry_key, source_coverage, verify, reference_size_comparison


class VerifyZorahFullCookTests(unittest.TestCase):
    def test_reference_cache_size_boundaries(self):
        for actual, accepted in ((94, False), (95, True), (100, True), (105, True), (106, False)):
            self.assertEqual(reference_size_comparison(actual, 100)["withinFivePercent"], accepted)
        self.assertFalse(reference_size_comparison(95, 101)["withinFivePercent"])
        self.assertTrue(reference_size_comparison(96, 101)["withinFivePercent"])
        self.assertTrue(reference_size_comparison(106, 101)["withinFivePercent"])
        self.assertFalse(reference_size_comparison(107, 101)["withinFivePercent"])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="MetallicFullCookVerify-")
        self.addCleanup(self.directory.cleanup)
        base = Path(self.directory.name)
        self.source = base / "source.gltf"
        self.manifest = base / "cook.json"
        self.asset = base / "cook.bin"
        self.asset.write_bytes(b"cook")
        primitive = {"attributes": {"POSITION": 0, "NORMAL": 2}, "indices": 1, "material": 0}
        material_variant = {**deepcopy(primitive), "material": 1}
        index_variant = {**deepcopy(primitive), "indices": 5}
        self.root = {
            "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0, 5]}],
            "accessors": [{"count": 4, "type": "VEC3"}, {"count": 6, "type": "SCALAR"},
                          {"count": 4, "type": "VEC3"}, {"count": 3, "type": "VEC3"},
                          {"count": 2, "type": "VEC3"}, {"count": 3, "type": "SCALAR"}],
            "meshes": [{"primitives": [primitive]}, {"primitives": [deepcopy(primitive)]},
                       {"primitives": [material_variant]}, {"primitives": [index_variant]}],
            "nodes": [
                {"mesh": 0, "children": [1], "extensions": {"EXT_mesh_gpu_instancing": {"attributes": {"TRANSLATION": 3}}}},
                {"mesh": 2, "children": [2], "extras": {"visible": False}},
                {"mesh": 1, "children": [3], "extensions": {"EXT_mesh_gpu_instancing": {"attributes": {"SCALE": 4}}}},
                {"mesh": 3}, {"mesh": 3},  # Node 4 is outside the selected scene.
                {"mesh": 0}],
        }
        self.report = {
            "status": "complete", "source": str(self.source), "asset": str(self.asset),
            "fileBytes": 4, "cookRevision": 3, "payloadValidation": "all-pages",
            "primitives": 3, "instances": 8, "groups": 3, "pages": 3,
            "levels": {"0": {"triangles": 5}}, "terminalPages": [0, 1, 2],
            "terminalPageCount": 3, "terminalInstanceGroups": 8, "terminalInstanceClusters": 8,
            "geometries": [
                {"primitive": 0, "sourcePrimitive": 0, "instances": 6, "groups": 1, "levels": 1, "terminalGroups": 1},
                {"primitive": 1, "sourcePrimitive": 2, "instances": 1, "groups": 1, "levels": 1, "terminalGroups": 1},
                {"primitive": 2, "sourcePrimitive": 3, "instances": 1, "groups": 1, "levels": 1, "terminalGroups": 1}],
            "rootCutAudit": {"rootPages": 3, "rootClusters": 3, "rootTriangles": 5, "rootBytes": 768,
                             "rootInstanceGroups": 8, "rootInstanceClusters": 8,
                             "primitivesByRootBytes": [
                {"primitive": 0, "sourcePrimitive": 0, "instances": 6, "material": 0, "lod0Triangles": 2,
                 "rootPages": 1, "rootClusters": 1, "rootTriangles": 2, "rootBytes": 256},
                {"primitive": 1, "sourcePrimitive": 2, "instances": 1, "material": 1, "lod0Triangles": 2,
                 "rootPages": 1, "rootClusters": 1, "rootTriangles": 2, "rootBytes": 256},
                {"primitive": 2, "sourcePrimitive": 3, "instances": 1, "material": 0, "lod0Triangles": 1,
                 "rootPages": 1, "rootClusters": 1, "rootTriangles": 1, "rootBytes": 256}]}}

    def run_verify(self):
        self.source.write_text(json.dumps(self.root), encoding="utf-8")
        self.manifest.write_text(json.dumps(self.report), encoding="utf-8")
        return verify(self.source, self.manifest, expected_geometries=3, expected_instances=8, cook_revision=3)

    def test_material_independent_geometry_keeps_instance_materials(self):
        coverage = source_coverage(self.root, share_materials=True)
        self.assertEqual(coverage["geometryCount"], 2)
        self.assertEqual(coverage["sourceTriangles"], 3)
        self.assertEqual(coverage["instanceCount"], 8)
        shared = coverage["geometryMappings"][0]
        self.assertEqual(shared["sourceAliases"], [0, 1, 2])
        self.assertEqual(shared["instanceMaterials"], {"0": 6, "1": 1})

    def test_hierarchical_instancing_dedup_and_noninstanced_children(self):
        result = self.run_verify()
        self.assertEqual(result["status"], "verified", result)
        self.assertEqual(result["sourcePrimitiveCount"], 4)
        self.assertEqual(result["sourceTriangles"], 5)
        self.assertEqual(result["instancedSourceTriangles"], 15)
        self.assertEqual(result["selectedSceneNodes"], 5)
        self.assertEqual(result["gpuInstancingNodeVisits"], 2)
        self.assertEqual([g["instances"] for g in result["geometryMappings"]], [6, 1, 1])
        self.assertEqual(result["geometryMappings"][0]["sourceAliases"], [0, 1])

    def test_geometry_identity_uses_all_five_fields_and_defaults(self):
        primitive = self.root["meshes"][0]["primitives"][0]
        equivalent = {**primitive, "mode": 4, "targets": []}
        self.assertEqual(geometry_key(primitive), geometry_key(equivalent))
        for field, value in (("mode", 5), ("indices", 5), ("material", 1),
                             ("attributes", {"POSITION": 0, "NORMAL": 2, "COLOR_0": 2}),
                             ("targets", [{"POSITION": 0}])):
            with self.subTest(field=field):
                self.assertNotEqual(geometry_key(primitive), geometry_key({**primitive, field: value}))

    def test_per_geometry_triangle_swap_fails_even_when_total_matches(self):
        rows = self.report["rootCutAudit"]["primitivesByRootBytes"]
        rows[0]["lod0Triangles"], rows[2]["lod0Triangles"] = 1, 2
        result = self.run_verify()
        self.assertEqual(result["status"], "failed")
        self.assertIn("lod0Triangles mismatch", " ".join(result["failures"]))

    def test_per_geometry_instance_swap_fails_even_when_total_matches(self):
        rows = self.report["geometries"]
        rows[0]["instances"], rows[1]["instances"] = 1, 6
        result = self.run_verify()
        self.assertEqual(result["status"], "failed")
        self.assertIn("instances mismatch", " ".join(result["failures"]))

    def test_duplicate_geometry_mapping_fails(self):
        self.report["geometries"][1]["primitive"] = 0
        self.assertEqual(self.run_verify()["status"], "failed")

    def test_missing_lod0_evidence_is_not_a_pass(self):
        del self.report["rootCutAudit"]["primitivesByRootBytes"][1]["lod0Triangles"]
        result = self.run_verify()
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("lod0Triangles", " ".join(result["missingEvidence"]))

    def test_old_audit_can_pass_metadata_but_not_complete_verification(self):
        self.report["source"] = ""
        self.report["payloadValidation"] = "not-requested"
        result = self.run_verify()
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["checks"]["geometryAndInstanceCoverage"], "passed")
        self.assertEqual(len(result["missingEvidence"]), 2)

    def test_checkpoint_and_wrong_file_size_fail(self):
        Path(str(self.asset) + ".partial").write_bytes(b"pending")
        self.assertEqual(self.run_verify()["checks"]["completedFile"], "failed")
        Path(str(self.asset) + ".partial").unlink()
        self.report["fileBytes"] = 5
        self.assertEqual(self.run_verify()["checks"]["completedFile"], "failed")

    def test_old_revision_and_empty_roots_fail(self):
        self.report["cookRevision"] = 2
        self.report["geometries"][1]["terminalGroups"] = 0
        result = self.run_verify()
        self.assertEqual(result["checks"]["cookRevision"], "failed")
        self.assertEqual(result["checks"]["geometryAndInstanceCoverage"], "failed")

    def test_cycle_and_mismatched_instance_attributes_are_rejected(self):
        self.root["nodes"][3]["children"] = [0]
        with self.assertRaisesRegex(ValueError, "Cycle"):
            source_coverage(self.root)
        del self.root["nodes"][3]["children"]
        self.root["nodes"][0]["extensions"]["EXT_mesh_gpu_instancing"]["attributes"]["SCALE"] = 4
        with self.assertRaisesRegex(ValueError, "inconsistent instance"):
            source_coverage(self.root)


if __name__ == "__main__":
    unittest.main()
