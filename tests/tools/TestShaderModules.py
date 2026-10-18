"""Regression tests for the Slang source ownership boundary."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('module_audit', Path(__file__).resolve().parents[2] / 'Tools/CheckShaderModules.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ShaderModuleOwnershipTests(unittest.TestCase):
    def run_case(self, files):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative, content in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding='utf-8')
            return audit.check(root)

    def test_owned_split_module_and_header_boundary(self):
        self.assertEqual([], self.run_case({
            'Shaders/Owner.slang': 'module Owner;\n#include "Vendor.hlsli"\n__include "Impl.slang";\n',
            'Shaders/Impl.slang': 'implementing Owner;\n// #include "Hidden.slang"\n',
            'tests/rhi/shaders/Probe.slang': 'module Probe;\nimport Owner;\n',
        }))

    def test_anonymous_source_and_bare_include_are_rejected(self):
        errors = self.run_case({
            'Shaders/Owner.slang': 'module Owner;\n#include "Impl.slang"\n',
            'Shaders/Impl.slang': 'float function() { return 1; }\n',
        })
        self.assertTrue(any('bare #include' in error for error in errors))
        self.assertTrue(any('expected exactly one' in error for error in errors))

    def test_cross_owner_and_orphan_implementation_are_rejected(self):
        errors = self.run_case({
            'Shaders/A.slang': 'module A;\n__include "Impl.slang";\n',
            'Shaders/B.slang': 'module B;\n',
            'Shaders/Impl.slang': 'implementing B;\n',
        })
        self.assertTrue(any('must declare implementing A' in error for error in errors))
        self.assertTrue(any('not included by its owner' in error for error in errors))

    def test_runtime_unit_requires_exact_generator_owner(self):
        files = {
            'Shaders/Modules/SceneSurface.slang': 'module SceneSurface;\n__include "MaterialValueDispatch.slang";\n',
            'Source/Runtime/Render/Material/MaterialValueProgram.cpp': 'implementing Other; namespace Metallic.SceneSurface',
        }
        self.assertTrue(any('generated MaterialValueDispatch' in error for error in self.run_case(files)))
        files['Source/Runtime/Render/Material/MaterialValueProgram.cpp'] = 'implementing SceneSurface; namespace Metallic.SceneSurface'
        self.assertEqual([], self.run_case(files))


    def test_generated_coverage_requires_named_module(self):
        files = {
            'Shaders/Modules/Material.slang': 'module Material;\n__include "Material/Coverage.slang";\n',
            'Shaders/Modules/Material/Coverage.slang': 'implementing Material;\nimport MaterialCoverageGenerated;\n',
            'Source/Runtime/Render/Material/MaterialValueProgram.cpp': 'module Other;',
        }
        self.assertTrue(any('generated Coverage' in error for error in self.run_case(files)))
        files['Source/Runtime/Render/Material/MaterialValueProgram.cpp'] = 'module MaterialCoverageGenerated;'
        self.assertEqual([], self.run_case(files))


if __name__ == '__main__':
    unittest.main()
