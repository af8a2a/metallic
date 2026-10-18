"""Check named ownership for repository Slang sources (no SDK source mutation)."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

# Keep literals intact so include paths and printf strings do not become comments.
COMMENTS = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*', re.S)
OWNER = re.compile(r'^\s*(module|implementing)\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*;', re.M)
BARE_INCLUDE = re.compile(r'^\s*#\s*include\s*["<]([^">]+\.slang)[">]', re.M | re.I)
MODULE_INCLUDE = re.compile(r'^\s*__include\s+"([^"]+)"\s*;', re.M)


def code_only(source: str) -> str:
    return COMMENTS.sub(lambda match: re.sub(r'[^\n]', ' ', match[0])
                        if match[0].startswith(('//', '/*')) else match[0], source)


def check(root: Path) -> list[str]:
    paths = sorted({path.resolve() for folder in (root / 'Shaders', root / 'tests')
                    for path in folder.rglob('*.slang')})
    errors: list[str] = []
    units: dict[Path, tuple[str, str, str]] = {}
    modules: dict[str, Path] = {}
    included: set[Path] = set()

    def label(path: Path) -> str:
        return path.relative_to(root).as_posix()

    for path in paths:
        code = code_only(path.read_text(encoding='utf-8-sig'))
        owners = OWNER.findall(code)
        if len(owners) != 1:
            errors.append(f'{label(path)}: expected exactly one module or implementing declaration, found {len(owners)}')
            continue
        kind, owner = owners[0]
        units[path] = kind, owner, code
        if label(path) == 'Shaders/Modules/Material/Coverage.slang' and re.search(
                r'\bimport\s+MaterialCoverageGenerated\s*;', code):
            generator = root / 'Source/Runtime/Render/Material/MaterialValueProgram.cpp'
            generated_source = generator.read_text(encoding='utf-8') if generator.exists() else ''
            if 'module MaterialCoverageGenerated;' not in generated_source:
                errors.append(f'{label(path)}: generated Coverage must declare module MaterialCoverageGenerated in its generator')
        if kind == 'module':
            if owner in modules:
                errors.append(f'{label(path)}: duplicate module {owner} also declared by {label(modules[owner])}')
            modules[owner] = path
        for match in BARE_INCLUDE.finditer(code):
            errors.append(f'{label(path)}: bare #include of Slang source {match[1]}; use import or an owned __include unit')

    for path, (kind, owner, code) in units.items():
        if kind == 'implementing' and owner not in modules:
            errors.append(f'{label(path)}: implementing owner {owner} has no module declaration')
        for match in MODULE_INCLUDE.finditer(code):
            # This one unit is emitted into a content-addressed runtime search
            # directory. Require its generator to retain the exact module owner;
            # other missing/generated headers do not bypass ownership checks.
            if label(path) == 'Shaders/Modules/SceneSurface.slang' and match[1] == 'MaterialValueDispatch.slang':
                generator = root / 'Source/Runtime/Render/Material/MaterialValueProgram.cpp'
                generated_source = generator.read_text(encoding='utf-8') if generator.exists() else ''
                if owner != 'SceneSurface' or 'implementing SceneSurface;' not in generated_source or 'namespace Metallic.SceneSurface' not in generated_source:
                    errors.append(f'{label(path)}: generated MaterialValueDispatch must implement SceneSurface in its generator')
                continue
            target = (path.parent / match[1]).resolve()
            target_unit = units.get(target)
            if target_unit is None:
                errors.append(f'{label(path)}: __include target {match[1]} is not an owned repository Slang unit')
            elif target_unit[:2] != ('implementing', owner):
                errors.append(f'{label(path)}: __include target {match[1]} must declare implementing {owner}')
            else:
                included.add(target)
    for path, (kind, owner, _) in units.items():
        if kind == 'implementing' and path not in included:
            errors.append(f'{label(path)}: implementing {owner} is not included by its owner')
    if not errors:
        print(f'Slang module ownership passed: {len(paths)} files, {len(modules)} modules, {len(included)} implementation units')
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = check(args.root.resolve())
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
