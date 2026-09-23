"""Reject asset loading and streaming scheduling in render pass implementations."""
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[1]
passes = root / "Source/Runtime/Render/RenderPass"
# Rendering may bind resident resources and compile shaders. Asset IO, scene
# acquisition and page/texture scheduling must go through StreamerSubsystem.
forbidden = re.compile(
    r"\b(?:SceneResourceManager|stbi_load|acquireStream|beginTextureStreaming|"
    r"uploadMaterialTextures|beginPrepareAsync|pumpPrepareAsync|cmdBeginFrame|"
    r"cmdPreTraversal|cmdPostTraversal|cmdEndFrame)\b"
)
failures = []
for path in sorted(passes.rglob("*")):
    if path.suffix not in {".cpp", ".h"}:
        continue
    for line, text in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if forbidden.search(text):
            failures.append(f"{path.relative_to(root)}:{line}: {text.strip()}")
if failures:
    print("Render pass streaming boundary violations:\n" + "\n".join(failures))
    sys.exit(1)
print("PASS: render passes contain no scene-loader or streaming-scheduler entry points")
