#language slang 2026
module PathDiagnostics;
import Core;
namespace Metallic.PathDiagnostics
{
#if METALLIC_PT_PATH_DIAGNOSTICS
public uint row(uint2 pixel, uint sampleIndex)
{
    const uint2 targets[8] = {
        uint2(METALLIC_PT_DIAGNOSTIC_X0, METALLIC_PT_DIAGNOSTIC_Y0),
        uint2(METALLIC_PT_DIAGNOSTIC_X1, METALLIC_PT_DIAGNOSTIC_Y1),
        uint2(METALLIC_PT_DIAGNOSTIC_X2, METALLIC_PT_DIAGNOSTIC_Y2),
        uint2(METALLIC_PT_DIAGNOSTIC_X3, METALLIC_PT_DIAGNOSTIC_Y3),
        uint2(METALLIC_PT_DIAGNOSTIC_X4, METALLIC_PT_DIAGNOSTIC_Y4),
        uint2(METALLIC_PT_DIAGNOSTIC_X5, METALLIC_PT_DIAGNOSTIC_Y5),
        uint2(METALLIC_PT_DIAGNOSTIC_X6, METALLIC_PT_DIAGNOSTIC_Y6),
        uint2(METALLIC_PT_DIAGNOSTIC_X7, METALLIC_PT_DIAGNOSTIC_Y7)};
    for (uint i = 0u; i < METALLIC_PT_DIAGNOSTIC_COUNT; ++i) {
        if (all(pixel == targets[i])) return i * 16u + sampleIndex;
    }
    return 0xffffffffu;
}
public void write(uint rowIndex, uint column, float4 value)
{
    if (rowIndex == 0xffffffffu) return;
    resolveUniform(getResourceParameters<ScenePathDiagnosticResourceParameters>().pathDiagnostics)[uint2(column, rowIndex)] = value;
}
public void clearPixel(uint2 pixel)
{
    uint firstRow = row(pixel, 0u);
    if (firstRow == 0xffffffffu) return;
    // Unique validated targets own disjoint rows. Clear unused samples, vertices
    // and lighting slots in the same invocation that subsequently records them.
    for (uint sampleIndex = 0u; sampleIndex < 16u; ++sampleIndex) {
        for (uint column = 0u; column < 104u; ++column) {
            write(firstRow + sampleIndex, column, float4(0.0));
        }
    }
}
public void vertex(uint rowIndex, uint depth, uint word, float4 value)
{
    if (depth < 3u) write(rowIndex, 8u + depth * 32u + word, value);
}
#endif
}
