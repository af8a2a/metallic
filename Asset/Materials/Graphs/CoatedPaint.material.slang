// Metallic read-only Slang SDK expression profile v1.
// parameter(0) is a float4 authored in linear Rec.709.
import MaterialAuthoringSDK;
Material evaluate()
{
    let paint = openPBR(parameter(0), 0.0, 0.3);
    return withCoat(paint, 1.0, 0.08);
}
