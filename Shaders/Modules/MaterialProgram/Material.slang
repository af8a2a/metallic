#language slang 2026
implementing MaterialProgram;

namespace Metallic.Material
{
// A concrete shader program owns its own model inputs and resource provider.
// This value wrapper adds no shared parameter union or runtime model dispatch.
// Domain modules supply evaluate() only for programs with their own contract.
public struct Material<TProgram>
{
    public TProgram program;
};

public extension<TProgram : ISurfaceMaterialProgram> Material<TProgram>
{
    // Mutations, such as stochastic texture RNG updates, remain on this instance.
    [mutating]
    public SurfaceMaterialResult<TProgram.Closure> evaluate(
        SurfaceMaterialContext context, MaterialInstanceRef instance)
    {
        return program.evaluate(context, instance);
    }
};
} // namespace Metallic.Material
