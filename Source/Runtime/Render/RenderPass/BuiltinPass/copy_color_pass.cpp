#include "Runtime/Render/RenderPass/BuiltinPass/builtin_passes.h"
#include "Runtime/Render/RenderPass/BuiltinPass/builtin_pass_common.h"

namespace metallic::render::builtin_pass {
namespace {

class CopyColorPass final : public UnsafePass {
public:
    RenderPassReflection reflect(const RenderGraphCompileContext&) const override
    {
        RenderPassReflection reflection;
        reflection.addTextureInput("source", "Source color texture")
            .transferRead();
        reflection.addTextureOutput("color", "Copied color texture")
            .transferWrite()
            .format = Format::Rgba8Unorm;
        return reflection;
    }

    Result execute(RenderGraphExecutionContext& context) override
    {
        TextureHandle source = context.inputTexture("source");
        TextureHandle color = context.outputTexture("color");
        if (!source.valid() || !color.valid()) {
            return makeError(Error::InvalidArgument);
        }

        context.commandBuffer().copyTexture(TextureCopyDesc{
            .source = source.texture(),
            .destination = color.texture(),
            .width = context.width(),
            .height = context.height(),
            .depth = 1,
            .sourceMipLevel = 0,
            .sourceBaseLayer = 0,
            .destinationMipLevel = 0,
            .destinationBaseLayer = 0,
        });
        return {};
    }
};

} // namespace

std::unique_ptr<RenderGraphPass> createCopyColorPass()
{
    return std::make_unique<CopyColorPass>();
}

} // namespace metallic::render::builtin_pass
