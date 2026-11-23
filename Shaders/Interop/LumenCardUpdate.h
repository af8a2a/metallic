#ifndef METALLIC_LUMEN_CARD_UPDATE_INTEROP
#define METALLIC_LUMEN_CARD_UPDATE_INTEROP

#ifdef __cplusplus
#include <cstdint>
namespace metallic::render {
#define METALLIC_CARD_UPDATE_FLAG inline constexpr uint32_t
#else
namespace Metallic.Lumen {
#define METALLIC_CARD_UPDATE_FLAG public static const uint
#endif

// One word per fixed Surface Cache atlas slot, published each frame.
METALLIC_CARD_UPDATE_FLAG kLumenCardCaptured = 1u;
METALLIC_CARD_UPDATE_FLAG kLumenCardRelit = 2u;
METALLIC_CARD_UPDATE_FLAG kLumenCardResetLighting = 4u;

#undef METALLIC_CARD_UPDATE_FLAG
}
#endif
