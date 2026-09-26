#pragma once

#include "RhiTest.h"
#include "imgui.h"

#include <algorithm>
#include <cmath>

namespace metallic::tests {

// Offscreen UI evidence from real ImGui geometry. These diagnostic panels use
// only their font atlas and solid primitives, so no native window is needed.
inline bool saveImGuiTestDrawDataPng(const ImDrawData& data, const unsigned char* atlas,
    int atlasWidth, int atlasHeight, int width, int height, const std::filesystem::path& path,
    std::string& message)
{
    std::vector<uint8_t> pixels(size_t(width) * height * 4, 24);
    for (size_t i = 3; i < pixels.size(); i += 4) { pixels[i] = 255; }
    const auto edge = [](ImVec2 a, ImVec2 b, ImVec2 p) { return (b.x-a.x)*(p.y-a.y)-(b.y-a.y)*(p.x-a.x); };
    for (const auto* list : data.CmdLists) {
        for (const auto& command : list->CmdBuffer) {
            if (command.UserCallback || command.GetTexID() != ImTextureID(1)) { continue; }
            for (unsigned int i = 0; i + 2 < command.ElemCount; i += 3) {
                const auto& a = list->VtxBuffer[list->IdxBuffer[command.IdxOffset + i] + command.VtxOffset];
                const auto& b = list->VtxBuffer[list->IdxBuffer[command.IdxOffset + i + 1] + command.VtxOffset];
                const auto& c = list->VtxBuffer[list->IdxBuffer[command.IdxOffset + i + 2] + command.VtxOffset];
                const float area = edge(a.pos, b.pos, c.pos); if (std::abs(area) < 1e-6f) { continue; }
                const int x0 = std::max({0, int(std::floor(std::min({a.pos.x,b.pos.x,c.pos.x}))), int(std::ceil(command.ClipRect.x))});
                const int y0 = std::max({0, int(std::floor(std::min({a.pos.y,b.pos.y,c.pos.y}))), int(std::ceil(command.ClipRect.y))});
                const int x1 = std::min({width, int(std::ceil(std::max({a.pos.x,b.pos.x,c.pos.x}))), int(command.ClipRect.z)});
                const int y1 = std::min({height, int(std::ceil(std::max({a.pos.y,b.pos.y,c.pos.y}))), int(command.ClipRect.w)});
                for (int y = y0; y < y1; ++y) { for (int x = x0; x < x1; ++x) {
                    const ImVec2 p(float(x)+.5f,float(y)+.5f);
                    const float wa = edge(b.pos,c.pos,p)/area, wb = edge(c.pos,a.pos,p)/area, wc = 1-wa-wb;
                    if (wa < 0 || wb < 0 || wc < 0) { continue; }
                    const int tx = std::clamp(int((wa*a.uv.x+wb*b.uv.x+wc*c.uv.x)*atlasWidth),0,atlasWidth-1);
                    const int ty = std::clamp(int((wa*a.uv.y+wb*b.uv.y+wc*c.uv.y)*atlasHeight),0,atlasHeight-1);
                    const auto* texel = atlas + (size_t(ty)*atlasWidth+tx)*4;
                    const auto channel = [&](int shift) { return wa*((a.col>>shift)&255)+wb*((b.col>>shift)&255)+wc*((c.col>>shift)&255); };
                    const float alpha = channel(IM_COL32_A_SHIFT)*texel[3]/(255.f*255.f);
                    auto* dest = pixels.data()+(size_t(y)*width+x)*4;
                    constexpr int shifts[]{IM_COL32_R_SHIFT,IM_COL32_G_SHIFT,IM_COL32_B_SHIFT};
                    for (int ch=0;ch<3;++ch) { dest[ch]=uint8_t(std::clamp(channel(shifts[ch])*texel[ch]/255.f*alpha+dest[ch]*(1-alpha),0.f,255.f)); }
                } }
            }
        }
    }
    return saveRgba8Png(path, pixels.data(), width, height, message);
}

} // namespace metallic::tests
