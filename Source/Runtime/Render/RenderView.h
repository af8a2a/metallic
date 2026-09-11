#pragma once

#include <array>
#include <cstdint>
#include "json.hpp"

namespace metallic::render {

struct ViewCamera {
    std::array<float, 3> eye{0.0f, 0.2f, 2.5f};
    std::array<float, 3> center{};
    std::array<float, 3> up{0.0f, 1.0f, 0.0f};
    float fovDegrees = 50.0f;
    float nearPlane = 0.05f;
    float farPlane = 1000.0f;
    float orthoHeight = 2.0f;
    bool orthographic = false;
    bool reversedZ = true;
    bool operator==(const ViewCamera&) const = default;
};

// GPU ABI shared with Libraries/Camera/ViewConstants.slang. Matrices can be
// derived from this unjittered camera without making the view depend on a pass.
struct alignas(16) ViewCameraConstants {
    float eye[4]{};
    float center[4]{};
    float upProjection[4]{};
    float viewport[4]{}; // aspect, render width, render height, vertical FOV radians
    float clipOrtho[4]{}; // near, far, orthographic height, reversed Z
};

struct alignas(16) ViewConstants {
    ViewCameraConstants current;
    ViewCameraConstants previous;
    float jitter[4]{}; // current xy, previous xy, in render pixels
    uint32_t frame[4]{}; // frame index, previous valid, temporal jitter enabled, camera cut serial
    float outputSize[4]{}; // display width, height, reciprocal width, reciprocal height
};
static_assert(sizeof(ViewCameraConstants) == 80 && sizeof(ViewConstants) == 208);

// Adapter for existing raster/shading parameter blocks with this camera layout.
template<class T>
void applyViewCamera(const ViewCameraConstants& view, T& target)
{
    for (uint32_t i = 0; i < 4; ++i) {
        target.eye[i] = view.eye[i];
        target.center[i] = view.center[i];
        target.upProjection[i] = view.upProjection[i];
        target.viewport[i] = view.viewport[i];
        target.clipOrtho[i] = view.clipOrtho[i];
    }
}

class RenderView {
public:
    const ViewCamera& camera() const { return camera_; }
    bool setCamera(const ViewCamera& camera);
    bool setCameraProperties(const nlohmann::json& properties);
    nlohmann::json cameraProperties() const;
    void setTemporalJitter(bool enabled) { temporalJitter_ = enabled; }
    bool temporalJitter() const { return temporalJitter_; }
    void cameraCut() { ++cutSerial_; }
    uint32_t cutSerial() const { return cutSerial_; }
    uint64_t revision() const { return revision_; }
    ViewConstants constants(uint64_t frameIndex, uint32_t renderWidth, uint32_t renderHeight,
        uint32_t outputWidth, uint32_t outputHeight, const ViewConstants* previous = nullptr) const;

private:
    ViewCamera camera_;
    bool temporalJitter_ = false;
    uint32_t cutSerial_ = 1;
    uint64_t revision_ = 1;
};

} // namespace metallic::render
