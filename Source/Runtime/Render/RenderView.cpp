#include "Runtime/Render/RenderView.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace metallic::render {

bool RenderView::setCamera(const ViewCamera& camera)
{
    const auto finite = [](const auto& values) {
        return std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); });
    };
    float distanceSquared = 0.0f, upLengthSquared = 0.0f, directionUp = 0.0f;
    for (size_t i = 0; i < 3; ++i) {
        distanceSquared += (camera.eye[i] - camera.center[i]) * (camera.eye[i] - camera.center[i]);
        upLengthSquared += camera.up[i] * camera.up[i];
        directionUp += (camera.center[i] - camera.eye[i]) * camera.up[i];
    }
    if (!finite(camera.eye) || !finite(camera.center) || !finite(camera.up) ||
        distanceSquared < 1e-12f || upLengthSquared < 1e-12f ||
        directionUp * directionUp >= distanceSquared * upLengthSquared * 0.999999f ||
        !std::isfinite(camera.fovDegrees) || camera.fovDegrees < 1.0f || camera.fovDegrees > 179.0f ||
        !std::isfinite(camera.nearPlane) || !std::isfinite(camera.farPlane) ||
        camera.nearPlane <= 0.0f || camera.farPlane <= camera.nearPlane ||
        !std::isfinite(camera.orthoHeight) || camera.orthoHeight <= 0.0f) { return false; }
    if (camera_ != camera) {
        if (camera_.orthographic != camera.orthographic || camera_.reversedZ != camera.reversedZ ||
            camera_.nearPlane != camera.nearPlane || camera_.farPlane != camera.farPlane) { cameraCut(); }
        camera_ = camera;
        ++revision_;
    }
    return true;
}

bool RenderView::setCameraProperties(const nlohmann::json& properties)
{
    if (!properties.is_object()) { return false; }
    try {
        ViewCamera camera = camera_;
        camera.eye = properties.value("eye", camera.eye);
        camera.center = properties.value("center", camera.center);
        camera.up = properties.value("up", camera.up);
        camera.fovDegrees = properties.value("fovDegrees", camera.fovDegrees);
        camera.nearPlane = properties.value("znear", camera.nearPlane);
        camera.farPlane = properties.value("zfar", camera.farPlane);
        camera.orthoHeight = properties.value("orthoHeight", camera.orthoHeight);
        const auto projection = properties.value("projection", std::string(camera.orthographic ? "orthographic" : "perspective"));
        if (projection != "perspective" && projection != "orthographic") { return false; }
        camera.orthographic = projection == "orthographic";
        camera.reversedZ = properties.value("reversedZ", camera.reversedZ);
        return setCamera(camera);
    } catch (const nlohmann::json::exception&) { return false; }
}

nlohmann::json RenderView::cameraProperties() const
{
    return {{"eye", camera_.eye}, {"center", camera_.center}, {"up", camera_.up},
        {"fovDegrees", camera_.fovDegrees}, {"znear", camera_.nearPlane}, {"zfar", camera_.farPlane},
        {"orthoHeight", camera_.orthoHeight}, {"projection", camera_.orthographic ? "orthographic" : "perspective"},
        {"reversedZ", camera_.reversedZ}};
}

ViewConstants RenderView::constants(uint64_t frameIndex, uint32_t renderWidth, uint32_t renderHeight,
    uint32_t outputWidth, uint32_t outputHeight, const ViewConstants* previous) const
{
    ViewConstants view;
    std::copy(camera_.eye.begin(), camera_.eye.end(), view.current.eye);
    std::copy(camera_.center.begin(), camera_.center.end(), view.current.center);
    std::copy(camera_.up.begin(), camera_.up.end(), view.current.upProjection);
    view.current.upProjection[3] = camera_.orthographic ? 1.0f : 0.0f;
    view.current.viewport[0] = float(std::max(renderWidth, 1u)) / float(std::max(renderHeight, 1u));
    view.current.viewport[1] = float(renderWidth);
    view.current.viewport[2] = float(renderHeight);
    view.current.viewport[3] = camera_.fovDegrees * 0.017453292519943295f;
    view.current.clipOrtho[0] = camera_.nearPlane;
    view.current.clipOrtho[1] = camera_.farPlane;
    view.current.clipOrtho[2] = camera_.orthoHeight;
    view.current.clipOrtho[3] = camera_.reversedZ ? 1.0f : 0.0f;
    view.frame[0] = static_cast<uint32_t>(frameIndex);
    view.frame[2] = temporalJitter_ ? 1u : 0u;
    view.frame[3] = cutSerial_;
    view.outputSize[0] = float(outputWidth);
    view.outputSize[1] = float(outputHeight);
    view.outputSize[2] = 1.0f / float(std::max(outputWidth, 1u));
    view.outputSize[3] = 1.0f / float(std::max(outputHeight, 1u));
    if (temporalJitter_) {
        const auto radicalInverse = [](uint64_t index, uint32_t base) {
            float value = 0.0f, scale = 1.0f / float(base);
            while (index != 0) { value += float(index % base) * scale; index /= base; scale /= float(base); }
            return value;
        };
        view.jitter[0] = radicalInverse(frameIndex % 1024u + 1u, 2) - 0.5f;
        view.jitter[1] = radicalInverse(frameIndex % 1024u + 1u, 3) - 0.5f;
    }
    view.frame[1] = previous != nullptr && previous->frame[0] + 1 == view.frame[0] &&
        previous->frame[2] == view.frame[2] && previous->frame[3] == cutSerial_ &&
        previous->current.viewport[1] == view.current.viewport[1] && previous->current.viewport[2] == view.current.viewport[2] &&
        previous->outputSize[0] == view.outputSize[0] && previous->outputSize[1] == view.outputSize[1];
    view.previous = view.frame[1] ? previous->current : view.current;
    view.jitter[2] = view.frame[1] ? previous->jitter[0] : view.jitter[0];
    view.jitter[3] = view.frame[1] ? previous->jitter[1] : view.jitter[1];
    return view;
}

} // namespace metallic::render
