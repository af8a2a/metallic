#pragma once

#include "Runtime/Render/Streamer/UploadStreamer.h"
#include "Runtime/Render/Streamer/MeshletStreamLatency.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace metallic::render {

struct MeshletStreamPrefetchCamera {
    std::array<float, 3> eye{0, 0, 0};
    std::array<float, 3> center{0, 0, -1};
    std::array<float, 3> up{0, 1, 0};
    float fovDegrees = 60;
    float orthoHeight = 10;
    bool orthographic = false;
};

struct MeshletStreamPrefetchConfig {
    double fallbackHorizonSeconds = 0.1;
    double minHorizonSeconds = 0.025;
    double maxHorizonSeconds = 0.25;
    // Distances are in scene world units; callers should adapt these to scene scale.
    double maxTranslationDistance = 2;
    double teleportDistance = 10;
    double maxRotationDegrees = 20;
    double cameraCutRotationDegrees = 60;
    double maxDeltaSeconds = 0.25;
};

struct MeshletStreamPrefetchResult {
    MeshletStreamPrefetchCamera camera;
    double horizonSeconds = 0;
    double translationDistance = 0;
    double rotationDegrees = 0;
    double linearSpeed = 0;
    double angularSpeedDegrees = 0;
    bool active = false;
    bool historyReset = false;
    bool measuredLatency = false;
};

// Demand and visibility always use the real camera. This forecast is only an
// additional, budget-limited page request source, never a replacement render view.
class MeshletStreamPrefetchPredictor {
public:
    void reset()
    {
        valid_ = false;
    }

    MeshletStreamPrefetchResult update(const MeshletStreamPrefetchCamera& camera,
        double deltaSeconds, const MeshletStreamLatencySummary& demandToDrawableMilliseconds,
        bool cameraCut = false, const MeshletStreamPrefetchConfig& config = {})
    {
        MeshletStreamPrefetchResult result{.camera = camera};
        Basis current;
        if (!makeBasis(camera, current)) {
            reset();
            result.historyReset = true;
            return result;
        }
        const auto rebase = [&] {
            previous_ = camera;
            previousBasis_ = current;
            valid_ = true;
            result.historyReset = true;
        };
        const double maxDelta = positive(config.maxDeltaSeconds, 0.25);
        if (!valid_ || cameraCut || !std::isfinite(deltaSeconds) ||
            deltaSeconds < 0.0001 || deltaSeconds > maxDelta ||
            camera.orthographic != previous_.orthographic ||
            camera.fovDegrees != previous_.fovDegrees || camera.orthoHeight != previous_.orthoHeight) {
            rebase();
            return result;
        }

        const Vec delta = subtract(asVec(camera.eye), asVec(previous_.eye));
        const double distance = length(delta);
        // Relative rotation includes roll as well as yaw/pitch. The axis/angle
        // representation follows the shortest rotation.
        const Rotation rotation = relativeRotation(previousBasis_, current);
        const double degrees = rotation.angle * kRadiansToDegrees;
        previous_ = camera;
        previousBasis_ = current;
        if (distance > positive(config.teleportDistance, 10) ||
            degrees > positive(config.cameraCutRotationDegrees, 60)) {
            result.historyReset = true;
            return result;
        }

        const double minHorizon = positive(config.minHorizonSeconds, 0.025);
        const double maxHorizon = std::max(minHorizon, positive(config.maxHorizonSeconds, 0.25));
        result.measuredLatency = demandToDrawableMilliseconds.count != 0 &&
            std::isfinite(demandToDrawableMilliseconds.p95) && demandToDrawableMilliseconds.p95 >= 0;
        // P95 already includes feedback/admission/upload. One extra frame covers
        // forecast sampling to GPU demand execution; no stage latency is added twice.
        const double latency = result.measuredLatency ? demandToDrawableMilliseconds.p95 * 0.001 :
            positive(config.fallbackHorizonSeconds, 0.1);
        result.horizonSeconds = std::clamp(latency + deltaSeconds, minHorizon, maxHorizon);
        result.linearSpeed = distance / deltaSeconds;
        result.angularSpeedDegrees = degrees / deltaSeconds;
        result.translationDistance = distance > 1e-7 ? std::min(result.linearSpeed * result.horizonSeconds,
            positive(config.maxTranslationDistance, 2)) : 0;
        result.rotationDegrees = degrees > 0.001 ? std::min(result.angularSpeedDegrees * result.horizonSeconds,
            positive(config.maxRotationDegrees, 20)) : 0;
        result.active = result.translationDistance > 0 || result.rotationDegrees > 0;
        if (!result.active) { return result; }

        const Vec shift = distance > 0 ? multiply(delta, result.translationDistance / distance) : Vec{};
        const Vec eye = add(asVec(camera.eye), shift);
        const double angle = result.rotationDegrees / kRadiansToDegrees;
        const Vec forward = rotate(current.forward, rotation.axis, angle);
        const Vec up = rotate(current.up, rotation.axis, angle);
        // Retain the camera's focus distance; only pose is extrapolated. This also
        // handles orthographic cameras without interpreting orthoHeight as depth.
        result.camera.eye = asFloat(eye);
        result.camera.center = asFloat(add(eye, multiply(forward, current.focusDistance)));
        result.camera.up = asFloat(up);
        Basis forecast;
        if (!makeBasis(result.camera, forecast)) {
            result.camera = camera;
            result.active = false;
            result.historyReset = true;
        }
        return result;
    }

private:
    using Vec = std::array<double, 3>;
    struct Basis { Vec right{}, up{}, forward{}; double focusDistance = 0; };
    struct Rotation { Vec axis{0, 1, 0}; double angle = 0; };
    static constexpr double kRadiansToDegrees = 57.29577951308232;
    MeshletStreamPrefetchCamera previous_;
    Basis previousBasis_;
    bool valid_ = false;

    static double positive(double value, double fallback)
    {
        return std::isfinite(value) && value > 0 ? value : fallback;
    }
    static Vec asVec(const std::array<float, 3>& value)
    {
        return {value[0], value[1], value[2]};
    }
    static std::array<float, 3> asFloat(const Vec& value)
    {
        return {float(value[0]), float(value[1]), float(value[2])};
    }
    static Vec add(const Vec& a, const Vec& b)
    {
        return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
    }
    static Vec subtract(const Vec& a, const Vec& b)
    {
        return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
    }
    static Vec multiply(const Vec& value, double scale)
    {
        return {value[0] * scale, value[1] * scale, value[2] * scale};
    }
    static double dot(const Vec& a, const Vec& b)
    {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    }
    static Vec cross(const Vec& a, const Vec& b)
    {
        return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    }
    static double length(const Vec& value)
    {
        return std::sqrt(dot(value, value));
    }
    static bool finite(const Vec& value)
    {
        return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
    }
    static bool makeBasis(const MeshletStreamPrefetchCamera& camera, Basis& basis)
    {
        if (!finite(asVec(camera.eye)) || !finite(asVec(camera.center)) || !finite(asVec(camera.up)) ||
            !std::isfinite(camera.fovDegrees) || camera.fovDegrees <= 0 || camera.fovDegrees >= 180 ||
            !std::isfinite(camera.orthoHeight) || camera.orthoHeight <= 0) { return false; }
        basis.forward = subtract(asVec(camera.center), asVec(camera.eye));
        basis.focusDistance = length(basis.forward);
        if (basis.focusDistance <= 1e-7) { return false; }
        basis.forward = multiply(basis.forward, 1 / basis.focusDistance);
        basis.right = cross(basis.forward, asVec(camera.up));
        const double rightLength = length(basis.right);
        if (rightLength <= 1e-7) { return false; }
        basis.right = multiply(basis.right, 1 / rightLength);
        basis.up = cross(basis.right, basis.forward);
        return true;
    }
    static Rotation relativeRotation(const Basis& previous, const Basis& current)
    {
        // R_current * transpose(R_previous), with matching handedness in both bases.
        std::array<std::array<double, 3>, 3> matrix{};
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
                matrix[i][j] = current.right[i] * previous.right[j] + current.up[i] * previous.up[j] +
                    current.forward[i] * previous.forward[j];
            }
        }
        const double cosine = std::clamp((matrix[0][0] + matrix[1][1] + matrix[2][2] - 1) * 0.5, -1.0, 1.0);
        Rotation result{.angle = std::acos(cosine)};
        if (result.angle < 1e-8) { return result; }
        Vec axis{matrix[2][1] - matrix[1][2], matrix[0][2] - matrix[2][0], matrix[1][0] - matrix[0][1]};
        const double axisLength = length(axis);
        if (axisLength > 1e-7) {
            result.axis = multiply(axis, 1 / axisLength);
        } else {
            // Near 180 degrees is a camera cut under the default policy. Keep an
            // axis for callers deliberately allowing such a large single-frame turn.
            size_t major = 0;
            if (matrix[1][1] > matrix[major][major]) { major = 1; }
            if (matrix[2][2] > matrix[major][major]) { major = 2; }
            axis[major] = std::sqrt(std::max(0.0, (matrix[major][major] + 1) * 0.5));
            if (axis[major] > 1e-7) {
                const size_t a = (major + 1) % 3, b = (major + 2) % 3;
                axis[a] = (matrix[major][a] + matrix[a][major]) / (4 * axis[major]);
                axis[b] = (matrix[major][b] + matrix[b][major]) / (4 * axis[major]);
                result.axis = multiply(axis, 1 / length(axis));
            }
        }
        return result;
    }
    static Vec rotate(const Vec& value, const Vec& axis, double angle)
    {
        return add(add(multiply(value, std::cos(angle)), multiply(cross(axis, value), std::sin(angle))),
            multiply(axis, dot(axis, value) * (1 - std::cos(angle))));
    }
};

} // namespace metallic::render

