#ifndef METALLIC_VIEW_CONSTANTS
#define METALLIC_VIEW_CONSTANTS

struct ViewCameraConstants
{
    float4 eye;
    float4 center;
    float4 upProjection;
    float4 viewport;
    float4 clipOrtho;
};

struct ViewConstants
{
    ViewCameraConstants current;
    ViewCameraConstants previous;
    float4 jitter;
    uint4 frame;
    float4 outputSize;
};

#endif
