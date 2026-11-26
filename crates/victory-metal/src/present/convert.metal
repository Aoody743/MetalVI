#include <metal_stdlib>
using namespace metal;

struct Vertex {
    float4 position [[position]];
    float2 uv;
};

vertex Vertex convert_vertex(uint id [[vertex_id]]) {
    float2 uv = float2((id << 1) & 2, id & 2);
    return {float4(uv * float2(2, -2) + float2(-1, 1), 0, 1), uv};
}

float3 encode_srgb(float3 linear) {
    return select(1.055 * pow(linear, float3(1.0 / 2.4)) - 0.055, 12.92 * linear, linear <= 0.0031308);
}

float3 decode_srgb(float3 code) {
    return select(pow((code + 0.055) / 1.055, float3(2.4)), code / 12.92, code <= 0.04045);
}

struct Options {
    uint source_mode, srgb_target;
    float exposure;
};

fragment float4 convert_fragment(Vertex in [[stage_in]], texture2d<float> source [[texture(0)]],
                                 constant Options &opts [[buffer(0)]]) {
    constexpr sampler filter(filter::linear, address::clamp_to_edge);
    float3 rgb = source.sample(filter, in.uv).rgb;
    if (opts.source_mode == 0) {
        // SDR code values; only decode when the target encodes automatically.
        return float4(opts.srgb_target ? decode_srgb(rgb) : rgb, 1);
    }
    if (opts.source_mode == 2) {
        rgb = max(rgb, 0.0) * opts.exposure;
        float luma = dot(rgb, float3(0.2126, 0.7152, 0.0722));
        // Preserve hue until the final gamut clamp. The shoulder is continuous
        // in value and slope at paper white (luma 1 -> 0.75).
        float gain = luma <= 1.0 ? 0.75 : (1.0 - 0.25 / (1.0 + 3.0 * (luma - 1.0))) / luma;
        rgb *= gain;
    }
    rgb = clamp(rgb, 0.0, 1.0);
    return float4(opts.srgb_target ? rgb : encode_srgb(rgb), 1);
}
