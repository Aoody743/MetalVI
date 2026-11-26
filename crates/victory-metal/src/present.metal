#include <metal_stdlib>
using namespace metal;

vertex float4 present_vertex(uint id [[vertex_id]]) {
    float2 p = float2((id << 1) & 2, id & 2);
    return float4(p * 2.0 - 1.0, 0.0, 1.0);
}

fragment float4 present_fragment(float4 position [[position]],
                                texture2d<float, access::read> src [[texture(0)]],
                                constant uint4 &viewport [[buffer(0)]]) {
    uint2 dst = uint2(position.xy) - viewport.xy;
    uint2 size = uint2(src.get_width(), src.get_height());
    uint2 texel = (2 * dst + 1) * size / (2 * viewport.zw);
    return float4(src.read(texel).rgb, 1.0);
}
