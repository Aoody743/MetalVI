#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 pos [[position]];
    float4 color [[flat]];
};

vertex VertexOut verify_vertex(uint vid [[vertex_id]], uint iid [[instance_id]],
                               constant uint *params [[buffer(4)]],
                               constant ushort &mode [[buffer(5)]]) {
    bool draw = mode == 0 && params[0] == 3 && params[1] == 1 && params[2] == 3 &&
                params[3] == 1 && params[4] == 0 && vid >= 3 && vid <= 5 && iid == 1;
    bool indexed = mode == 1 && params[0] == 3 && params[1] == 1 && params[2] == 1 &&
                   as_type<int>(params[3]) == -5 && params[4] == 4 && vid <= 2 && iid == 4;
    const float2 triangle[3] = {float2(-0.19, -0.53), float2(0.19, -0.53), float2(0, 0.57)};
    float2 co = triangle[vid % 3] + float2(-1.25 + 0.5 * iid, 0);
    float4 color = draw ? float4(1, 0, 0, 1) :
                   indexed ? float4(0, 1, 0, 1) : float4(1, 1, 0, 1);
    return {float4(co, 0, 1), color};
}

fragment float4 solid(VertexOut i [[stage_in]]) { return i.color; }

kernel void record_hits(device atomic_uint *hits [[buffer(0)]],
                        uint3 group [[threadgroup_position_in_grid]]) {
    atomic_fetch_add_explicit(hits + group.y * 4 + group.x, 1, memory_order_relaxed);
}

kernel void produce_count(device uint *count [[buffer(0)]]) {
    count[1] = 2;
}

using TestMesh = mesh<VertexOut, void, 3, 1, topology::triangle>;
[[mesh]] void test_mesh(TestMesh out, uint3 group [[threadgroup_position_in_grid]],
                         device atomic_uint *hits [[buffer(0)]]) {
    atomic_fetch_add_explicit(hits + group.x, 1, memory_order_relaxed);
    out.set_primitive_count(1);
    const float2 triangle[3] = {float2(-0.3, -0.53), float2(0.3, -0.53), float2(0, 0.57)};
    for (uint i = 0; i < 3; ++i) {
        out.set_vertex(i, {float4(triangle[i] + float2(group.x == 0 ? -0.5 : 0.5, 0), 0, 1), float4(1, 0, 0, 1)});
        out.set_index(i, i);
    }
}
