#include <metal_stdlib>
using namespace metal;

struct Options {
    uint kind, stride, max_count, has_count, index_capacity;
};

struct Record {
    uint words[5];
};

kernel void translate_indirect(device const uint *src [[buffer(0)]],
                               device const uint *count [[buffer(1)]],
                               device Record *dst [[buffer(2)]],
                               constant Options &opts [[buffer(3)]],
                               device Record *draw_params [[buffer(4)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= opts.max_count) return;
    Record result = {};
    uint live = opts.has_count ? min(opts.max_count, *count) : opts.max_count;
    if (id < live) {
        device const uint *args = src + ulong(id) * opts.stride;
        uint words = opts.kind == 0 ? 4 : (opts.kind == 1 ? 5 : 3);
        for (uint i = 0; i < words; ++i) result.words[i] = args[i];
        if (opts.kind == 1) {
            uint first = result.words[2];
            if (first >= opts.index_capacity) {
                result = {};
            } else {
                result.words[0] = min(result.words[0], opts.index_capacity - first);
            }
        }
    }
    dst[id] = result;
    if (opts.kind <= 1) draw_params[id] = result;
}
