#include <metal_stdlib>
using namespace metal;

struct Options {
    uint kind, stride, max_count, has_count, index_capacity, object_stride, mesh_threads, groups, inputs, trace, first;
};

struct Record {
    uint words[5];
};

struct TraceRecord {
    uint raw_count, active;
    Record before, after;
};

kernel void translate_indirect(device const uint *src [[buffer(0)]],
                               device const uint *count [[buffer(1)]],
                               device Record *dst [[buffer(2)]],
                               constant Options &opts [[buffer(3)]],
                               device Record *draw_params [[buffer(4)]],
                               device atomic_uint *clamped [[buffer(5)]],
                               device uint *trace [[buffer(6)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= opts.max_count) return;
    Record result = {};
    Record original = {};
    // `first`: these records start at that record of the title's list (a
    // list of more than RECORD_LIMIT records encoded in parts); the count is
    // the whole list's.
    uint raw_count = opts.has_count ? *count : opts.first + opts.max_count;
    uint live = min(opts.max_count, raw_count > opts.first ? raw_count - opts.first : 0u);
    if (id < live) {
        device const uint *args = src + ulong(id) * opts.stride;
        uint words = opts.kind == 0 || opts.kind == 11 ? 4 : (opts.kind == 1 ? 5 : 3);
        for (uint i = 0; i < words; ++i) result.words[i] = args[i];
        original = result;
        if (opts.kind == 1) {
            uint first = result.words[2];
            if (first >= opts.index_capacity) {
                result = {};
            } else {
                result.words[0] = min(result.words[0], opts.index_capacity - first);
            }
        }
        if (opts.groups) {
            if (opts.kind == 2 || opts.kind == 10 || opts.kind == 12) {
                if (opts.kind == 12 && (result.words[1] != 1 || result.words[2] != 1)) {
                    result = {};
                } else {
                    uint remaining = opts.groups;
                    for (uint axis = 0; axis < 3; ++axis) {
                        result.words[axis] = min(result.words[axis], remaining);
                        remaining /= max(1u, result.words[axis]);
                    }
                }
            } else {
                uint instances = min(result.words[1], opts.inputs);
                if (opts.kind == 11) instances = min(instances, opts.groups);
                result.words[1] = instances;
                uint n = min(result.words[0], opts.inputs / max(instances, 1u));
                if (opts.kind == 11) {
                    n = min(n, (opts.groups / max(instances, 1u)) * opts.object_stride);
                    n -= n % 3;
                }
                result.words[0] = n;
            }
            bool changed = false;
            for (uint i = 0; i < words; ++i) changed |= original.words[i] != result.words[i];
            if (changed) atomic_fetch_add_explicit(clamped, 1u, memory_order_relaxed);
        }
    }
    if (opts.kind == 2) {
        result.words[0] = min(result.words[0], 65535u);
        result.words[1] = min(result.words[1], 65535u);
        result.words[2] = min(result.words[2], 65535u);
        if (result.words[2] != 0 && ulong(result.words[0]) * ulong(result.words[1]) > (1ul << 20) / ulong(result.words[2]))
            result = {};
    }
    if (opts.kind == 12 && (result.words[0] > 65535 || result.words[1] != 1 || result.words[2] != 1))
        result = {};
    if (opts.kind == 11) {
        // GeometryConfig::draw ABI: 20 bytes of args, then at +24 the
        // 24-byte IRRuntimeDrawInfo containing the live instance count.
        uint n = (result.words[0] / 3u) * 3u;
        uint instances = result.words[1];
        if (n == 0 || instances == 0 || n > (1u << 24) || instances > 65535u || ulong(n) * ulong(instances) > (1ul << 26)) {
            n = 0;
            instances = 0;
        }
        uint groups_x = n ? (n + opts.object_stride - 1u) / opts.object_stride : 0u;
        if (groups_x > 65535u || ulong(groups_x) * ulong(instances) > (1ul << 20)) {
            n = 0;
            instances = 0;
            groups_x = 0;
        }
        device uint *p = (device uint *)draw_params + ulong(id) * 12;
        p[0] = n; p[1] = instances; p[2] = result.words[2]; p[3] = result.words[3]; p[4] = 0; p[5] = 0;
        p[6] = 0x03030000; p[7] = opts.mesh_threads | (opts.object_stride << 16);
        p[8] = opts.mesh_threads | (1u << 16); p[9] = (opts.object_stride / 3u) | (3u << 16);
        p[10] = 0; p[11] = 0;
        result = {};
        if (groups_x != 0 && instances != 0) {
            result.words[0] = groups_x;
            result.words[1] = instances;
            result.words[2] = 1;
        }
    }
    if (opts.kind <= 1 && (result.words[0] > (1u << 24) || result.words[1] > 65535u || ulong(result.words[0]) * ulong(result.words[1]) > (1ul << 26)))
        result = {};
    dst[id] = result;
    if (opts.kind <= 1) draw_params[id] = result;
    if (opts.trace) {
        reinterpret_cast<device TraceRecord *>(trace)[id] = {raw_count, uint(id < live), original, result};
    }
}
