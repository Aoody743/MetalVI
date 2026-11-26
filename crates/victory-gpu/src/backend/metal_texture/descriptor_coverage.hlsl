cbuffer Args : register(b0) { uint kind, slot, x, y, z, lod, row, reserved; }
RWByteAddressBuffer outbuf : register(u0, space1);

[numthreads(4, 1, 1)]
void main(uint tid : SV_GroupIndex) {
    float4 value = 0;
    switch (kind) {
    case 0: { Texture1D<float4> t = ResourceDescriptorHeap[slot]; value = t.Load(int2(x, lod)); break; }
    case 1: { Texture1DArray<float4> t = ResourceDescriptorHeap[slot]; value = t.Load(int3(x, z, lod)); break; }
    case 2: { Texture2D<float4> t = ResourceDescriptorHeap[slot]; value = t.Load(int3(x, y, lod)); break; }
    case 3: { Texture2DArray<float4> t = ResourceDescriptorHeap[slot]; value = t.Load(int4(x, y, z, lod)); break; }
    case 4: { Texture3D<float4> t = ResourceDescriptorHeap[slot]; value = t.Load(int4(x, y, z, lod)); break; }
    case 5: {
        TextureCube<float4> t = ResourceDescriptorHeap[slot];
        SamplerState s = SamplerDescriptorHeap[0];
        float3 d = 0; d[x / 2] = (x % 2 == 0) ? 1 : -1;
        value = t.SampleLevel(s, d, lod);
        break;
    }
    case 6: {
        TextureCubeArray<float4> t = ResourceDescriptorHeap[slot];
        SamplerState s = SamplerDescriptorHeap[0];
        float3 d = 0; d[x / 2] = (x % 2 == 0) ? 1 : -1;
        value = t.SampleLevel(s, float4(d, z), lod);
        break;
    }
    case 7: { Texture2DMS<float4> t = ResourceDescriptorHeap[slot]; value = t.Load(int2(x, y), z); break; }
    case 8: { Texture2D<uint4> t = ResourceDescriptorHeap[slot]; value = t.Load(int3(x, y, lod)); break; }
    case 9: {
        Texture2D<float4> t = ResourceDescriptorHeap[slot];
        SamplerState s = SamplerDescriptorHeap[0];
        if (y == 0)
            value = t.SampleLevel(s, float2(0.01, 0.01), lod);
        else
            value = t.SampleGrad(s, float2(0.01, 0.01), float2(0.001, 0), float2(0, 0.001));
        break;
    }
    case 10: if (tid == 0) { RWTexture1D<float4> t = ResourceDescriptorHeap[slot]; value = t[x]; t[x] = float4(1, 0, 0, 1); } break;
    case 11: if (tid == 0) { RWTexture1DArray<float4> t = ResourceDescriptorHeap[slot]; value = t[uint2(x, z)]; t[uint2(x, z)] = float4(1, 0, 0, 1); } break;
    case 12: if (tid == 0) { RWTexture2D<float4> t = ResourceDescriptorHeap[slot]; value = t[uint2(x, y)]; t[uint2(x, y)] = float4(1, 0, 0, 1); } break;
    case 13: if (tid == 0) { RWTexture2DArray<float4> t = ResourceDescriptorHeap[slot]; value = t[uint3(x, y, z)]; t[uint3(x, y, z)] = float4(1, 0, 0, 1); } break;
    case 14: if (tid == 0) { RWTexture3D<float4> t = ResourceDescriptorHeap[slot]; value = t[uint3(x, y, z)]; t[uint3(x, y, z)] = float4(1, 0, 0, 1); } break;
    case 15: {
        Texture2D<float4> t = ResourceDescriptorHeap[slot];
        SamplerState s = SamplerDescriptorHeap[0];
        value = t.Sample(s, float2(0.01 + tid * 0.0001, 0.01));
        break;
    }
    default: break;
    }
    if (tid == 0) outbuf.Store4(row * 16, asuint(value));
}
