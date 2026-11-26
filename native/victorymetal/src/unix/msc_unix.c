/* victorymetal: unix side of the Metal Shader Converter calls (msc.h).
 * Links libmetalirconverter (Apple's Metal Shader Converter, x86_64 slice). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <metal_irconverter/metal_irconverter.h>

#include "../msc.h"

typedef int NTSTATUS;
#define STATUS_SUCCESS 0

static void fail(struct msc_compile_params *p, const char *what, IRError *err) {
  if (err) {
    snprintf(p->error, sizeof(p->error), "%s (IRError code %u)", what, IRErrorGetCode(err));
    IRErrorDestroy(err);
  } else {
    snprintf(p->error, sizeof(p->error), "%s", what);
  }
  p->ret = -1;
}

NTSTATUS _MSC_Compile(void *args) {
  struct msc_compile_params *p = args;
  p->metallib = NULL;
  p->metallib_len = 0;
  p->threadgroup[0] = p->threadgroup[1] = p->threadgroup[2] = 0;

  /* Root signature 1.1: constants and root descriptors in order, heaps
   * directly indexed (the flags victory's D3D12 root signatures carry). */
  IRRootParameter1 *rp = calloc(p->num_params ? p->num_params : 1, sizeof(IRRootParameter1));
  IRDescriptorRange1 *ranges = calloc(p->num_params ? p->num_params : 1, sizeof(IRDescriptorRange1));
  for (uint32_t i = 0; i < p->num_params; i++) {
    const struct msc_root_param *in = &p->params[i];
    rp[i].ShaderVisibility = IRShaderVisibilityAll;
    switch (in->kind) {
    case MSC_CONSTANTS:
      rp[i].ParameterType = IRRootParameterType32BitConstants;
      rp[i].Constants.ShaderRegister = in->shader_register;
      rp[i].Constants.RegisterSpace = in->space;
      rp[i].Constants.Num32BitValues = in->num_32bit_values;
      break;
    case MSC_CBV:
    case MSC_SRV:
    case MSC_UAV:
      rp[i].ParameterType = in->kind == MSC_CBV ? IRRootParameterTypeCBV
                            : in->kind == MSC_SRV ? IRRootParameterTypeSRV
                                                  : IRRootParameterTypeUAV;
      rp[i].Descriptor.ShaderRegister = in->shader_register;
      rp[i].Descriptor.RegisterSpace = in->space;
      rp[i].Descriptor.Flags = IRRootDescriptorFlagNone;
      break;
    case MSC_TABLE_SRV:
    case MSC_TABLE_UAV:
      ranges[i].RangeType = in->kind == MSC_TABLE_SRV ? IRDescriptorRangeTypeSRV : IRDescriptorRangeTypeUAV;
      ranges[i].NumDescriptors = 1;
      ranges[i].BaseShaderRegister = in->shader_register;
      ranges[i].RegisterSpace = in->space;
      ranges[i].Flags = IRDescriptorRangeFlagDescriptorsVolatile | IRDescriptorRangeFlagDataVolatile;
      ranges[i].OffsetInDescriptorsFromTableStart = 0;
      rp[i].ParameterType = IRRootParameterTypeDescriptorTable;
      rp[i].DescriptorTable.NumDescriptorRanges = 1;
      rp[i].DescriptorTable.pDescriptorRanges = &ranges[i];
      break;
    default:
      free(rp);
      free(ranges);
      fail(p, "unknown root parameter kind", NULL);
      return STATUS_SUCCESS;
    }
  }
  IRStaticSamplerDescriptor *ss = calloc(p->num_samplers ? p->num_samplers : 1, sizeof(IRStaticSamplerDescriptor));
  for (uint32_t i = 0; i < p->num_samplers; i++) {
    const struct msc_static_sampler *in = &p->samplers[i];
    ss[i].Filter = (IRFilter)in->filter;
    ss[i].AddressU = (IRTextureAddressMode)in->address_u;
    ss[i].AddressV = (IRTextureAddressMode)in->address_v;
    ss[i].AddressW = (IRTextureAddressMode)in->address_w;
    ss[i].MipLODBias = in->mip_lod_bias;
    ss[i].MaxAnisotropy = in->max_anisotropy;
    ss[i].ComparisonFunc = (IRComparisonFunction)in->comparison_func;
    ss[i].BorderColor = (IRStaticBorderColor)in->border_color;
    ss[i].MinLOD = in->min_lod;
    ss[i].MaxLOD = in->max_lod;
    ss[i].ShaderRegister = in->shader_register;
    ss[i].RegisterSpace = in->register_space;
    ss[i].ShaderVisibility = IRShaderVisibilityAll;
  }

  IRVersionedRootSignatureDescriptor desc;
  memset(&desc, 0, sizeof(desc));
  desc.version = IRRootSignatureVersion_1_1;
  desc.desc_1_1.NumParameters = p->num_params;
  desc.desc_1_1.pParameters = rp;
  desc.desc_1_1.NumStaticSamplers = p->num_samplers;
  desc.desc_1_1.pStaticSamplers = ss;
  desc.desc_1_1.Flags = IRRootSignatureFlagCBVSRVUAVHeapDirectlyIndexed | IRRootSignatureFlagSamplerHeapDirectlyIndexed;

  IRError *err = NULL;
  IRRootSignature *sig = IRRootSignatureCreateFromDescriptor(&desc, &err);
  free(rp);
  free(ranges);
  free(ss);
  if (!sig) {
    fail(p, "IRRootSignatureCreateFromDescriptor", err);
    return STATUS_SUCCESS;
  }

  p->num_locations = 0;
  p->top_level_size = 0;
  size_t nloc = IRRootSignatureGetResourceCount(sig);
  if (nloc > MSC_MAX_LOCATIONS) {
    IRRootSignatureDestroy(sig);
    fail(p, "root signature has too many parameters", NULL);
    return STATUS_SUCCESS;
  }
  if (nloc) {
    IRResourceLocation *loc = calloc(nloc, sizeof(IRResourceLocation));
    IRRootSignatureGetResourceLocations(sig, loc);
    for (size_t i = 0; i < nloc; i++) {
      p->locations[i].type = loc[i].resourceType;
      p->locations[i].space = loc[i].space;
      p->locations[i].slot = loc[i].slot;
      p->locations[i].offset = loc[i].topLevelOffset;
      p->locations[i].size = loc[i].sizeBytes;
      uint32_t end = loc[i].topLevelOffset + (uint32_t)loc[i].sizeBytes;
      if (end > p->top_level_size)
        p->top_level_size = end;
    }
    p->num_locations = (uint32_t)nloc;
    free(loc);
  }

  IRCompiler *compiler = IRCompilerCreate();
  IRCompilerSetGlobalRootSignature(compiler, sig);
  IRCompilerSetMinimumGPUFamily(compiler, IRGPUFamilyApple7);
  /* D3D12 semantics: reads past a view give 0, writes past it are dropped
   * (title memory is the GPU's memory here: a stray write is corruption). */
  IRCompilerSetCompatibilityFlags(compiler, IRCompatibilityFlagBoundsCheck);
  IRObject *input = IRObjectCreateFromDXIL(p->dxil, p->dxil_len, IRBytecodeOwnershipNone);
  IRObject *output = IRCompilerAllocCompileAndLink(compiler, NULL, input, &err);
  if (!output) {
    fail(p, "IRCompilerAllocCompileAndLink", err);
  } else {
    IRMetalLibBinary *lib = IRMetalLibBinaryCreate();
    if (!IRObjectGetMetalLibBinary(output, (IRShaderStage)p->stage, lib)) {
      fail(p, "IRObjectGetMetalLibBinary: no metallib for that stage", NULL);
    } else {
      size_t n = IRMetalLibGetBytecodeSize(lib);
      p->metallib = malloc(n);
      IRMetalLibGetBytecode(lib, p->metallib);
      p->metallib_len = n;
      p->ret = 0;
      IRShaderReflection *named = IRShaderReflectionCreate();
      p->entry[0] = 0;
      if (IRObjectGetReflection(output, (IRShaderStage)p->stage, named)) {
        const char *e = IRShaderReflectionGetEntryPointFunctionName(named);
        if (e)
          snprintf(p->entry, sizeof(p->entry), "%s", e);
      }
      IRShaderReflectionDestroy(named);
      if (p->stage == IRShaderStageCompute) {
        IRShaderReflection *refl = IRShaderReflectionCreate();
        if (IRObjectGetReflection(output, IRShaderStageCompute, refl)) {
          IRVersionedCSInfo info;
          if (IRShaderReflectionCopyComputeInfo(refl, IRReflectionVersion_1_0, &info)) {
            p->threadgroup[0] = info.info_1_0.tg_size[0];
            p->threadgroup[1] = info.info_1_0.tg_size[1];
            p->threadgroup[2] = info.info_1_0.tg_size[2];
            IRShaderReflectionReleaseComputeInfo(&info);
          }
        }
        IRShaderReflectionDestroy(refl);
      }
    }
    IRMetalLibBinaryDestroy(lib);
    IRObjectDestroy(output);
  }
  IRObjectDestroy(input);
  IRCompilerDestroy(compiler);
  IRRootSignatureDestroy(sig);
  return STATUS_SUCCESS;
}

NTSTATUS _MSC_Free(void *args) {
  struct msc_free_params *p = args;
  free(p->ptr);
  return STATUS_SUCCESS;
}
