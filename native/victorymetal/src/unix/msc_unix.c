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

static NTSTATUS compile(struct msc_compile_params *p, struct msc_stage_in_params *stage_in,
                        struct msc_geometry_params *geometry) {
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
  IRCompilerSetCompatibilityFlags(compiler, IRCompatibilityFlagBoundsCheck | IRCompatibilityFlagTextureMinLODClamp);
  if (geometry && p->stage == IRShaderStageVertex) {
    IRCompilerEnableGeometryAndTessellationEmulation(compiler, true);
    IRCompilerSetInputTopology(compiler, IRInputTopologyTriangle);
  }
  if (stage_in)
    IRCompilerSetStageInGenerationMode(compiler, IRStageInCodeGenerationModeUseSeparateStageInFunction);
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
        if (geometry && p->stage == IRShaderStageVertex) {
          IRVersionedVSInfo vs;
          if (IRShaderReflectionCopyVertexInfo(named, IRReflectionVersion_1_0, &vs)) {
            geometry->vertex_output_size = vs.info_1_0.vertex_output_size_in_bytes;
            IRShaderReflectionReleaseVertexInfo(&vs);
          }
        } else if (geometry && p->stage == IRShaderStageGeometry) {
          IRVersionedGSInfo gs;
          if (IRShaderReflectionCopyGeometryInfo(named, IRReflectionVersion_1_0, &gs)) {
            if (gs.info_1_0.input_primitive == IRInputPrimitiveTriangle) {
              geometry->max_input_primitives = gs.info_1_0.max_input_primitives_per_mesh_threadgroup;
              geometry->max_payload_size = gs.info_1_0.max_payload_size_in_bytes;
              geometry->gs_instances = gs.info_1_0.instance_count;
            }
            IRShaderReflectionReleaseGeometryInfo(&gs);
          }
        }
        if (stage_in) {
          IRVersionedInputLayoutDescriptor layout = {0};
          layout.version = IRInputLayoutDescriptorVersion_1;
          layout.desc_1_0.numElements = stage_in->count;
          for (uint32_t i = 0; i < stage_in->count; ++i) {
            const struct msc_input_element *e = &stage_in->elements[i];
            layout.desc_1_0.semanticNames[i] = e->semantic;
            layout.desc_1_0.inputElementDescs[i] = (IRInputElementDescriptor1){
                .semanticIndex = e->semantic_index, .format = (IRFormat)e->format,
                .inputSlot = e->slot, .alignedByteOffset = e->offset,
                .inputSlotClass = e->per_instance ? IRInputClassificationPerInstanceData : IRInputClassificationPerVertexData,
                .instanceDataStepRate = e->per_instance ? e->step_rate : 0,
            };
          }
          IRMetalLibBinary *helper = IRMetalLibBinaryCreate();
          if (helper && IRMetalLibSynthesizeStageInFunction(compiler, named, &layout, helper)) {
            stage_in->metallib_len = IRMetalLibGetBytecodeSize(helper);
            stage_in->metallib = malloc(stage_in->metallib_len);
            if (stage_in->metallib)
              IRMetalLibGetBytecode(helper, stage_in->metallib);
          }
          if (helper)
            IRMetalLibBinaryDestroy(helper);
        }
      }
      IRShaderReflectionDestroy(named);
      if (stage_in && !stage_in->metallib) {
        free(p->metallib);
        p->metallib = NULL;
        p->metallib_len = 0;
        fail(p, "IRMetalLibSynthesizeStageInFunction", NULL);
      }
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
      } else if (p->stage == IRShaderStageMesh) {
        IRShaderReflection *refl = IRShaderReflectionCreate();
        if (IRObjectGetReflection(output, IRShaderStageMesh, refl)) {
          IRVersionedMSInfo info;
          if (IRShaderReflectionCopyMeshInfo(refl, IRReflectionVersion_1_0, &info)) {
            p->threadgroup[0] = info.info_1_0.num_threads[0];
            p->threadgroup[1] = info.info_1_0.num_threads[1];
            p->threadgroup[2] = info.info_1_0.num_threads[2];
            IRShaderReflectionReleaseMeshInfo(&info);
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

NTSTATUS _MSC_Compile(void *args) {
  return compile(args, NULL, NULL);
}

NTSTATUS _MSC_CompileStageIn(void *args) {
  struct msc_stage_in_params *p = args;
  p->metallib = NULL;
  p->metallib_len = 0;
  if (p->compile->stage != IRShaderStageVertex || !p->count || p->count > 31) {
    fail(p->compile, "invalid vertex input layout", NULL);
    return STATUS_SUCCESS;
  }
  return compile(p->compile, p, NULL);
}

NTSTATUS _MSC_CompileGeometry(void *args) {
  struct msc_geometry_params *p = args;
  struct msc_stage_in_params *s = &p->stage_in;
  s->metallib = NULL;
  s->metallib_len = 0;
  p->vertex_output_size = p->max_input_primitives = p->max_payload_size = p->gs_instances = 0;
  if (s->compile->stage == IRShaderStageVertex && s->count <= 31)
    return compile(s->compile, s, p);
  if (s->compile->stage == IRShaderStageGeometry && !s->count)
    return compile(s->compile, NULL, p);
  fail(s->compile, "invalid triangle geometry-emulation stage or vertex layout", NULL);
  return STATUS_SUCCESS;
}

NTSTATUS _MSC_Free(void *args) {
  struct msc_free_params *p = args;
  free(p->ptr);
  return STATUS_SUCCESS;
}
