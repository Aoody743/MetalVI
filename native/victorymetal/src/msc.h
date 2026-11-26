/* victorymetal: Metal Shader Converter (DXIL -> metallib) across the unixlib.
 * MetalVI's own addition to the winemetal layer (not from dxmt). */
#ifndef VICTORYMETAL_MSC_H
#define VICTORYMETAL_MSC_H

#include <stdint.h>

#ifndef WINEMETAL_API
#ifdef __WINE_PE_BUILD
#define WINEMETAL_API __declspec(dllexport)
#else
#define WINEMETAL_API
#endif
#endif

/* Unix call numbers, appended after winemetal's 146 (0..145). */
#define MSC_UNIX_COMPILE 146
#define MSC_UNIX_FREE 147
#define VM_UNIX_GPU_RESOURCE_ID 148
#define MSC_UNIX_COMPILE_STAGE_IN 149
#define VM_UNIX_RENDER_PIPELINE_STAGE_IN 150
#define MSC_UNIX_COMPILE_GEOMETRY 151
#define VM_UNIX_RENDER_PIPELINE_GEOMETRY 152

/* One root parameter of the root signature victory builds per pipeline
 * (pipeline/host.rs): 32-bit constants or a root descriptor. Heaps are
 * directly indexed (SM 6.6), so there are no tables. */
/* MSC_TABLE_SRV/UAV: a descriptor table of one view at the register (how
 * root SRVs/UAVs get bounds, as the D3D12 path's one-view tables). */
enum msc_param_kind { MSC_CONSTANTS = 0, MSC_CBV = 1, MSC_SRV = 2, MSC_UAV = 3, MSC_TABLE_SRV = 4, MSC_TABLE_UAV = 5 };

struct msc_root_param {
  uint32_t kind;
  uint32_t shader_register;
  uint32_t space;
  uint32_t num_32bit_values; /* constants only */
};

/* A static sampler (D3D12_STATIC_SAMPLER_DESC's fields, as numbers). */
struct msc_static_sampler {
  uint32_t filter, address_u, address_v, address_w;
  float mip_lod_bias;
  uint32_t max_anisotropy, comparison_func, border_color;
  float min_lod, max_lod;
  uint32_t shader_register, register_space;
};

/* Where a root parameter lives in the top-level argument buffer
 * (IRResourceLocation: type is IRResourceType numbering). */
struct msc_location {
  uint32_t type, space, slot, offset;
  uint64_t size;
};
#define MSC_MAX_LOCATIONS 64

/* stage: IRShaderStage numbering (metal_irconverter.h). */
struct msc_compile_params {
  const void *dxil;
  uint64_t dxil_len;
  const struct msc_root_param *params;
  uint32_t num_params;
  uint32_t num_samplers;
  const struct msc_static_sampler *samplers;
  uint32_t stage;
  uint32_t reserved;
  /* out: metallib bytes (unix malloc, give back with MSC_Free) */
  void *metallib;
  uint64_t metallib_len;
  /* out: compute threadgroup size from the reflection */
  uint32_t threadgroup[3];
  /* out: the metallib's entry point name */
  char entry[256];
  /* out: the top-level argument buffer layout of the root signature */
  uint32_t num_locations;
  uint32_t top_level_size;
  struct msc_location locations[MSC_MAX_LOCATIONS];
  /* out: 0 on success; otherwise error holds a message */
  int32_t ret;
  char error[1024];
};

struct msc_free_params {
  void *ptr;
};

struct msc_input_element {
  const char *semantic;
  uint32_t semantic_index, format, slot, offset, per_instance, step_rate;
};

struct msc_stage_in_params {
  struct msc_compile_params *compile;
  const struct msc_input_element *elements;
  uint32_t count;
  void *metallib;
  uint64_t metallib_len;
};

struct msc_geometry_params {
  struct msc_stage_in_params stage_in;
  uint32_t vertex_output_size;
  uint32_t max_input_primitives;
  uint32_t max_payload_size;
  uint32_t gs_instances;
};

/* gpuResourceID of a texture or sampler state (argument-buffer capable),
 * for descriptor entries; 0 for anything else. */
struct vm_resource_id_params {
  uint64_t handle;
  uint64_t ret;
};
WINEMETAL_API uint64_t VM_GpuResourceID(uint64_t handle);

WINEMETAL_API int MSC_Compile(struct msc_compile_params *params);
WINEMETAL_API int MSC_CompileStageIn(struct msc_stage_in_params *params);
WINEMETAL_API int MSC_CompileGeometry(struct msc_geometry_params *params);
WINEMETAL_API void MSC_Free(void *ptr);

#endif
