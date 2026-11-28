/* victorymetal: PE side of the Metal Shader Converter calls (msc.h). */
#include "wineunixlib.h"
#include "msc.h"

int MSC_Compile(struct msc_compile_params *params) {
  params->ret = -1;
  params->error[0] = 0;
  if (WINE_UNIX_CALL(MSC_UNIX_COMPILE, params))
    return -1;
  return params->ret;
}

int MSC_CompileStageIn(struct msc_stage_in_params *params) {
  params->compile->ret = -1;
  params->compile->error[0] = 0;
  if (WINE_UNIX_CALL(MSC_UNIX_COMPILE_STAGE_IN, params))
    return -1;
  return params->compile->ret;
}

void MSC_Free(void *ptr) {
  struct msc_free_params params = {ptr};
  WINE_UNIX_CALL(MSC_UNIX_FREE, &params);
}

int MSC_CompileGeometry(struct msc_geometry_params *params) {
  struct msc_compile_params *compile = params->stage_in.compile;
  compile->ret = -1;
  compile->error[0] = 0;
  if (WINE_UNIX_CALL(MSC_UNIX_COMPILE_GEOMETRY, params))
    return -1;
  return compile->ret;
}

uint64_t VM_GpuResourceID(uint64_t handle) {
  struct vm_resource_id_params params = {handle, 0};
  if (WINE_UNIX_CALL(VM_UNIX_GPU_RESOURCE_ID, &params))
    return 0;
  return params.ret;
}

uint32_t MSC_Features(void) {
  return MSC_FEATURE_FLAGS | MSC_FEATURE_ARCHIVE_STATS;
}

uint64_t VM_ArchiveAddFailures(void) {
  struct vm_archive_stats_params params = {0};
  if (WINE_UNIX_CALL(VM_UNIX_ARCHIVE_ADD_FAILURES, &params))
    return 0;
  return params.add_failures;
}
