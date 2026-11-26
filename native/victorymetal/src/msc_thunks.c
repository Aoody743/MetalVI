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

uint64_t VM_GpuResourceID(uint64_t handle) {
  struct vm_resource_id_params params = {handle, 0};
  if (WINE_UNIX_CALL(VM_UNIX_GPU_RESOURCE_ID, &params))
    return 0;
  return params.ret;
}
