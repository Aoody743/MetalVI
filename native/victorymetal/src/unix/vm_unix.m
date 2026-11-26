/* victorymetal: small Metal queries the winemetal layer lacks (msc.h). */
#import <Metal/Metal.h>
#include "../msc.h"

typedef int NTSTATUS;

NTSTATUS _VM_GpuResourceID(void *args) {
  struct vm_resource_id_params *p = args;
  p->ret = 0;
  id obj = (id)p->handle;
  if ([obj conformsToProtocol:@protocol(MTLTexture)])
    p->ret = [(id<MTLTexture>)obj gpuResourceID]._impl;
  else if ([obj conformsToProtocol:@protocol(MTLSamplerState)])
    p->ret = [(id<MTLSamplerState>)obj gpuResourceID]._impl;
  return 0;
}
