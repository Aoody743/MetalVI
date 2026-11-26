/*
 * airconv stubs for the T1 smoke build.
 *
 * dxmt's winemetal unix side also implements the DXBC->AIR (sm50_*) entry
 * points, which are normally provided by libairconv (C++ + LLVM 15).
 * T1 only exercises the Metal object path (device / buffer / library-from-
 * source / compute), so the sm50 entry points are stubbed out here instead of
 * pulling in LLVM.  Every stub returns failure (-1) and reports "stubbed".
 * DO NOT ship this in a real victorymetal: link the real libairconv instead.
 */
#include <stddef.h>
#include <stdint.h>
#include "airconv_public.h"

static int stubbed(void)
{
    static const char msg[] =
        "airconv is stubbed out in the T1 smoke build; DXBC->metallib is unavailable\n";
    /* keep the string referenced so it shows up in the binary */
    return msg[0] == 0 ? -1 : -1;
}

AIRCONV_API int SM50Initialize(const void *pBytecode, size_t BytecodeSize, sm50_shader_t *ppShader,
                               struct MTL_SHADER_REFLECTION *pRefl, sm50_error_t *ppError)
{
    (void)pBytecode; (void)BytecodeSize; (void)pRefl;
    if (ppShader) *ppShader = 0;
    if (ppError) *ppError = 0;
    return stubbed();
}

AIRCONV_API void SM50Destroy(sm50_shader_t pShader) { (void)pShader; }

AIRCONV_API int SM50Compile(sm50_shader_t pShader, struct SM50_SHADER_COMPILATION_ARGUMENT_DATA *pArgs,
                            const char *FunctionName, sm50_bitcode_t *ppBitcode, sm50_error_t *ppError)
{
    (void)pShader; (void)pArgs; (void)FunctionName;
    if (ppBitcode) *ppBitcode = 0;
    if (ppError) *ppError = 0;
    return stubbed();
}

AIRCONV_API void SM50GetCompiledBitcode(sm50_bitcode_t pBitcode, struct SM50_COMPILED_BITCODE *pData)
{
    (void)pBitcode;
    if (pData) { pData->Data = 0; pData->Size = 0; }
}

AIRCONV_API void SM50DestroyBitcode(sm50_bitcode_t pBitcode) { (void)pBitcode; }

AIRCONV_API size_t SM50GetErrorMessage(sm50_error_t pError, char *pBuffer, size_t BufferSize)
{
    static const char msg[] = "stubbed airconv: DXBC->metallib not built in T1";
    size_t n = sizeof(msg) - 1;
    (void)pError;
    if (pBuffer && BufferSize) {
        size_t c = n < BufferSize - 1 ? n : BufferSize - 1;
        for (size_t i = 0; i < c; i++) pBuffer[i] = msg[i];
        pBuffer[c] = 0;
    }
    return n;
}

AIRCONV_API void SM50FreeError(sm50_error_t pError) { (void)pError; }

AIRCONV_API int SM50CompileTessellationPipelineHull(sm50_shader_t pVertexShader, sm50_shader_t pHullShader,
                                                    struct SM50_SHADER_COMPILATION_ARGUMENT_DATA *pHullShaderArgs,
                                                    const char *FunctionName, sm50_bitcode_t *ppBitcode,
                                                    sm50_error_t *ppError)
{
    (void)pVertexShader; (void)pHullShader; (void)pHullShaderArgs; (void)FunctionName;
    if (ppBitcode) *ppBitcode = 0;
    if (ppError) *ppError = 0;
    return stubbed();
}

AIRCONV_API int SM50CompileTessellationPipelineDomain(sm50_shader_t pHullShader, sm50_shader_t pDomainShader,
                                                      struct SM50_SHADER_COMPILATION_ARGUMENT_DATA *pDomainShaderArgs,
                                                      const char *FunctionName, sm50_bitcode_t *ppBitcode,
                                                      sm50_error_t *ppError)
{
    (void)pHullShader; (void)pDomainShader; (void)pDomainShaderArgs; (void)FunctionName;
    if (ppBitcode) *ppBitcode = 0;
    if (ppError) *ppError = 0;
    return stubbed();
}

AIRCONV_API int SM50CompileGeometryPipelineVertex(sm50_shader_t pVertexShader, sm50_shader_t pGeometryShader,
                                                  struct SM50_SHADER_COMPILATION_ARGUMENT_DATA *pVertexShaderArgs,
                                                  const char *FunctionName, sm50_bitcode_t *ppBitcode,
                                                  sm50_error_t *ppError)
{
    (void)pVertexShader; (void)pGeometryShader; (void)pVertexShaderArgs; (void)FunctionName;
    if (ppBitcode) *ppBitcode = 0;
    if (ppError) *ppError = 0;
    return stubbed();
}

AIRCONV_API int SM50CompileGeometryPipelineGeometry(sm50_shader_t pVertexShader, sm50_shader_t pGeometryShader,
                                                    struct SM50_SHADER_COMPILATION_ARGUMENT_DATA *pGeometryShaderArgs,
                                                    const char *FunctionName, sm50_bitcode_t *ppBitcode,
                                                    sm50_error_t *ppError)
{
    (void)pVertexShader; (void)pGeometryShader; (void)pGeometryShaderArgs; (void)FunctionName;
    if (ppBitcode) *ppBitcode = 0;
    if (ppError) *ppError = 0;
    return stubbed();
}

AIRCONV_API void SM50GetArgumentsInfo(sm50_shader_t pShader, struct MTL_SM50_SHADER_ARGUMENT *pConstantBuffers,
                                      struct MTL_SM50_SHADER_ARGUMENT *pArguments)
{
    (void)pShader; (void)pConstantBuffers; (void)pArguments;
}
