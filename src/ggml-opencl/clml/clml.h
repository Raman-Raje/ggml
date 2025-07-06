#pragma once

#include <CL/cl.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    cl_int major;
    cl_int minor;
} ggml_clml_version;

void ggml_clml_query_version(void);
int ggml_clml_extension_supported(const char *extensions, const char *needle);
ggml_clml_version ggml_clml_version_get(void);

#ifdef __cplusplus
}
#endif
