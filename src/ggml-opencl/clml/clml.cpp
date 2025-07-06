#include "ggml.h"

#include <CL/cl.h>
#include <cstring>
#include <vector>
#include "clml.h"

#ifdef GGML_OPENCL_USE_CLML
#include <CL/cl_qcom_ml_ops.h>
#endif

static cl_int g_clml_major = 0;
static cl_int g_clml_minor = 0;

#define CLML_CHECK(err)                                            \
    do {                                                           \
        cl_int err_ = (err);                                       \
        if (err_ != CL_SUCCESS) {                                  \
            ggml_clml_error(#err, __func__, __FILE__, __LINE__, err_); \
        }                                                          \
    } while (0)

#define CLML_CALL(API, ...)                                        \
    do {                                                           \
        cl_int rc = API(__VA_ARGS__);                              \
        CLML_CHECK(rc);                                            \
    } while (0)


int ggml_clml_extension_supported(const char *extensions, const char *needle) {
    if (!extensions || !needle) return 0;
    // Look for 'needle' as a full word (not as a substring inside another ext)
    const char *start = extensions;
    size_t needle_len = strlen(needle);
    while ((start = strstr(start, needle))) {
        if ((start == extensions || start[-1] == ' ')
            && (start[needle_len] == '\0' || start[needle_len] == ' ')) {
            return 1; // Found as a whole word
        }
        start += needle_len;
    }
    return 0;
}

void ggml_clml_query_version(void) {
    cl_uint count = 0;
    cl_int e = clQueryMLInterfaceVersionsQCOM(nullptr, nullptr, 0, &count);
    if (e != CL_SUCCESS || count == 0) return;

    std::vector<cl_int> major(count), minor(count);
    e = clQueryMLInterfaceVersionsQCOM(major.data(), minor.data(), count, nullptr);
    if (e == CL_SUCCESS) {
        g_clml_major = major[count - 1];
        g_clml_minor = minor[count - 1];
    }
}

ggml_clml_version ggml_clml_version_get(void) {
    ggml_clml_version v = { g_clml_major, g_clml_minor };
    return v;
}


static inline void ggml_clml_error(const char *stmt, const char *func,
                                   const char *file, int line, cl_int err) {
    fprintf(stderr, "CLML error: %s returned %d\n", stmt, err);
    fprintf(stderr, "  in function %s at %s:%d\n", func, file, line);
    GGML_ABORT("CLML error");
}