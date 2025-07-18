#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>

#ifdef GGML_USE_OPENCL
#include "ggml-opencl.h"
#endif

static void ggml_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

// This is a simple model with two tensors a and b
struct simple_model {
    struct ggml_tensor * a;

    // the backend to perform the computation (OpenCl, Vulkan, etc.)
    ggml_backend_t backend = NULL;

    // the allocator to allocate the tensors
    ggml_backend_buffer_t buffer = NULL;


    // the context to define the tensor information (dimensions, size, memory data)
    struct ggml_context * ctx;
};

// initialize the tensors of the model in this case two matrices 2x2
void load_model(simple_model & model, float * a, int rows_A, int cols_A) {
    ggml_log_set(ggml_log_callback_default, nullptr);

    // initialize the backend
    #ifdef GGML_USE_OPENCL
        fprintf(stderr, "%s: using OpenCL backend\n", __func__);
        model.backend = ggml_backend_opencl_init();
        if (!model.backend) {
            fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
        }
    #endif

    // if there aren't GPU Backends fallback to CPU backend
    if (!model.backend) {
        model.backend = ggml_backend_cpu_init();
    }

    int num_tensors = 1;

    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

    // create context
    model.ctx = ggml_init(params);

    model.a = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, cols_A, rows_A);
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);

    // load data from cpu memory to backend buffer
    ggml_backend_tensor_set(model.a, a, 0, ggml_nbytes(model.a));
}

// build the compute graph to perform a matrix multiplication
struct ggml_cgraph * build_graph(const simple_model& model) {
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };

    // create a temporally context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);

    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);

    // result = a*b^T
    struct ggml_tensor * result = ggml_relu(ctx0, model.a);

    // build operations nodes
    ggml_build_forward_expand(gf, result);

    // delete the temporally context used to build the graph
    ggml_free(ctx0);
    return gf;
}

// compute with backend
struct ggml_tensor * compute(const simple_model & model, ggml_gallocr_t allocr) {
    // reset the allocator to free all the memory allocated during the previous inference

    struct ggml_cgraph * gf = build_graph(model);

    // allocate tensors
    ggml_gallocr_alloc_graph(allocr, gf);

    int n_threads = 1; // number of threads to perform some operations with multi-threading

    if (ggml_backend_is_cpu(model.backend)) {
        ggml_backend_cpu_set_n_threads(model.backend, n_threads);
    }

    ggml_backend_graph_compute(model.backend, gf);

    // in this case, the output tensor is the last one in the graph
    return ggml_graph_node(gf, -1);
}

int main(void) {
    ggml_time_init();

    // initialize data of matrices to perform matrix multiplication
    const int rows_A = 4, cols_A = 2;

    /*
        matrix A = [
            2, -8,
            -5, 1,
            4, 2,
            8, 0
        ]    
    */
    float matrix_A[rows_A * cols_A] = {
        2, -8,
        -5, 1,
        4, 2,
        8, 0
    };

    simple_model model;
    load_model(model, matrix_A, rows_A, cols_A);

    // calculate the temporaly memory required to compute
    ggml_gallocr_t allocr = NULL;

    {
        allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));

        // create the worst case graph for memory usage estimation
        struct ggml_cgraph * gf = build_graph(model);
        ggml_gallocr_reserve(allocr, gf);
        size_t mem_size = ggml_gallocr_get_buffer_size(allocr, 0);

        fprintf(stderr, "%s: compute buffer size: %.4f KB\n", __func__, mem_size/1024.0);
    }

    // perform computation
    struct ggml_tensor * result = compute(model, allocr);

    // create a array to print result
    std::vector<float> out_data(ggml_nelements(result));

    // bring the data from the backend memory
    ggml_backend_tensor_get(result, out_data.data(), 0, ggml_nbytes(result));
   
    // expected result = [
    //     2, 0,
    //     0, 1,
    //     4, 2,
    //     8, 0
    // ]

    // print in matrix form
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_A; j++) {
            printf("%.2f ", out_data[i*cols_A + j]);
        }
        printf("\n");
    }
    printf("\n");

    // release backend memory used for computation
    ggml_gallocr_free(allocr);

    // free memory
    ggml_free(model.ctx);

    // release backend memory and free backend
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);    

    return 0;
}
