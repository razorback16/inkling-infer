#pragma once
#include <stdint.h>
#include <stddef.h>

// One context per serial inference queue. Handles remain owned by the caller.
void *ink_open(void);
int ink_supports_gate_publication(void *context);
const char *ink_error(void);
void ink_release(void *handle);
void *ink_buffer(void *context, size_t bytes);
void *ink_contents(void *buffer);
// Copy retained KV prefixes before the next model execution.
typedef struct {
    void *source, *destination;
    size_t source_offset, length;
} InkCacheCopy;
int ink_copy_cache(void *context, const InkCacheCopy *copies, size_t count);
// Buffer order follows the Metal function's signature. Optional inline uint32
// arguments occupy the next binding. Groups and group size are specified in 3D.
int ink_run(void *context, const char *kernel, void *const *buffers, unsigned count,
            const uint32_t *args, unsigned nargs, const uint32_t groups[3],
            const uint32_t threads[3], int quantization);

// A reusable dispatch schedule. Tensor IDs index the bindings supplied on each
// execution, allowing caches and workspace allocations to move independently.
typedef struct {
    void *pipeline;
    uint32_t group[3], threads, count, nargs;
    uint32_t buffer[31], offset[31], args[8];
} InkDispatch;
void *ink_pipeline(void *context, const char *kernel, int quantization);
int ink_execute(void *context, const InkDispatch *steps, unsigned count, void *const *buffers);

// The returned command owns its resources. Waiting consumes the handle.
void *ink_submit(void *context, const InkDispatch *steps, unsigned count, void *const *buffers);
int ink_wait(void *submitted);
