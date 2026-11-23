#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "inkling.h"

@interface InkDevice : NSObject
@property(strong) id<MTLDevice> device;
@property(strong) id<MTLCommandQueue> queue;
@property(strong) id<MTLLibrary> library;
@property(strong) NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
@end
@implementation InkDevice
@end

static _Thread_local char lastError[4096];
static int fail(NSString *message) {
    snprintf(lastError, sizeof(lastError), "%s", message.UTF8String);
    return -1;
}
const char *ink_error(void) { return lastError; }
void ink_release(void *handle) { if (handle) CFRelease(handle); }

extern const char shaderStart[] __asm("section$start$__TEXT$__inkling_msl");
extern const char shaderEnd[] __asm("section$end$__TEXT$__inkling_msl");

void *ink_open(void) {
    @autoreleasepool {
        InkDevice *ctx = [InkDevice new];
        ctx.device = MTLCreateSystemDefaultDevice();
        ctx.queue = [ctx.device newCommandQueue];
        if (!ctx.queue) { fail(@"Metal GPU unavailable"); return NULL; }
        MTLCompileOptions *opts = [MTLCompileOptions new];
        opts.languageVersion = MTLLanguageVersion3_1;
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        if (@available(macOS 15, *)) opts.mathMode = MTLMathModeSafe;
        else opts.fastMathEnabled = NO;
#else
        opts.fastMathEnabled = NO;
#endif
        NSError *error = nil;
        NSString *source = [[NSString alloc] initWithBytes:shaderStart length:(uintptr_t)shaderEnd - (uintptr_t)shaderStart
                                                  encoding:NSUTF8StringEncoding];
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 260000
        if (@available(macOS 26.1, *)) {
            opts.languageVersion = MTLLanguageVersion4_0;
            ctx.library = [ctx.device newLibraryWithSource:source options:opts error:&error];
        }
#endif
        if (!ctx.library) {
            opts.languageVersion = MTLLanguageVersion3_1;
            ctx.library = [ctx.device newLibraryWithSource:source options:opts error:&error];
        }
        if (!ctx.library) { fail(error.localizedDescription); return NULL; }
        ctx.pipelines = [NSMutableDictionary new];
        return (__bridge_retained void *)ctx;
    }
}

int ink_supports_gate_publication(void *context) {
    InkDevice *ctx = (__bridge InkDevice *)context;
    return [ctx.library.functionNames containsObject:@"published_blend_router"];
}

void *ink_buffer(void *context, size_t bytes) {
    @autoreleasepool {
        InkDevice *ctx = (__bridge InkDevice *)context;
        id<MTLBuffer> buf = [ctx.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buf) { fail(@"Metal buffer allocation failed"); return NULL; }
        return (__bridge_retained void *)buf;
    }
}
void *ink_contents(void *buffer) { return [(__bridge id<MTLBuffer>)buffer contents]; }

int ink_copy_cache(void *context, const InkCacheCopy *copies, size_t count) {
    @autoreleasepool {
        InkDevice *ctx = (__bridge InkDevice *)context;
        id<MTLCommandBuffer> command = [ctx.queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
        if (!encoder) return fail(@"Unable to encode cache growth");
        for (size_t i = 0; i < count; ++i) {
            const InkCacheCopy *copy = copies + i;
            [encoder copyFromBuffer:(__bridge id<MTLBuffer>)copy->source sourceOffset:copy->source_offset
                           toBuffer:(__bridge id<MTLBuffer>)copy->destination destinationOffset:0
                               size:copy->length];
        }
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        return command.status == MTLCommandBufferStatusCompleted ? 0 : fail(command.error.localizedDescription);
    }
}

static id<MTLComputePipelineState> getPipeline(InkDevice *ctx, const char *kernel, int quantization) {
        NSString *name = @(kernel);
        NSString *key = [NSString stringWithFormat:@"%@/%d", name, quantization];
        id<MTLComputePipelineState> pipeline = ctx.pipelines[key];
        if (!pipeline) {
            NSError *error = nil;
            id<MTLFunction> fn;
            if (quantization >= 0) {
                MTLFunctionConstantValues *values = [MTLFunctionConstantValues new];
                if ([name isEqualToString:@"decode_depth_input"] || [name isEqualToString:@"depth_sliding_heads"]) {
                    uint16_t config = quantization;
                    [values setConstantValue:&config type:MTLDataTypeUShort atIndex:1];
                } else {
                    bool fp4 = quantization == 4;
                    [values setConstantValue:&fp4 type:MTLDataTypeBool atIndex:0];
                }
                fn = [ctx.library newFunctionWithName:name constantValues:values error:&error];
            } else fn = [ctx.library newFunctionWithName:name];
            if (!fn) { fail(error.localizedDescription ?: name); return nil; }
            pipeline = [ctx.device newComputePipelineStateWithFunction:fn error:&error];
            if (!pipeline) { fail(error.localizedDescription); return nil; }
            if (pipeline.threadExecutionWidth != 32) { fail(@"Expected a 32-lane GPU SIMD group"); return nil; }
            ctx.pipelines[key] = pipeline;
        }
        return pipeline;
}

void *ink_pipeline(void *context, const char *name, int quantization) {
    @autoreleasepool {
        return (__bridge void *)getPipeline((__bridge InkDevice *)context, name, quantization);
    }
}

int ink_run(void *context, const char *kernel, void *const *buffers, unsigned count,
            const uint32_t *args, unsigned nargs, const uint32_t groups[3],
            const uint32_t threads[3], int quantization) {
    @autoreleasepool {
        InkDevice *ctx = (__bridge InkDevice *)context;
        NSString *name = @(kernel);
        id<MTLComputePipelineState> pipeline = getPipeline(ctx, kernel, quantization);
        if (!pipeline) return -1;
        if ((size_t)threads[0] * threads[1] * threads[2] > pipeline.maxTotalThreadsPerThreadgroup)
            return fail(@"Kernel exceeds this GPU's threadgroup limit");
        id<MTLCommandBuffer> command = [ctx.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) return fail(@"Unable to create Metal compute encoder");
        encoder.label = name;
        [encoder setComputePipelineState:pipeline];
        for (unsigned i = 0; i < count; ++i)
            [encoder setBuffer:(__bridge id<MTLBuffer>)buffers[i] offset:0 atIndex:i];
        if (nargs) [encoder setBytes:args length:nargs * sizeof(*args) atIndex:count];
        [encoder dispatchThreadgroups:MTLSizeMake(groups[0], groups[1], groups[2])
               threadsPerThreadgroup:MTLSizeMake(threads[0], threads[1], threads[2])];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        return command.status == MTLCommandBufferStatusCompleted ? 0 : fail(command.error.localizedDescription);
    }
}

static void *submit_schedule(void *context, const InkDispatch *steps, unsigned count, void *const *buffers, BOOL retained) {
    @autoreleasepool {
        InkDevice *ctx = (__bridge InkDevice *)context;
        // The synchronous schedule owns every bound resource until completion.
        id<MTLCommandBuffer> command = retained ? [ctx.queue commandBuffer] : [ctx.queue commandBufferWithUnretainedReferences];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) { fail(@"Unable to create Metal compute encoder"); return NULL; }
        id<MTLCommandBuffer> prefix = nil;
        void *bound[32] = {};
        uint32_t offsets[32] = {};
        for (unsigned s = 0; s < count; ++s) {
            // Start the GPU while the CPU encodes the rest of a long model pass.
            if (s == 16 && count > 32) {
                [encoder endEncoding];
                prefix = command;
                [prefix commit];
                command = retained ? [ctx.queue commandBuffer] : [ctx.queue commandBufferWithUnretainedReferences];
                encoder = [command computeCommandEncoder];
                if (!encoder) {
                    [prefix waitUntilCompleted];
                    fail(@"Unable to encode the rest of the schedule"); return NULL;
                }
                memset(bound, 0, sizeof(bound));
                memset(offsets, 0, sizeof(offsets));
            }
            const InkDispatch *op = &steps[s];
            id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)op->pipeline;
            if (op->threads > pipeline.maxTotalThreadsPerThreadgroup) {
                [encoder endEncoding];
                if (prefix) [prefix waitUntilCompleted];
                fail(@"Kernel exceeds this GPU's threadgroup limit"); return NULL;
            }
            [encoder setComputePipelineState:pipeline];
            for (unsigned i = 0; i < op->count; ++i) {
                void *binding = buffers[op->buffer[i]];
                if (bound[i] != binding) {
                    [encoder setBuffer:(__bridge id<MTLBuffer>)binding offset:op->offset[i] atIndex:i];
                } else if (binding && offsets[i] != op->offset[i]) {
                    [encoder setBufferOffset:op->offset[i] atIndex:i];
                }
                bound[i] = binding;
                offsets[i] = op->offset[i];
            }
            if (op->nargs) {
                [encoder setBytes:op->args length:op->nargs * sizeof(uint32_t) atIndex:op->count];
                bound[op->count] = (void *)UINTPTR_MAX;
            }
            [encoder dispatchThreadgroups:MTLSizeMake(op->group[0], op->group[1], op->group[2])
                   threadsPerThreadgroup:MTLSizeMake(op->threads, 1, 1)];
        }
        [encoder endEncoding];
        [command commit];
        return (__bridge_retained void *)(prefix ? @[prefix,command] : @[command]);
    }
}

int ink_wait(void *submitted) {
    @autoreleasepool {
        NSArray *commands = (__bridge_transfer NSArray *)submitted;
        int result = 0;
        for (id<MTLCommandBuffer> command in commands) {
            [command waitUntilCompleted];
            if (command.status != MTLCommandBufferStatusCompleted)
                result = fail(command.error.localizedDescription);
        }
        return result;
    }
}

void *ink_submit(void *context, const InkDispatch *steps, unsigned count, void *const *buffers) {
    return submit_schedule(context, steps, count, buffers, YES);
}

int ink_execute(void *context, const InkDispatch *steps, unsigned count, void *const *buffers) {
    void *command = submit_schedule(context, steps, count, buffers, NO);
    return command ? ink_wait(command) : -1;
}
