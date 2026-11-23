OUT = ../.build/native
SHADERS = reference_math.metal steel.metal experts.metal attention.metal prefill.metal matmul.metal grouped.metal selected.metal fastdecode.metal layer.metal controller.metal eager_attention.metal

all: $(OUT)/libinkling.dylib

$(OUT)/inkling.metal: $(SHADERS) Makefile
	@mkdir -p $(@D)
	cat $(SHADERS) > $@

$(OUT)/libinkling.dylib: inkling.m inkling.h $(OUT)/inkling.metal Makefile
	xcrun clang -O3 -Wall -fobjc-arc -mmacosx-version-min=14.0 -dynamiclib \
	    -install_name @rpath/libinkling.dylib -framework Foundation -framework Metal \
	    -Wl,-sectcreate,__TEXT,__inkling_msl,$(OUT)/inkling.metal inkling.m -o $@
