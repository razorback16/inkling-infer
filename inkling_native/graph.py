"""Record the 48 layer visits once and submit the entire schedule to Metal."""
import ctypes as C
import weakref
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .metal import Buffer, PTR, UINT, to_bits


class Dispatch(C.Structure):
    _fields_ = [("pipeline", PTR), ("group", UINT * 3), ("threads", UINT),
                ("count", UINT), ("nargs", UINT), ("buffer", UINT * 31),
                ("offset", UINT * 31), ("args", UINT * 8)]


@dataclass(frozen=True)
class Tensor:
    index: int
    offset: int = 0

    def at(self, byte_offset):
        return Tensor(self.index, self.offset + byte_offset)


class Schedule:
    def __init__(self, gpu):
        self.gpu = gpu
        self.objects = []
        self.sizes = {}
        self.steps = []
        self.weights = {}
        self.external = {}
        self.null = self.bind(None)
        gpu.lib.ink_pipeline.argtypes = [PTR, C.c_char_p, C.c_int]
        gpu.lib.ink_pipeline.restype = PTR
        gpu.lib.ink_execute.argtypes = [PTR, C.POINTER(Dispatch), UINT, C.POINTER(PTR)]
        gpu.lib.ink_execute.restype = C.c_int

    def bind(self, buffer):
        ref = Tensor(len(self.objects))
        self.objects.append(buffer)
        return ref

    def weight(self, buffer):
        if buffer is None:
            return self.null
        if buffer not in self.weights:
            self.weights[buffer] = self.bind(buffer)
        return self.weights[buffer]

    def dynamic(self, name):
        ref = self.bind(None)
        self.external[name] = ref.index
        return ref

    def temp(self, size):
        ref = self.bind(None)
        self.sizes[ref.index] = size
        return ref

    def op(self, name, refs, args, group, threads, bits=-1):
        self.steps.append((name, refs, args, group, threads, bits))

    def decode_workspace(self, last):
        if getattr(self, "rows", 0) != 1:
            return {}
        offsets, free = {}, defaultdict(list)
        extent = 0
        for step, (_, refs, *_) in enumerate(self.steps):
            scratch = {ref.index for ref in refs if ref.index in self.sizes}
            for idx in sorted(scratch):
                if idx in offsets:
                    continue
                size = (self.sizes[idx] + 255) // 256 * 256
                if free[size]:
                    offsets[idx] = free[size].pop()
                else:
                    offsets[idx] = extent
                    extent += size
            for idx in scratch:
                if last[idx] == step:
                    size = (self.sizes[idx] + 255) // 256 * 256
                    free[size].append(offsets[idx])
        if extent:
            workspace = Buffer(self.gpu, (extent,), np.uint8)
            for idx in offsets:
                self.objects[idx] = workspace
        return offsets

    def finish(self):
        # Reuse scratch once its last consumer has been encoded.
        last = {ref.index: i for i, step in enumerate(self.steps) for ref in step[1]}
        offsets = self.decode_workspace(last)
        free = defaultdict(list)
        pipelines = {}
        self.ops = (Dispatch * len(self.steps))()
        for i, (name, refs, args, group, threads, bits) in enumerate(self.steps):
            key = name, bits
            if key not in pipelines:
                pipe = self.gpu.lib.ink_pipeline(self.gpu.handle, name.encode(), bits)
                if not pipe:
                    raise RuntimeError(self.gpu.error())
                pipelines[key] = pipe
            op = self.ops[i]
            op.pipeline, op.threads, op.group[:] = pipelines[key], threads, group
            op.nargs, op.count = len(args), len(refs)
            op.args[:len(args)] = args
            for j, ref in enumerate(refs):
                idx = ref.index
                if idx in self.sizes and self.objects[idx] is None:
                    size = self.sizes[idx]
                    self.objects[idx] = free[size].pop() if free[size] else Buffer(self.gpu, (size,), np.uint8)
                op.buffer[j], op.offset[j] = idx, offsets.get(idx, 0) + ref.offset
            for idx in {r.index for r in refs}:
                if not offsets and idx in self.sizes and last[idx] == i:
                    free[self.sizes[idx]].append(self.objects[idx])
        self.handles = (PTR * len(self.objects))(*(b.handle if b else None for b in self.objects))
        self.steps.clear()

    def set(self, name, buffer):
        self.handles[self.external[name]] = buffer.handle

    def execute(self):
        if self.gpu.lib.ink_execute(self.gpu.handle, self.ops, len(self.ops), self.handles):
            raise RuntimeError(self.gpu.error())


HISTORY_WIDTH = 3 * (1024 * 2 + 2304 * 2)


def make_tables(model, fp8):
    activation, codes = model.gpu.expert_tables()
    return dict(fp8=model.gpu.upload(fp8), activation=activation, codes=codes,
                messages=model.gpu.upload(model.messages))


def gate_sigmoid(model):
    if "sigmoid" not in model.tables:
        table=Buffer(model.gpu,(65536,),np.float32)
        setup=Schedule(model.gpu)
        setup.op("build_attention_sigmoid",[setup.bind(table)],[],(256,1,1),256)
        setup.finish()
        setup.execute()
        model.tables["sigmoid"]=table
    return model.tables["sigmoid"]


class DecodePrediction:
    """An in-flight token owns its buffers and claims one uncommitted KV row per visit."""

    def __init__(self, plan, command, cache, history, result):
        self.plan = plan
        self.command = command
        self.release = weakref.finalize(self, plan.gpu.lib.ink_release, command)
        self.visits = [(c,c.position+1,c.length+1,c.window,c.written)
                       for c in cache[:plan.predicted_visits]]
        for _,_,length,_,extent in self.visits:
            extent[0] = length+1
        self.history, self.result = history, result

    def valid(self, cache):
        return (cache[0].graph_history is self.history and
                all(c is old and c.position == position and c.length == length
                    and c.written is extent and c.window == window
                    for c,(old,position,length,window,extent) in zip(cache,self.visits)))

    def release_row(self):
        for _,_,length,_,extent in self.visits:
            if extent[0] == length+1:
                extent[0] = length
        self.visits = ()

    def wait(self):
        self.release.detach()
        self.release_row()
        if self.plan.gpu.lib.ink_wait(self.command):
            raise RuntimeError(self.plan.gpu.error())


class ForwardPlan(Schedule):
    joint_groups=20
    def __init__(self, model, rows, logits):
        super().__init__(model.gpu)
        self.model, self.rows = model, rows
        self.input = Buffer(self.gpu, (rows, 2304), np.uint16)
        self.state = Buffer(self.gpu, (48, 12), np.uint32)
        self.output = Buffer(self.gpu, (model.w.tensors["unembed.weight"].shape[0],), np.float32) if logits else None
        self.state_ref = self.bind(self.state)
        self.query_tail = None
        self.old = self.dynamic("history")
        self.new = self.dynamic("next")
        self.image_mask = Buffer(self.gpu, (rows,), np.uint8)
        x = self.temp(rows * 2304 * 2)
        self.input_ref = self.bind(self.input)
        self.op("normalize_embedding", [self.input_ref, self.w("embed_norm.weight"),
                                        self.bind(self.image_mask), x], [], (rows,1,1), 256)
        self.blend_sums = {}
        self.prepared_inner = {}
        self.gate_publication = None
        self.prepared_router = {}
        self.fixed_depth = {}
        self.score_steps = []
        self.score_buffer = None
        blocks, bank = [], None
        self.predicted_visits = 48
        self.predict = rows == 1 and logits
        self.pending = None
        for visit, i in enumerate(model.visits):
            self.visit = visit
            if i == 6:
                if visit == 6:
                    prelude = x, blocks
                else:
                    bank = [*blocks[len(prelude[1])+1:], self.norm(x, "bf", "float")]
                    x, blocks = prelude
            if i == 18:
                bank = None
            messages = None
            if bank:
                query = self.weight(model.tables["messages"])
                if rows == 1:
                    # Each core layer reads the same previous pass. Submit
                    # all twelve independent four-query products together.
                    if i == 6:
                        normalized=self.temp(48*2304*2)
                        if len(bank)==2:
                            raw=self.temp(48*2304*4)
                            prepared=[raw.at(j*2304*4) for j in range(4)]
                            self.op("decode_previous_pass",[*bank,query,raw,normalized],[],(48,1,1),160)
                        else:
                            prepared=self.mix_queries(bank,query,4,batches=12)
                            self.op("normalize_pass_messages",[prepared[0],normalized],[],(48,1,1),256)
                        self.pass_messages={
                            prepared[0].at(j*2304*4):normalized.at(j*2304*2) for j in range(48)
                        }
                    messages = [value.at((i-6)*4*2304*4) for value in prepared]
                else:
                    messages = self.mix_queries(bank, query.at((i-6)*4*2304*4), 4)
            x, blocks = self.layer(i, x, blocks, messages)
            if visit+1 == self.predicted_visits:
                self.predicted_dispatch_count = len(self.steps)
        if logits:
            last = [b.at((rows-1)*2304*4) for b in blocks]
            last.append(self.norm(x.at((rows-1)*2304*2), "bf", "float", rows=1))
            mixed = self.mix(last, self.w("attn_res.output_query"), rows=1)
            act = self.norm(mixed, "float", "bf", "norm.weight", rows=1)
            vocab, width = model.w.tensors["unembed.weight"].shape
            weight,scales=[self.weight(b) for b in model.w.output_projection()]
            self.op("affine6_vocabulary_logits", [act, weight, scales, self.bind(self.output)],
                    [1, vocab, width], ((vocab+7)//8, 1, 1), 256)
        self.pack_fixed_queries()
        if self.predict:
            self.gpu.lib.ink_submit.argtypes = self.gpu.lib.ink_execute.argtypes
            self.gpu.lib.ink_submit.restype = PTR
            self.gpu.lib.ink_wait.argtypes = [PTR]
            self.gpu.lib.ink_wait.restype = C.c_int
            self.future_state = Buffer(self.gpu, self.state.shape, np.uint32)
            self.alternate_state = Buffer(self.gpu, self.state.shape, np.uint32)
            self.future_input = Buffer(self.gpu, self.input.shape, np.uint16)
            self.predicted_token = Buffer(self.gpu, (1,), np.uint32)
            tiles = (model.vocab + 1023) // 1024
            partial = self.bind(Buffer(self.gpu, (tiles,2), np.uint32))
            self.op("greedy_max_tiles",[self.weight(self.output),partial],[model.vocab],(tiles,1,1),256)
            self.op("greedy_embedding",[partial,self.w("embed.weight"),self.bind(self.future_input),
                    self.bind(self.predicted_token)],[tiles],(1,1,1),256)
        self.finish()
        if self.predict:
            self.prediction_ops = (Dispatch * self.predicted_dispatch_count)(*self.ops[:self.predicted_dispatch_count])
            self.with_greedy = self.ops
            self.ops = (Dispatch * (len(self.with_greedy)-2)).from_buffer(self.with_greedy)
        self.cache_bindings = [
            (self.external[j,"k"], self.external[j,"v"], self.ops[step].group)
            for j, step in enumerate(self.score_steps)
        ]
        if rows == 1:
            self.window_layout = None
            self.general_softmax = self.ops[self.score_steps[0] + 1].pipeline
            self.window_softmax = self.gpu.lib.ink_pipeline(self.gpu.handle, b"cached_window_probabilities", -1)
            if not self.window_softmax:
                raise RuntimeError(self.gpu.error())

    def w(self, name):
        return self.weight(self.model.w.buffer(name))

    def norm(self, x, src, dst, weight=None, rows=None, out=None, stride=2304, offset=0, packed=False):
        n = self.rows if rows is None else rows
        out = out or self.temp(n*stride*(2 if dst == "bf" else 4))
        self.op("norm_"+src+"_"+dst, [x, self.w(weight) if weight else self.null, out],
                [int(bool(weight)), stride, offset, int(packed)], (n, 1, 1), 256)
        return out

    def mix(self, states, query, rows=None, queries=1):
        n = self.rows if rows is None else rows
        out = self.temp(n*2304*4)
        refs = states + [self.null]*(5-len(states)) + [query,out]
        if n==1 and queries==1:
            self.op("depth_single",refs,[len(states),queries],(n,1,1),640)
        else:
            self.op("depth_mix",refs,[len(states),queries,n],(n,1,1),288)
        return out

    def mix_queries(self, states, query, count, batches=1):
        if count==1:
            return [self.mix(states,query)]
        if count==3 and len(states)<4:
            stride=self.rows*2304*4
            out=self.temp(count*stride)
            refs=states+[self.null]*(5-len(states))+[query,out]
            self.op("depth_mix",refs,[len(states),count,self.rows],(self.rows,count,1),288)
            return [out.at(j*stride) for j in range(count)]
        stride=self.rows*2304*4
        out=self.temp(batches*count*stride)
        refs=states+[self.null]*(5-len(states))+[query,out]
        self.op("depth_matrix",refs,[len(states),count,self.rows,0],(self.rows,batches,1),256)
        return [out.at(j*stride) for j in range(count)]

    def linear(self, x, name, fp32=False, rows=None, out=None):
        n = self.rows if rows is None else rows
        cout, cin = self.model.w.tensors[name].shape
        out = out or self.temp(n*cout*(4 if fp32 else 2))
        if n==1 and name.endswith("wo_ud.weight"):
            weight=self.weight(self.model.w.decode_projection(name))
            self.op("packed_bf16",[x,weight,out],[n,cout,cin,0],((cout+7)//8,1,1),256)
            return out
        if n==1 and fp32 and name.endswith("gate.weight"):
            self.op("router_gemv",[x,self.w(name),out],[n,cout,cin,0],(cout,1,1),32)
            return out
        if n==1 and not fp32 and cout<=384:
            self.op("split_bf",[x,self.w(name),out],[n,cout,cin,0],(cout,1,1),256)
            return out
        vector = cin==4608 or n < 8 and not (fp32 and n>=2 and "attn_sink" not in name)
        kernel = ("coalesced_" if vector else "dense_") + ("fp32" if fp32 else "bf16")
        groups = ((cout+3)//4,n,1) if vector else ((cout+31)//32,(n+63)//64,1)
        packed_weights=not vector and n>=8 and cin>=128
        weight=self.weight(self.model.w.prefill_projection(name)) if packed_weights else self.w(name)
        if packed_weights:kernel="prefill_packed_dense"
        if not vector and not fp32:
            approx=self.temp(n*cout*4)
            self.op("prefill_packed_dense" if packed_weights else "dense_fp32",[x,weight,approx],[n,cout,cin,0],groups,128)
            packed=self.weight(self.model.w.repair_projection(name))
            self.op("round_projection",[x,packed,approx,out],[n,cout,cin,0],
                    ((n*cout+1023)//1024,1,1),256)
        else:
            self.op(kernel,[x,weight,out],[n,cout,cin,0],groups,32 if vector else 128)
        return out

    def message(self,p,x,msg,weight,normalized_state=None):
        n=self.rows
        if n>=8:
            state=normalized_state or self.norm(x,"float","bf",packed=2)
            incoming=self.norm(msg,"float","bf",packed=2)
            z=self.temp(n*64*2)
            weights=self.weight(self.model.w.prompt_controller_weights(p+"down_weight"))
            self.op("prompt_controller_down",[state,incoming,weights,z],[n],(8,(n+7)//8,1),256)
            out=self.temp(n*2304*2)
            gates=self.weight(self.model.w.prompt_controller_gates(p))
            if n>=512:
                mixed=self.temp(n*2304*4)
                self.op("prompt_controller_blend",[z,gates,x,msg,mixed],[n],(18,(n+7)//8,1),128)
                return self.norm(mixed,"float","bf",weight,out=out)
            self.op("normalized_prompt_controller",[z,gates,x,msg,self.w(weight),out],[n],((n+7)//8,1,1),640)
            return out
        pair=self.temp(n*4608*2)
        self.norm(x,"float","bf",out=pair,stride=4608)
        self.norm(msg,"float","bf",out=pair,stride=4608,offset=2304)
        z=self.linear(pair,p+"down_weight")
        forget=self.linear(z,p+"forget_weight",True)
        gain=self.linear(z,p+"gain_weight",True)
        mixed=self.temp(n*2304*4)
        self.op("message_blend",[x,msg,forget,gain,mixed],[],(n*9,1,1),256)
        return self.norm(mixed,"float","bf",weight)

    def controllers(self, prefixes, states, msgs):
        count=len(states)
        def pad(items):
            return list(items)+[self.null]*(3-len(items))
        def weights(key):
            return pad([self.w(p+key+"_weight") for p in prefixes])
        if states[0] in self.prepared_inner:
            inner=self.prepared_inner.pop(states[0])
        else:
            paired=self.temp(count*4608*2)
            self.op("controller_norms",[*pad(states),*pad(msgs),paired],[],(2*count,1,1),256)
            inner=self.temp(count*64*2)
            self.op("controller_down",[paired,*weights("down"),inner],[],(64,count,1),256)
        output=self.temp(count*2304*4)
        parts=self.temp(count*160*4)
        paired=pad([self.weight(self.model.w.paired_blend_weights(p)) for p in prefixes])
        self.op("blend_with_sums",[inner,*paired,*pad(states),*pad(msgs),output,parts],[],(40,count,1),256)
        self.blend_sums[output]=parts
        return [output.at(j*2304*4) for j in range(count)]

    def conv(self,p,x,slot,dynamic=None,rows=None):
        n=self.rows if rows is None else rows
        width=[1024,1024,2304,2304][slot]
        offset=(self.visit*HISTORY_WIDTH+[0,3072,6144,13056][slot])*2
        out=self.temp(n*width*2)
        if dynamic:
            delta=self.linear(dynamic,p+".weight_proj",rows=n)
            w=self.w(p+".static_w")
        else:
            w,delta=self.w(p+".weight"),self.null
        self.op("short_conv",[x,self.old.at(offset),self.new.at(offset),w,delta,out],
                [width,n,int(bool(dynamic)),0],(width//256,n,1),256)
        return out

    def prompt_attention_projections(self, prefix, q, k, v):
        names = ("wq_du", "wr_du", "attn_sink_proj", "gate_attn", "wk_dv", "wv_dv")
        inputs = (q, q, q, q, k, v)
        project, outputs = [], []
        tiles = 0
        for name, source in zip(names, inputs):
            weight = prefix + name + ".weight"
            columns, _ = self.model.w.tensors[weight].shape
            rounded = self.temp(self.rows * columns * 2)
            project.extend((source, self.weight(self.model.w.prefill_projection(weight)),
                            self.weight(self.model.w.repair_projection(weight)), rounded))
            tiles += (columns + 31) // 32
            outputs.append(rounded)
        self.op("repaired_attention_projections", project, [self.rows],
                (tiles, (self.rows + 63) // 64, 1), 128)
        return outputs

    def attention(self,p,inputs,local,query_rows=None,blend=None,depth=None):
        q,k,v=inputs
        n=self.rows
        state=self.state_ref.at(self.visit*48)
        fp8=self.model.cache_dtype==np.uint8
        key=self.dynamic((self.visit,"k"))
        val=self.dynamic((self.visit,"v"))
        if n==1:
            names=("wq_du","wr_du","attn_sink_proj","gate_attn","wk_dv","wv_dv")
            dims=(3072,384,24,3072,1024,1024)
            outputs=[self.temp(d*2) for d in dims]
            head_indices = (0,1,2,4,5)
            head_names = [p+names[j]+".weight" for j in head_indices]
            if not local:
                head_names.extend(p+name+"_sconv.weight_proj" for name in ("k","v"))
            buffers = [self.model.w.paired_high_bytes(name, four_rows=True) for name in head_names]
            head_prefix = "paired_"
            if any(buf is None for buf in buffers):
                head_prefix = ""
                buffers = [self.model.w.decode_projection(name) for name in head_names]
            weights = [None]*len(names)
            for j, buf in zip(head_indices, buffers):
                weights[j] = self.weight(buf)
            gate_name = p+names[3]+".weight"
            gate_weights = self.model.w.paired_high_bytes(gate_name, four_rows=True)
            gate_prefix = "paired_" if gate_weights is not None else ""
            weights[3] = self.weight(gate_weights or self.model.w.decode_projection(gate_name))
            if local:
                query,r=self.temp(3072*2),self.temp(384*4)
                sinks,gate=outputs[2:4]
                offset=self.visit*HISTORY_WIDTH*2
                refs=[q,k,v,*weights,self.w(p+"q_l2_clip.scale"),self.w(p+"rel_l2_clip.scale"),
                      self.w(p+"k_sconv.weight"),self.w(p+"v_sconv.weight"),
                      self.old.at(offset),self.new.at(offset),key,val,
                      self.weight(self.model.tables["fp8"]),state,query,r,gate,sinks]
                if depth is not None and head_prefix=="paired_":
                    depth_refs,config=depth
                    # Head weights, cache buffers and recurrent state keep the same relative order.
                    head_refs=[*refs[3:6],*refs[7:21],refs[22]]
                    blocks=depth_refs[:4]
                    fixed_scores=depth_refs[13:17]
                    mixed=[depth_refs[5],depth_refs[9],depth_refs[10],depth_refs[11],q]
                    self.op("depth_sliding_heads",[*head_refs,*blocks,*fixed_scores,*mixed],[],(44,1,1),1024,config)
                elif blend is not None and head_prefix=="paired_":
                    source,norms,parts=blend
                    refs[:3]=[source[0],norms[0],parts]
                    refs[6]=q
                    self.op("blended_sliding_heads",refs,[],(44,1,1),1024)
                else:
                    if depth is not None:
                        depth_refs,config=depth
                        self.op("decode_depth_input",depth_refs,[],(1,1,1),640,config)
                    if blend is not None:
                        source,norms,parts=blend
                        self.op("normalize_blend_inputs",[*source,*norms,parts,q],[1],(18,3,1),128)
                    self.op(head_prefix+"sliding_heads",refs,[],(44,1,1),1024)
            else:
                taps=self.temp(64*2)
                tap_weights=[self.weight(buf) for buf in buffers[5:]]
                self.op(head_prefix+"global_head_projections",[q,k,v,*weights[:3],*weights[4:],*outputs[:3],*outputs[4:],
                        *tap_weights,taps],[],(175,1,1),256)
                qp,rp,sinks,gate,kp,vp=outputs
        elif n >= 64 and (query_rows is None or query_rows == n):
            qp,rp,sinks,gate,kp,vp = self.prompt_attention_projections(p,q,k,v)
        else:
            query_rows=n if query_rows is None else query_rows
            q=q.at((n-query_rows)*2304*2)
            qp=self.linear(q,p+"wq_du.weight",rows=query_rows)
            rp=self.linear(q,p+"wr_du.weight",rows=query_rows)
            sinks=self.linear(q,p+"attn_sink_proj.weight",rows=query_rows)
            gate=self.linear(q,p+"gate_attn.weight",rows=query_rows)
            kp=self.linear(k,p+"wk_dv.weight")
            vp=self.linear(v,p+"wv_dv.weight")
        if n!=1:
            kin=self.conv(p+"k_sconv",kp,0,None if local else k)
            vin=self.conv(p+"v_sconv",vp,1,None if local else v)
            self.op("store_kv",[kin,vin,key,val,self.weight(self.model.tables["fp8"]),state],
                    [int(fp8)],(8,n,1),32)
        if query_rows is not None and query_rows<n:
            self.query_tail=Buffer(self.gpu,(12,),np.uint32)
            state=self.bind(self.query_tail)
            n=query_rows
        def clip(width,proj,scale):
            out=self.temp(n*24*width*2)
            self.op("query_clip",[proj,self.w(p+scale+".scale"),out,state],[width,int(not local)],(24,n,1),32)
            return out
        scores=self.dynamic((self.visit,"scores"))
        gated=self.temp(n*3072*2)
        if n==1:
            if not local:
                query,r=self.temp(3072*2),self.temp(384*4)
                offset=self.visit*HISTORY_WIDTH*2
                self.op("decode_global_heads",[kp,vp,taps,self.w(p+"k_sconv.static_w"),self.w(p+"v_sconv.static_w"),
                        self.old.at(offset),self.new.at(offset),key,val,self.weight(self.model.tables["fp8"]),state,
                        qp,rp,self.w(p+"q_l2_clip.scale"),self.w(p+"rel_l2_clip.scale"),query,r],[],(14,1,1),256)
            self.op("decode_scores" if local else "decode_score_pairs",[query,key,r,self.w(p+"rel_logits_proj.proj"),scores,state],[],
                    (1,1,8),128)
        else:
            query=clip(128,qp,"q_l2_clip")
            r=clip(16,rp,"rel_l2_clip")
            bias=self.temp(n*24*1024*2)
            self.op("relative_bias",[r,self.w(p+"rel_logits_proj.proj"),bias],[],(2,n*24,1),128)
            self.op("attention_scores",[query,key,bias,scores,state],[],(1,(n+31)//32,8),384)
        self.score_steps.append(len(self.steps)-1)
        if n!=1 and local:
            self.op("sliding_row_probabilities",[scores,sinks,state],[],(n*6,1,1),128)
        else:
            self.op("decode_probabilities" if n==1 else "attention_probabilities",[scores,sinks,state],[],(n*24,1,1),256)
        if n==1:
            kernel="projected_decode_av" if local else "global_projected_decode_av"
            table=self.weight(gate_sigmoid(self.model))
            self.op(gate_prefix+kernel,[scores,val,q,weights[3],gated,state,table],[],(16,8,1),320)
        else:
            context=self.temp(n*3072*4)
            self.op("ordered_attention_values",[scores,val,context,state],[],(2,(n+15)//16,24),256)
            self.op("lookup_gate_context",[context,gate,gated,self.weight(gate_sigmoid(self.model))],[],(n*12,1,1),256)
        return gated if n==1 else self.linear(gated,p+"wo_ud.weight",rows=n)

    def router_buffers(self, prefix, activation):
        gate_ids = None
        if self.gpu.supports_gate_publication and prefix+"experts.w13_weight_amax" in self.model.w.tensors:
            if self.gate_publication is None:
                # 128 (rank, probability) pairs and a final uint arrival counter.
                self.gate_publication = Buffer(self.gpu, (257,), np.uint32)
            values = self.weight(self.gate_publication)
            gate_ids = self.temp(8*4)
        else:
            values = self.temp(128*4)
        self.prepared_router[activation] = values, gate_ids
        return values, gate_ids

    def moe(self,p,x,residual=None,rows=None):
        n=self.rows if rows is None else rows
        gate_input=x
        published_ids=None
        if x in self.prepared_router:
            logits,published_ids=self.prepared_router.pop(x)
        elif n==1:
            logits,gate_input=self.temp(128*4),self.temp(2304*2)
            self.op("expert_router",[x,self.weight(self.model.w.decode_router(p+"gate.weight")),logits,gate_input],[],(128,1,1),32)
        else:
            logits=self.linear(x,p+"gate.weight",True,rows=n)
        ids,routes=self.temp(n*8*4),self.temp(n*8*4)
        banks=[[self.weight(b) for b in self.model.w.expert(p+"experts."+k)] for k in ("w13_weight","w2_weight")]
        bits=[8 if bank[2]==self.null else 4 for bank in banks]
        overlap=n==1 and bits[0]==4
        if published_ids is not None:
            gate_ids,rank=published_ids,logits
        elif overlap:
            gate_ids=self.temp(8*4)
            rank=self.temp(128*8)
            self.op("gate_expert_indices",[logits,self.w(p+"gate.bias"),gate_ids,rank],[],(1,1,1),128)
        else:
            self.op("expert_route",[logits,self.w(p+"gate.bias"),ids,routes],[],(n,1,1),128)
        active,out=self.temp(n*8*1152*2),self.temp(n*2304*2)
        if n<8:
            codes=self.weight(self.model.tables["codes"])
            lut=self.weight(self.model.tables["activation"])
            interleaved=n==1 and bits[0]==8
            if overlap:
                self.op("gate_and_route_weights",[gate_input,gate_ids,*banks[0],codes,lut,active,rank,ids,routes],[],(145,8,1),128,4)
            else:
                self.op("selected_gate_packed" if n==1 else "selected_gate",[gate_input,ids,*banks[0],codes,lut,active],[],(8,72,n) if interleaved else (144,8,n),256 if interleaved else 128,bits[0])
            refs=[active,ids,*banks[1],codes,routes,out]
            if residual is None:
                self.op("selected_down",refs,[],(576,n,1),256,bits[1])
            else:
                offset=(self.visit*HISTORY_WIDTH+13056)*2
                refs.extend([self.old.at(offset),self.new.at(offset),
                             self.w(p.removesuffix("mlp.")+"mlp_sconv.weight"),residual])
                tiles=2 if bits[1]==8 else 1
                self.op("finish_selected_down",refs,[],(576//tiles,1,1),256*tiles,bits[1])
        else:
            cap=min(n*8,(n*8+127)//128+127)
            counts,cursors=self.temp(512),self.temp(512)
            tiles,order=self.temp(cap*16),self.temp(n*8*8)
            self.op("expert_counts",[ids,counts],[n*8],(128,1,1),256)
            self.op("expert_schedule",[counts,cursors,tiles],[cap],(1,1,1),128)
            self.op("expert_order",[ids,cursors,order],[n*8],((n*8+255)//256,1,1),256)
            states=self.temp(n*8*2304*2)
            tables=[self.weight(self.model.tables[k]) for k in ("activation","codes")]
            groups=(72*16,(cap+15)//16,1)
            for name,source,bank,dest,bit in zip(("grouped_gate","grouped_down"),(x,active),banks,(active,states),bits):
                events=self.temp((cap*72+n*8*1152)*2 if name=="grouped_gate" else n*2304)
                self.op(name,[source,*bank,tiles,order,dest,*tables,events],[cap],groups,128,bit)
                if name=="grouped_gate":
                    self.op("repair_"+name,[events,source,*bank,tables[1],order,tiles,tables[0],dest],
                            [cap],groups,256,bit)
            self.op("mix_down_replay",[states,routes,events,active,*banks[1],tables[1],ids,order.at(n*8*4),out],
                    [],(9,n,1),256,bits[1])
        return out

    def fixed_scores(self, blocks, query_id):
        result=[]
        for block in blocks:
            table, used, *_=self.fixed_depth[block]
            used.add(query_id)
            result.append(table.at(query_id*4))
        return result+[self.null]*(4-len(result))

    def pack_fixed_queries(self):
        for _, used, query_ref, index_ref, step in self.fixed_depth.values():
            ids=tuple(sorted(used))
            key=("depth_lookup",ids)
            if key not in self.model.w.buffers:
                indices=np.zeros((len(ids)+7)//8*8,np.uint32)
                indices[:len(ids)]=ids
                queries=np.zeros((len(indices)//8,2304,8),np.float32)
                for slot,q in enumerate(ids):
                    name=f"attn_res.mlp_queries.{q//2}" if q%2 else f"attn_res.attn_queries.{q//2-1}"
                    queries[slot//8,:,slot%8]=self.model.w.array(name).reshape(-1,2304)[0]
                self.model.w.buffers[key]=(self.gpu.upload(queries),self.gpu.upload(indices))
            self.objects[query_ref.index],self.objects[index_ref.index]=self.model.w.buffers[key]
            name,refs,_,_,threads,bits=self.steps[step]
            groups=(len(ids)+7)//8
            if name.startswith("joint_output_"):
                rows=threads//32
                groups+=(2304+rows-1)//rows
            self.steps[step]=(name,refs,[len(ids)],(groups,1,1),threads,bits)

    def global_depth_inputs(self,blocks,hidden,query,p,messages):
        count=len(blocks)+1
        output=self.temp(3*(64 if messages else 2304)*2)
        raw=self.temp(3*2304*4) if messages else self.null
        if messages:
            weights=[self.weight(self.model.w.controller_stream_weights(p+"controllers."+n+".down_weight"))
                     for n in ("attn","k","v")]
        else:
            weights=[self.w(p+n+"_input_norm.weight") for n in ("q","k","v")]
        normalized=[self.pass_messages[m] for m in messages[:3]] if messages else [self.null]*3
        refs=blocks+[self.null]*(4-len(blocks))+[hidden,query,*weights,*normalized,raw,output]
        self.op("global_depth_inputs",refs,[count,int(bool(messages))],(16 if messages else 1,3,1),512 if messages else 288)
        if messages:
            self.prepared_inner[raw]=output
            return [raw.at(j*2304*4) for j in range(3)]
        return [output.at(j*2304*2) for j in range(3)]

    def layer(self,i,x,blocks,messages):
        p=f"layers.{i}."
        local=i in self.model.local
        global_depth=self.rows==1 and not local and i%6!=0
        depth=None
        fused=self.rows==1 and local and i!=0
        if fused:
            norm=self.temp(2304*4) if i%6==0 else self.null
            mixed=self.temp(2304*(4 if messages else 2))
            refs=blocks+[self.null]*(4-len(blocks))+[
                self.null,x,self.null,self.null,self.null,self.weight(self.model.w.depth_query_streams(f"attn_res.attn_queries.{i-1}")),
                self.w(p+"attn_norm.weight"),norm,mixed]+self.fixed_scores(blocks,2*i)
            config = len(blocks)+1 | (int(i%6==0)<<3) | (int(not messages)<<4) | 32
            if messages:
                inner=self.temp(3*64*2)
                self.prepared_inner[mixed]=inner
                weights=[self.weight(self.model.w.controller_stream_weights(p+"controllers."+name+".down_weight")) for name in ("attn","k","v")]
                refs.extend([weights[0],inner,self.pass_messages[messages[0]],*weights[1:]])
                config|=128|256
            else:
                refs.extend([self.null]*5)
            if messages:
                self.op("decode_depth_input",refs,[],(16,3,1),640,config)
            else:
                depth=(refs,config)
            inputs=[mixed]
        elif global_depth:
            q=self.w(f"attn_res.attn_queries.{i-1}")
            inputs=self.global_depth_inputs(blocks,x,q,p,messages)
        else:
            norm=self.norm(x,"bf","float")
            if i:
                q=self.w(f"attn_res.attn_queries.{i-1}")
                inputs=self.mix_queries(blocks+[norm],q,1 if local else 3)
            else:
                inputs=[x]
        if local:
            inputs*=3
        norms=["attn_norm"]*3 if local else ["q_input_norm","k_input_norm","v_input_norm"]
        if messages:
            prefixes=[p+"controllers."+name+"." for name in ("attn","k","v")]
            if self.rows==1:
                inputs=self.controllers(prefixes,inputs,messages[:3])
            else:
                shared_state=self.norm(inputs[0],"float","bf",packed=2) if local and self.rows>=8 else None
                inputs=[self.message(prefix,state,msg,p+name+".weight",shared_state)
                        for prefix,state,msg,name in zip(prefixes,inputs,messages,norms)]
        blend=None
        if self.rows==1 and i and (messages or (not local and not global_depth)):
            output=self.temp(3*2304*2)
            refs=[*inputs,*[self.w(p+n+".weight") for n in norms]]
            if inputs[0] in self.blend_sums:
                parts=self.blend_sums.pop(inputs[0])
                if local:
                    blend=(inputs,refs[3:],parts)
                else:
                    self.op("normalize_blend_inputs",[*refs,parts,output],[0],(18,3,1),128)
            else:
                self.op("input_norms",[*refs,output],[int(local)],(3,1,1),256)
            inputs=[output.at(j*2304*2) for j in range(3)]
        elif not fused and not global_depth and not messages:
            shared={}
            for j,(s,name) in enumerate(zip(inputs,norms)):
                if (s,name) not in shared:
                    shared[s,name]=self.norm(s,"float" if i else "bf","bf",p+name+".weight",packed=self.rows==1 and local)
                inputs[j]=shared[s,name]
        # K/V and their histories always include the full prompt. Once the
        # final visit has those values, only its causal suffix feeds the
        # last hidden state and the three taps retained for the next token.
        final_prompt=self.rows>16 and self.visit+1==len(self.model.visits) and not local and not messages
        query_rows=16 if final_prompt else self.rows
        mlp_rows=8 if final_prompt else self.rows
        attn=self.attention(p+"attn.",inputs,local,query_rows,blend,depth)

        res=self.temp(mlp_rows*2304*4)
        if i%6==0:
            blocks=blocks+[norm]
            if self.rows==1:
                table=self.temp(48*4)
                queries,indices=self.bind(None),self.bind(None)
                self.fixed_depth[norm]=(table,set(),queries,indices,len(self.steps))
                self.op("fixed_depth_eight",[norm,queries,indices,table],[],(1,1,1),640)
        if self.rows==1:
            offset=(self.visit*HISTORY_WIDTH+6144)*2
            name = p+"attn.wo_ud.weight"
            paired = self.model.w.paired_high_bytes(name)
            if paired is None:
                kernel = "decode_output_residual"
                weight = self.weight(self.model.w.decode_projection(name))
            else:
                kernel = "paired_output_residual"
                weight = self.weight(paired)
            if i%6==0 and paired is not None and self.joint_groups:
                _,fixed,_,_,_,_=self.steps.pop()
                groups=self.joint_groups
                self.op(f"joint_output_{groups}",[attn,weight,x,self.w(p+"attn_sconv.weight"),
                        self.old.at(offset),self.new.at(offset),res,*fixed],[],(1,1,1),groups*32)
            else:
                self.op(kernel,[attn,weight,x,self.w(p+"attn_sconv.weight"),
                        self.old.at(offset),self.new.at(offset),res],[int(i%6!=0)],(288,1,1),256)
            output=self.temp(2304*(4 if messages else 2))
            refs=blocks+[self.null]*(4-len(blocks))+[
                self.null,self.null,self.null,self.null,self.null,
                self.weight(self.model.w.depth_query_streams(f"attn_res.mlp_queries.{i}")),self.w(p+"mlp_norm.weight"),res,output]+self.fixed_scores(blocks,2*i+1)
            config = len(blocks)+1 | (int(i%6!=0)<<3) | (int(not messages)<<4)
            if messages:
                inner=self.temp(64*2)
                self.prepared_inner[output]=inner
                refs.extend([self.weight(self.model.w.controller_stream_weights(p+"controllers.mlp.down_weight")),inner,self.pass_messages[messages[3]]])
                config|=128
            else:
                logits,published_ids=self.router_buffers(p+"mlp.",output)
                if published_ids is not None:
                    refs[4],refs[6],refs[7]=logits,self.w(p+"mlp.gate.bias"),published_ids
                    logits=self.null
                    config|=512
                refs.extend([self.weight(self.model.w.decode_router(p+"mlp.gate.weight")),logits,self.null])
                config|=64
            refs.extend([self.null,self.null])
            self.op("decode_depth_input",refs,[],(16 if messages else 8,1,1),640,config)
            if messages:
                output=self.controllers([p+"controllers.mlp."],[output],[messages[3]])[0]
                packed=self.temp(2304*2)
                logits,published_ids=self.router_buffers(p+"mlp.",packed)
                args=[output,self.w(p+"mlp_norm.weight"),self.blend_sums.pop(output),
                      self.weight(self.model.w.decode_router(p+"mlp.gate.weight")),logits,packed]
                kernel="blend_router"
                if published_ids is not None:
                    kernel="published_blend_router"
                    args.extend([self.w(p+"mlp.gate.bias"),published_ids])
                self.op(kernel,args,[],(16,1,1),256)
                output=packed
            return self.moe(p+"mlp.",output,res),blocks
        skip=self.rows-mlp_rows
        history=(self.visit*HISTORY_WIDTH+6144)*2
        self.op("prompt_attention_residual",
                [attn,self.old.at(history),self.new.at(history),self.w(p+"attn_sconv.weight"),
                 x.at(skip*2304*2),res], [query_rows,query_rows-mlp_rows,int(i%6!=0)],
                (9,mlp_rows,1),64)
        mlp_query=self.w(f"attn_res.mlp_queries.{i}")
        mixed=self.mix([b.at(skip*2304*4) for b in blocks]+[self.norm(res,"float","float",rows=mlp_rows)],mlp_query,rows=mlp_rows)
        if messages:
            act=self.message(p+"controllers.mlp.",mixed,messages[3],p+"mlp_norm.weight")
        else:
            act=self.norm(mixed,"float","bf",p+"mlp_norm.weight",rows=mlp_rows)
        expert=self.moe(p+"mlp.",act,rows=mlp_rows)
        out=self.temp(self.rows*2304*2)
        history=(self.visit*HISTORY_WIDTH+13056)*2
        self.op("prompt_expert_residual",
                [expert,self.old.at(history),self.new.at(history),self.w(p+"mlp_sconv.weight"),
                 res,out.at(skip*2304*2)], [mlp_rows,0,1], (9,mlp_rows,1),64)
        return out,blocks

    def discard_prediction(self):
        if self.pending:
            prediction, self.pending = self.pending, None
            prediction.wait()

    def execute_with_prediction(self, cache, history, fresh, confirmed):
        # State and embeddings have distinct bindings so CPU preparation never
        # races the command that produces the current logits.
        lib = self.gpu.lib
        skip = self.predicted_dispatch_count if confirmed else 0
        begin = C.cast(C.byref(self.with_greedy, skip*C.sizeof(Dispatch)), C.POINTER(Dispatch))
        current = lib.ink_submit(self.gpu.handle,begin,len(self.with_greedy)-skip,self.handles)
        if not current:
            raise RuntimeError(self.gpu.error())
        try:
            self.future_state, self.alternate_state = self.alternate_state, self.future_state
            c = cache[0]
            dest = history if not c.shared_history else Buffer(self.gpu,history.shape,history.dtype)
            refs = (PTR * len(self.handles)).from_buffer_copy(self.handles)
            refs[self.input_ref.index] = self.future_input.handle
            refs[self.state_ref.index] = self.future_state.handle
            refs[self.old.index],refs[self.new.index] = fresh.handle,dest.handle
            rows=self.future_state.array[:self.predicted_visits]
            rows[:]=self.state.array[:self.predicted_visits]
            for field in (0,1,3,6):
                rows[:,field]+=1
            window=rows[:,4]
            visible=np.where(window,np.minimum(rows[:,3],window+62),rows[:,3])
            pitch=(visible+31)//32*32
            rows[:,10]=pitch
            widths=np.where(window,pitch//32,(pitch+63)//64).tolist()
            for step,width in zip(self.score_steps,widths):
                self.prediction_ops[step].group[0]=width
                self.prediction_ops[step+1].pipeline=self.ops[step+1].pipeline
                self.prediction_ops[step+1].threads=self.ops[step+1].threads
            future = lib.ink_submit(self.gpu.handle,self.prediction_ops,len(self.prediction_ops),refs)
            if not future:
                raise RuntimeError(self.gpu.error())
            self.pending = DecodePrediction(self,future,cache,fresh,dest)
        finally:
            failed = lib.ink_wait(current)
            if confirmed:
                confirmed.wait()
            if failed:
                self.discard_prediction()
                raise RuntimeError(self.gpu.error())

    def run(self,tokens,cache,image_rows):
        if len(cache) != len(self.cache_bindings):
            raise ValueError("Expected one cache per model visit")
        previous, self.pending = self.pending, None
        hit = False
        if previous:
            hit = previous.valid(cache) and not image_rows and int(tokens[0]) == self.predicted_token.array[0]
            if hit:
                previous.release_row()
            else:
                previous.wait()
        if not hit:
            if self.gate_publication is not None:
                self.gate_publication.array[-1] = 0
            self.input.array[:] = self.model.w.tensors["embed.weight"][tokens]
            self.image_mask.array.fill(0)
            if image_rows:
                for row, value in image_rows:
                    self.input.array[row] = to_bits(value)
                    self.image_mask.array[row] = 1
        history=cache[0].graph_history
        if history is None:
            history=Buffer(self.gpu,(48,HISTORY_WIDTH),np.uint16)
            history.array.fill(0)
        fresh=previous.result if previous else cache[0].graph_spare
        if fresh is None:
            fresh=Buffer(self.gpu,(48,HISTORY_WIDTH),np.uint16)
        self.set("history",history)
        self.set("next",fresh)
        max_pitch = ((cache[0].position + self.rows + 31) // 32) * 32
        needed = self.rows * 24 * max_pitch
        if self.score_buffer is None or self.score_buffer.shape[0] < needed:
            capacity = needed if self.rows > 1 else 24 * ((max_pitch + 255) // 256 * 256)
            self.score_buffer = Buffer(self.gpu, (capacity,), np.uint16)
            for j in range(len(cache)):
                self.set((j,"scores"), self.score_buffer)
        if self.rows == 1:
            layout = tuple(c.window == 1024 for c in cache)
            if layout != self.window_layout:
                for step, standard_window in zip(self.score_steps, layout):
                    softmax = self.ops[step + 1]
                    softmax.pipeline = self.window_softmax if standard_window else self.general_softmax
                    softmax.threads = 512 if standard_window else 256
                self.window_layout = layout
        copies=[]
        # Visits with the same retained prefix share attention dimensions.
        # A fork or a window rollover can split these groups independently.
        groups={}
        for j,(c,(key_slot,value_slot,grid)) in enumerate(zip(cache,self.cache_bindings)):
            past=c.reserve(self.rows,copies)
            self.handles[key_slot]=c.keys.handle
            self.handles[value_slot]=c.values.handle
            fp8=c.dtype==np.uint8
            key=(c.position,past,c.window,fp8)
            group=groups.get(key)
            if group is None:
                length=past+self.rows
                visible=min(length,c.window+62) if c.window else length
                pitch=(visible+31)//32*32
                width=(pitch+63)//64 if self.rows==1 and not c.window else pitch//32
                state=(c.position,past,self.rows,length,c.window,fp8,past,c.window,fp8,1024,pitch,0)
                group=groups[key]=([],width,state)
            visits,width,_=group
            visits.append(j)
            grid[0]=width
        for visits,_,state in groups.values():
            self.state.array[visits]=state
        if self.query_tail is not None:
            tail=self.query_tail.array
            tail[:]=self.state.array[-1]
            tail[0]+=self.rows-16
            tail[1]+=self.rows-16
            tail[2]=16
        if copies:
            self.gpu.copy_cache(copies)
        c = cache[0]
        eligible = (self.predict and c.predict_next and not image_rows and (previous is None or hit)
                    and all(v.length+2 <= v.keys.shape[0] for v in cache[:self.predicted_visits])
                    and 24*((c.position+2+31)//32*32) <= self.score_buffer.shape[0])
        if eligible or hit:
            if eligible:
                self.execute_with_prediction(cache,history,fresh,previous if hit else None)
            else:
                previous.wait()
                skip=self.predicted_dispatch_count
                tail=C.cast(C.byref(self.ops,skip*C.sizeof(Dispatch)),C.POINTER(Dispatch))
                if self.gpu.lib.ink_execute(self.gpu.handle,tail,len(self.ops)-skip,self.handles):
                    raise RuntimeError(self.gpu.error())
        else:
            self.execute()
        for c in cache:
            c.position+=self.rows
            c.length+=self.rows
        cache[0].graph_spare=None if self.pending or cache[0].shared_history else history
        cache[0].graph_history=fresh
        cache[0].shared_history=False
        if self.output:
            return self.output.array[:self.model.vocab].copy()
