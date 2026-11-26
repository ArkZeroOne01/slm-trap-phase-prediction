import torch,time,json,sys,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
from cnn import PaperCNN
from optics import to_mask
torch.set_num_threads(8);torch.manual_seed(123)
m=PaperCNN().cuda().eval()
for b in m.blocks:
 for i in [0,3]:b.branch[i]=torch.nn.utils.fusion.fuse_conv_bn_eval(b.branch[i],b.branch[i+1]);b.branch[i+1]=torch.nn.Identity()
cpu=torch.zeros(21,1,2,1024,1024,pin_memory=True);cpu[:,:,0,::20,::20]=1
inputs=[cpu[t].cuda() for t in range(21)];dest=[torch.empty(1,1024,1024,pin_memory=True) for _ in range(21)]
records=[]
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
 def eager(transfer=False):
  state=None;ts=[];out=[];start=time.perf_counter()
  for t in range(21):
   if transfer:inputs[t].copy_(cpu[t],non_blocking=True)
   aa,pp=m(inputs[t]);mask=to_mask(aa,pp);out.append(mask)
   if transfer:
    dest[t].copy_(mask,non_blocking=True);torch.cuda.current_stream().synchronize();ts.append((time.perf_counter()-start)*1000)
  return out,ts
 for _ in range(3):eager()
 torch.cuda.synchronize()
 # Warmup on capture stream, ensuring all kernels and FFT plans initialized.
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  for _ in range(2):eager()
 torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
 graphs=[];masks=[];state=None;started=time.perf_counter()
 for t in range(21):
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g,stream=stream):
   aa,pp=m(inputs[t]);newstate=None;mask=to_mask(aa,pp)
  state=newstate;graphs.append(g);masks.append(mask)
 torch.cuda.synchronize();capture_ms=(time.perf_counter()-started)*1000
 def replay(transfer=False):
  ts=[];start=time.perf_counter()
  for t,g in enumerate(graphs):
   if transfer:inputs[t].copy_(cpu[t],non_blocking=True)
   g.replay()
   if transfer:
    dest[t].copy_(masks[t],non_blocking=True);torch.cuda.current_stream().synchronize();ts.append((time.perf_counter()-start)*1000)
  return masks,ts
 ref,_=eager();torch.cuda.synchronize();reference=[x.clone() for x in ref];replay();torch.cuda.synchronize();error=max((x-y).abs().max().item() for x,y in zip(reference,masks))
 for scope,transfer in [('gpu_resident',False),('cpu_to_cpu_blocking_per_frame',True)]:
  for mode,fn in [('eager',eager),('cuda_graph',replay)]:
   for _ in range(2):fn(transfer)
   seq=[];arr=[]
   for _ in range(10):
    torch.cuda.synchronize();start=time.perf_counter();_,ts=fn(transfer);torch.cuda.synchronize();seq.append((time.perf_counter()-start)*1000)
    if transfer:arr.append(ts)
   row=dict(scope=scope,mode=mode,sequence_ms=statistics.mean(seq),per_frame_ms=statistics.mean(seq)/21,median_sequence_ms=statistics.median(seq),runs_ms=seq,mean_arrivals_ms=[statistics.mean(x[t] for x in arr) for t in range(21)] if transfer else [])
   records.append(row);print(json.dumps(row),flush=True)
 result=dict(model='Pan/Lu backbone CNN, random seed123, no checkpoint',size=1024,frames=21,gpu=torch.cuda.get_device_name(),precision='BF16+FP32 optics,BN fused',graph_count=21,capture_ms=capture_ms,max_phase_absolute_error=error,peak_memory_mib=torch.cuda.max_memory_allocated()/2**20,pinned_cpu_buffers=True,records=records)
 (R/'results.json').write_text(json.dumps(result,indent=2));print('capture_ms',capture_ms,'max_error',error,'peak_memory_mib',result['peak_memory_mib'],flush=True)
