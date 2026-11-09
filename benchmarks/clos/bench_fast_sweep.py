"""One SHMEM initialization, existing peer schedules, performance only.

Launch on each node with torchrun. --plan-only needs no torch/NPU installation.
No numerical validation is performed; every report explicitly records that.
"""
import argparse
import csv
import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import socket
import statistics
import time
import zlib

ORDERS = ("rank", "group", "group_staggered", "group_round_robin", "group_window")
SHAPES = ((8192, 2816, 8, 32), (16384, 7168, 6, 64))


def candidates(ep, aiv, weights=range(1, 9), visits=(4, 8, 16, 32)):
    result = []
    for order in ORDERS:
        grouped = order in ORDERS[3:]
        for peers, (workers, qps), visit in itertools.product(
            (1, 2, 4) if grouped else (1, 2, 4, 8), ((1, 1), (1, 2), (2, 1)), visits
        ):
            owners = math.ceil((8 if grouped else ep - 1) / peers)
            if owners * workers > aiv or owners * workers * qps > 32:
                continue
            if grouped and peers * (ep // 8) > 64:
                continue
            for weight in weights if workers == 2 else (1,):
                item = dict(order=order, peers=peers, workers=workers, qps=qps, visit=visit, weight=weight)
                item["id"] = f"{order}_p{peers}_w{workers}_q{qps}_v{visit}_r{weight}"
                item["payload_aivs"] = owners * workers
                result.append(item)
    return result


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ep-size", type=int, choices=(64, 128), default=128)
    p.add_argument("--output", type=Path, default=Path("fast_sweep"))
    p.add_argument("--control-port", type=int, default=29610)
    p.add_argument("--shmem-port", type=int, default=29620)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--seed", type=int, default=20260823)
    p.add_argument("--experts", type=int, default=256)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--final-samples", type=int, default=30)
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--weights", type=int, nargs="+", default=list(range(1, 9)))
    p.add_argument("--visits", type=int, nargs="+", default=[4, 8, 16, 32])
    p.add_argument("--shape", choices=("all", "2816", "7168"), default="all")
    p.add_argument("--smoke", action="store_true", help="Small strategy subset, same full shapes")
    p.add_argument("--plan-only", action="store_true")
    args = p.parse_args()
    if args.experts % args.ep_size or args.experts < 8 or args.warmup < 0 or min(args.samples, args.final_samples, args.top) < 1:
        p.error("invalid experts or sampling parameters")
    if not set(args.weights) <= set(range(1, 9)) or not set(args.visits) <= {4, 8, 16, 32}:
        p.error("weights must be 1..8; visits must be 4/8/16/32")
    return args


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf8")
    temporary.replace(path)


def pack(value):
    if isinstance(value, tuple):
        return {"__tuple__": [pack(x) for x in value]}
    if isinstance(value, list):
        return [pack(x) for x in value]
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    return value


def unpack(value):
    if isinstance(value, dict):
        if set(value) == {"__tuple__"}:
            return tuple(unpack(x) for x in value["__tuple__"])
        return {k: unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack(x) for x in value]
    return value


class StoreGroup:
    """Small JSON control collectives over TCPStore, with no Gloo group."""
    def __init__(self, store, rank, size):
        self.store, self._rank, self._size, self.seq = store, rank, size, 0

    def rank(self):
        return self._rank

    def size(self):
        return self._size

    def gather(self, obj):
        key = str(self.seq)
        self.seq += 1
        encode = lambda value: zlib.compress(json.dumps(value, separators=(',', ':')).encode(), 1)
        decode = lambda value: json.loads(zlib.decompress(value))
        self.store.set(f"{key}/{self._rank}", encode(pack(obj)))
        if self._rank == 0:
            rows = [decode(self.store.get(f"{key}/{i}")) for i in range(self._size)]
            self.store.set(key + "/all", encode(rows))
        result = unpack(decode(self.store.get(key + "/all")))
        self.store.set(f"{key}/read/{self._rank}", "1")
        if self._rank == 0:
            for i in range(self._size):
                self.store.get(f"{key}/read/{i}")
        return result

    def agree(self, obj):
        # Plans can contain hundreds of configurations. Compare canonical
        # hashes rather than broadcasting EP identical full plans to everyone.
        digest = hashlib.sha256(json.dumps(pack(obj), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        rows = self.gather(digest)
        if any(row != rows[0] for row in rows):
            raise RuntimeError("rank configuration mismatch")


def configure(buffer, config):
    from deep_ep.backends.clos.config import Config
    order = ORDERS.index(config["order"])
    group_peers = config["peers"] if order >= 3 else 0
    wanted = [order, order, group_peers, config["workers"], config["qps"], config["weight"]]
    actual = list(buffer.runtime._configure_tuning(*wanted))
    if actual != wanted:
        raise RuntimeError(f"native tuning mismatch: {actual} != {wanted}")
    buffer._udma_config = Config(udma_peers_per_worker=config["peers"], udma_wqes_per_visit=config["visit"])
    buffer._udma_dispatch_peer_order = buffer._udma_combine_peer_order = config["order"]
    buffer._udma_peers_per_group_per_worker = group_peers or None
    buffer._udma_workers_per_peer, buffer._udma_qps_per_worker = config["workers"], config["qps"]
    buffer._udma_cross_lane0_weight = config["weight"]


def parse_capture(directory, op, configs, warmup, samples):
    paths = list(directory.rglob("kernel_details.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one profiler CSV, found {len(paths)}")
    with paths[0].open(encoding="utf-8-sig", newline="") as f:
        rows = sorted(csv.DictReader(f), key=lambda x: float(x["Start Time(us)"].strip()))
    # buffer.barrier() and pre-combine notify share a kernel name. Match the
    # complete ordered call, so the outer barrier is never charged to notify.
    names = (["cached_notify_combine_kernel", "cached_notify_dispatch_kernel", "v1_dispatch_udma_kernel"]
             if op == "cached_dispatch" else
             ["cached_notify_combine_kernel", "cached_notify_combine_kernel",
              "v1_combine_udma_comm_kernel", "v1_combine_udma_reduce_kernel"])
    width = warmup + samples + 2
    required = len(configs) * width + 4  # global leading/trailing guards
    selected = [r for r in rows if r["Name"] in names]
    if len(selected) != required * len(names):
        raise RuntimeError(f"incomplete capture: {len(selected)} stage records; expected {required * len(names)}")
    sequence = [r["Name"] for r in selected]
    if sequence != names * required:
        raise RuntimeError("kernel order differs from expected barrier/notify/communication/reduce sequence")
    stages = [[float(r["Duration(us)"]) for r in selected[i::len(names)]] for i in range(len(names))]
    result = []
    for index, config in enumerate(configs):
        start = 2 + index * width + warmup
        values = [v[start:start + samples] for v in stages]
        if any(not math.isfinite(x) or x <= 0 for v in values for x in v):
            raise RuntimeError("invalid kernel durations")
        result.append(dict(config=config, samples=samples, mean_us=statistics.mean(values[2]),
                           barrier_us=statistics.mean(values[0]),notify_us=statistics.mean(values[1]),
                           reduce_us=statistics.mean(values[3]) if len(names) == 4 else 0,
                           communication_samples_us=values[2]))
    return result


def measure(torch, torch_npu, buffer, group, args, directory, op, configs, invoke, logical_bytes, samples):
    started = time.monotonic()
    directory.mkdir(parents=True, exist_ok=True)
    group.agree([op, configs, samples])
    if group.rank() == 0:
        print(f"PHASE_BEGIN {directory.parent.name}/{directory.name} configurations={len(configs)} samples={samples}", flush=True)
    write(directory / "manifest.json", dict(op=op,configs=configs,samples=samples,warmup=args.warmup,
                                              precision_checked=False,logical_bytes=logical_bytes))
    def call():
        buffer.barrier(with_cpu_sync=True)
        result = invoke()
        result[-1].current_stream_wait()
        torch.npu.synchronize()
    # Start/stop the profiler once for the entire strategy list, not per strategy.
    profiler = torch_npu.profiler
    with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
                          on_trace_ready=profiler.tensorboard_trace_handler(str(directory)),
                          record_shapes=False, profile_memory=False, with_stack=False,
                          experimental_config=profiler._ExperimentalConfig(
                              export_type=[profiler.ExportType.Text],
                              profiler_level=profiler.ProfilerLevel.Level0,
                              aic_metrics=profiler.AiCMetrics.AiCoreNone)):
        configure(buffer, configs[0])
        for _ in range(2):call()
        for index, config in enumerate(configs):
            configure(buffer, config)
            for _ in range(args.warmup + samples + 2):call()
            if group.rank() == 0 and (index % 20 == 0 or index == len(configs)-1):
                print(f"SWEEP {directory.parent.name}/{op} {index+1}/{len(configs)} {config['id']}", flush=True)
        for _ in range(2):call()
    local = None
    error = None
    try:
        local = parse_capture(directory, op, configs, args.warmup, samples)
        write(directory / "samples.json", local)
    except Exception as exc:
        error = str(exc)
    errors = group.gather(error)
    if any(errors):
        raise RuntimeError(f"profiler parsing failed: {[(i,e) for i,e in enumerate(errors) if e]}")
    gathered = group.gather(dict(bytes=logical_bytes, rows=[
        {k:r[k] for k in ('mean_us','notify_us','reduce_us')} for r in local]))
    result = []
    for index, config in enumerate(configs):
        rows = [r["rows"][index] for r in gathered]
        times = [r["mean_us"] for r in rows]
        bandwidths = [g["bytes"] / r["mean_us"] / 1000 for g,r in zip(gathered,rows)]
        result.append(dict(config=config, GBps=sum(g['bytes'] for g in gathered)/sum(times)/1000,
                           GBps_range=[min(bandwidths),max(bandwidths)],
                           mean_us=statistics.mean(times),time_range_us=[min(times),max(times)],
                           notify_us_range=[min(r['notify_us'] for r in rows),max(r['notify_us'] for r in rows)],
                           reduce_us_range=[min(r['reduce_us'] for r in rows),max(r['reduce_us'] for r in rows)],
                           samples_per_rank=samples,precision_checked=False))
    result.sort(key=lambda r:r['GBps'],reverse=True)
    if group.rank() == 0:
        write(directory / 'ranking.json', result)
        print(f"PHASE_DONE {directory.parent.name}/{directory.name} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    return result


def main():
    args = parse_args()
    shapes = [dict(tokens=t,hidden=h,topk=k,aiv=a) for t,h,k,a in SHAPES if args.shape in ('all',str(h))]
    plans = []
    for shape in shapes:
        configs = candidates(args.ep_size,shape['aiv'],args.weights,args.visits)
        if args.smoke:
            configs = [next(c for c in configs if c['order']==o and c['workers']==w and c['qps']==q)
                       for o,w,q in (('rank',1,1),('group',1,2),('group_staggered',2,1),
                                      ('group_round_robin',2,1),('group_window',2,1))]
        plans.append(dict(shape=shape,configs=configs))
    if args.plan_only:
        print(json.dumps(dict(ep=args.ep_size,plans=plans),indent=2));return
    import torch
    import torch_npu
    import torch.distributed as dist
    from deep_ep.backends.clos.elastic import ElasticBuffer
    from deep_ep.backends.clos import buffer as buffer_module, elastic as elastic_module
    rank,world,device = (int(os.environ[k]) for k in ('RANK','WORLD_SIZE','LOCAL_RANK'))
    if world != args.ep_size or int(os.environ.get('LOCAL_WORLD_SIZE','8')) != 8:
        raise RuntimeError('requires eight contiguous ranks per server and matching EP')
    torch.npu.set_device(device)
    store=dist.TCPStore(os.environ['MASTER_ADDR'],args.control_port,world,rank==0,
                        timeout=datetime.timedelta(seconds=args.timeout),wait_for_workers=True)
    group=StoreGroup(store,rank,world)
    # Narrow benchmark adapters: keep all bootstrap/consensus checks, replacing
    # only their transport with TCPStore; production distributed APIs are untouched.
    buffer_module._bootstrap_all_gather=lambda g,c,n,value: group.gather(value)
    elastic_module._validate_group_consensus=lambda g,name,value: group.agree([name,value])
    hosts=group.gather(socket.gethostname())
    chunks=[hosts[i:i+8] for i in range(0,world,8)]
    if any(len(set(c))!=1 for c in chunks) or len({c[0] for c in chunks})!=world//8:
        raise RuntimeError('rank ordering does not match physical servers')
    os.environ['DEEPEP_CLOS_TUNING_SESSION']='1'
    os.environ['DEEPEP_CLOS_COMBINE_PRE_NOTIFY']='1'
    for name,value in os.environ.items():
        if name.startswith('DEEPEP_CLOS_') and value not in ('','0') and name not in (
            'DEEPEP_CLOS_TUNING_SESSION','DEEPEP_CLOS_COMBINE_PRE_NOTIFY'):
            raise RuntimeError(f'conflicting experimental environment: {name}')
    group.agree(plans)
    import importlib.util
    native = Path(importlib.util.find_spec('deep_ep._clos_C').origin)
    identity_paths = [Path(__file__), Path(elastic_module.__file__), native,
                      native.with_name('libdeepep_clos_kernels.so'),
                      Path(os.environ['SHMEM_ROOT'])/'lib/libshmem.so']
    identity = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in identity_paths}
    group.agree(identity)
    capacity=max(s['tokens'] for s in shapes)
    workspace=max(ElasticBuffer.get_buffer_size_hint(group,capacity,s['hidden'],s['topk']) for s in shapes)
    out=args.output/f'rank{rank}'
    out.mkdir(parents=True,exist_ok=False)
    write(out/'identity.json',identity)
    write(out/'plan.json',dict(ep=world,experts=args.experts,seed=args.seed,plans=plans,workspace_bytes=workspace,
                              huge_pages=True,flush_l2=False,pre_combine_notify=True,precision_checked=False))
    if rank==0:print(f'BUFFER_INIT bytes={workspace} cases={[len(p["configs"]) for p in plans]}',flush=True)
    init_started = time.monotonic()
    buffer=ElasticBuffer(group,num_bytes=workspace,num_max_tokens_per_rank=capacity,hidden=0,num_topk=0,
                         explicitly_destroy=True,use_1g_huge_pages=True,
                         shmem_ip_port=f'tcp://{os.environ["MASTER_ADDR"]}:{args.shmem_port}',
                         transport='udma',udma_auto_policy=False,udma_schedule='peer',udma_dispatch_peer_order='rank',
                         udma_combine_peer_order='rank',udma_ranks_per_group=8,
                         udma_workers_per_peer=1,udma_qps_per_worker=1,
                         udma_peers_per_worker=8,udma_wqes_per_visit=8)
    init_seconds = time.monotonic() - init_started
    if rank == 0:
        print(f'BUFFER_READY initializations=1 elapsed_s={init_seconds:.1f}', flush=True)
    result=[]
    try:
        for plan in plans:
            shape,configs=plan['shape'],plan['configs']
            t,h,k,a=(shape[n] for n in ('tokens','hidden','topk','aiv'))
            tag=f't{t}_h{h}_k{k}_aiv{a}'
            gen=torch.Generator().manual_seed(args.seed+rank)
            routes=torch.rand((t,args.experts),generator=gen).topk(k,dim=1).indices
            ranked=(routes//(args.experts//world)).sort(dim=1).values
            send_rows=int((ranked[:,1:]!=ranked[:,:-1]).sum())+t
            x=torch.randn((t,h),generator=gen,dtype=torch.bfloat16).npu()
            idx=routes.npu()
            configure(buffer,configs[0])
            fresh=buffer.dispatch(x,topk_idx=idx,num_experts=args.experts,num_sms=a,do_expand=False,
                                  async_with_compute_stream=True)
            fresh[-1].current_stream_wait();torch.npu.synchronize()
            recv,_,_,handle,_=fresh
            owned=recv.clone();torch.npu.synchronize()
            invocations={'cached_dispatch':lambda:buffer.dispatch(x,handle=handle,num_sms=a,do_expand=False,
                                                                  async_with_compute_stream=True),
                         'combine_unweighted':lambda:buffer.combine(owned,handle,num_sms=a,async_with_compute_stream=True)}
            item=dict(shape=shape,operations={})
            for op,invoke in invocations.items():
                logical_bytes=(handle.num_recv_tokens if op=='cached_dispatch' else send_rows)*h*2
                coarse=measure(torch,torch_npu,buffer,group,args,out/tag/(op+'_coarse'),op,configs,invoke,logical_bytes,args.samples)
                finalists=[r['config'] for r in coarse[:args.top]]
                rounds=[]
                for turn,order in enumerate((finalists,list(reversed(finalists)))):
                    rounds.append(measure(torch,torch_npu,buffer,group,args,out/tag/f'{op}_final{turn}',op,
                                          order,invoke,logical_bytes,args.final_samples))
                final=[]
                for config in finalists:
                    rows=[next(r for r in round_ if r['config']['id']==config['id']) for round_ in rounds]
                    final.append(dict(config=config,score_GBps=min(r['GBps'] for r in rows),repeats=rows))
                final.sort(key=lambda r:r['score_GBps'],reverse=True)
                item['operations'][op]=dict(winner=final[0],finalists=final,coarse=coarse)
                if rank==0:print('WINNER',tag,op,json.dumps(final[0]),flush=True)
            result.append(item)
            write(out/'results.json',dict(ep=world,buffer_initializations=1,precision_checked=False,results=result))
            del invocations,fresh,recv,handle,owned,x,idx
            torch.npu.synchronize();torch.npu.empty_cache()
    finally:
        buffer.destroy()
        torch.npu.synchronize()
    group.agree('complete')
    write(out/'exit.json',dict(code=0,buffer_initializations=1,initialization_seconds=init_seconds,
                               precision_checked=False,complete=True))
    if rank==0:
        write(args.output/'report.json',dict(ep=world,experts=args.experts,dtype='bf16',expand=False,
            huge_pages=True,flush_l2=False,pre_combine_notify=True,seed=args.seed,
            bandwidth_formula='sum(rank payload bytes including self) / sum(rank mean communication us) / 1000',
            identity=identity,buffer_initializations=1,precision_checked=False,results=result))
        print('FAST_SWEEP_COMPLETE',flush=True)


if __name__=='__main__':
    main()
