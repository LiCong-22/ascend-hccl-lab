"""Verify and time a single-node HCCL AllReduce on at least two Ascend NPUs."""

import argparse
from datetime import timedelta
import json
from math import ceil
import os
from statistics import median
from time import perf_counter_ns


def message_size(value):
    size = int(value)
    if size <= 0 or size % 4:
        raise argparse.ArgumentTypeError("message size must be a positive multiple of 4")
    return size


def measure(torch, dist, device, rank, world_size, size, warmup, iterations):
    tensor = torch.empty(size // 4, dtype=torch.float32, device=device)
    expected = world_size * (world_size + 1) // 2
    timings = []

    for step in range(warmup + iterations):
        tensor.fill_(rank + 1)
        torch.npu.synchronize()
        begin = perf_counter_ns()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.npu.synchronize()
        if step >= warmup:
            timings.append((perf_counter_ns() - begin) / 1000)

    # Every rank validates the result before continuing to the next message size.
    valid = (tensor == expected).all().to(dtype=torch.int32).reshape(1)
    dist.all_reduce(valid, op=dist.ReduceOp.MIN)
    if valid.item() != 1:
        raise RuntimeError(f"AllReduce validation failed for {size} bytes per rank")

    # Report the slowest rank for each repetition. This reduction is outside timing.
    group_timings = torch.tensor(timings, dtype=torch.float32, device=device)
    dist.all_reduce(group_timings, op=dist.ReduceOp.MAX)
    ordered = sorted(group_timings.cpu().tolist())
    p50 = median(ordered)
    print_result = {
        "operation": "all_reduce_sum",
        "ranks": world_size,
        "bytes_per_rank": size,
        "iterations": iterations,
        "validated": True,
        "host_latency_p50_us": round(p50, 3),
        "host_latency_p95_us": round(ordered[ceil(0.95 * iterations) - 1], 3),
        "algorithm_bandwidth_gb_s": round(size / (p50 * 1000), 6),
    }
    if rank == 0:
        print(json.dumps(print_result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=message_size, nargs="+", default=[4096, 65536, 1048576])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 1:
        parser.error("--warmup and --iterations must be positive")

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_size = int(os.environ["LOCAL_WORLD_SIZE"])
    except (KeyError, ValueError):
        parser.error("launch with torchrun --standalone --nproc-per-node=2")
    if world_size < 2 or world_size != local_size:
        parser.error("use a single host with at least two NPU ranks")

    import torch
    import torch.distributed as dist
    import torch_npu  # Registers the NPU device and HCCL backend.

    device_count = torch_npu.npu.device_count()
    if not 0 <= local_rank < local_size <= device_count:
        parser.error(f"{local_size} ranks requested but {device_count} NPU devices are visible")

    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl", timeout=timedelta(seconds=120))
    try:
        for size in args.sizes:
            measure(torch, dist, device, dist.get_rank(), world_size, size,
                    args.warmup, args.iterations)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
