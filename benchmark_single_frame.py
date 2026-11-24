"""Measure one causal Mamba prediction step with the model already loaded."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import torch

from encoding import encode, to_mask
from model import CNNFieldCausalTransformer

ROOT = Path(__file__).resolve().parent


def load_model(size, weights_path, device, seed):
    torch.manual_seed(seed)
    model = CNNFieldCausalTransformer(
        size=size, patch=8, width=32, heads=4, memory_grid=4, history_frames=20
    )
    if size == 256:
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
    return model.to(device).eval()


def load_frames(path, last_frame, size, device):
    with np.load(path, allow_pickle=False) as data:
        coordinates = data["q_xy"].astype(np.float32)
        phases = data["trap_phase"].astype(np.float32)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (96, 2):
        raise ValueError("Expected q_xy with shape [T, 96, 2]")
    if phases.shape != coordinates.shape[:2]:
        raise ValueError("Expected trap_phase with shape [T, 96]")
    if last_frame >= len(coordinates):
        raise ValueError("Requested frame is outside the input sequence")
    if not np.isfinite(coordinates).all() or not np.isfinite(phases).all():
        raise ValueError("Inputs must be finite")
    return encode(
        torch.from_numpy(coordinates[:last_frame + 1]).to(device),
        torch.from_numpy(phases[:last_frame + 1]).to(device),
        size,
    )


def step(model, frame, state):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        (amplitude, phase), next_state = model.step(frame, state)
    slm_phase = to_mask(amplitude, phase).remainder(2 * torch.pi)
    return slm_phase, next_state


@torch.inference_mode()
def benchmark(size, args, use_graph):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required")
    device = torch.device("cuda")
    model = load_model(size, args.weights, device, args.seed)
    frames = load_frames(args.input, args.frame_index, size, device)
    state = None
    for index in range(args.frame_index):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, state = model.step(frames[index:index + 1], state)
    frame = frames[args.frame_index:args.frame_index + 1]

    graph = None
    if use_graph:
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(args.warmup):
                step(model, frame, state)
        torch.cuda.current_stream().wait_stream(warmup_stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output, _ = step(model, frame, state)
        graph.replay()
    else:
        for _ in range(args.warmup):
            step(model, frame, state)
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    # CUDA events initialize lazily. Prime them outside the timing loop.
    start.record()
    stop.record()
    torch.cuda.synchronize()
    event_ms, wall_ms = [], []
    for _ in range(args.runs):
        began = perf_counter()
        start.record()
        if graph is None:
            output, _ = step(model, frame, state)
        else:
            graph.replay()
        stop.record()
        stop.synchronize()
        wall_ms.append((perf_counter() - began) * 1000)
        event_ms.append(start.elapsed_time(stop))
    if args.output is not None:
        with args.output.open("xb") as stream:
            np.savez_compressed(stream, slm_phase=output[0].cpu().numpy())
    print(json.dumps({
        "gpu": torch.cuda.get_device_name(),
        "grid_size": size,
        "mode": "cuda_graph" if use_graph else "eager",
        "weights": str(args.weights) if size == 256 else "random",
        "random_seed": args.seed if size != 256 else None,
        "frame_index": args.frame_index,
        "history_frames": args.frame_index,
        "warmup": args.warmup,
        "runs": args.runs,
        "cuda_event_median_ms": median(event_ms),
        "synchronized_wall_median_ms": median(wall_ms),
    }, indent=2))


def main(size, use_graph=False):
    parser = argparse.ArgumentParser(description=f"Benchmark one {size}x{size} Mamba frame")
    parser.add_argument("--input", type=Path, default=ROOT / "example_input.npz")
    if size == 256:
        parser.add_argument("--weights", type=Path, default=ROOT / "weights.pt")
    else:
        parser.set_defaults(weights=None)
        parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--frame-index", type=int, choices=range(21), default=0,
                        metavar="INDEX", help="zero-based frame index, 0–20")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--output", type=Path, help="optional output NPZ path")
    parser.set_defaults(seed=20261124)
    args = parser.parse_args()
    if args.warmup < 1 or args.runs < 1:
        parser.error("--warmup and --runs must be positive")
    benchmark(size, args, use_graph)
