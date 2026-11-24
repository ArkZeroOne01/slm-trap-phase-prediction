"""Predict one SLM phase sequence from coordinates and input phases."""
import argparse
from pathlib import Path

import numpy as np
import torch

from encoding import encode, to_mask
from model import CNNFieldCausalTransformer

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------- caches
# The model and its CUDA graph only depend on (weights file, device, sequence
# length), so keep them alive between calls instead of rebuilding every time.
_MODEL_CACHE = {}
_GRAPH_CACHE = {}


def _get_model(weights_path, device):
    key = (str(Path(weights_path).resolve()), str(device))
    model = _MODEL_CACHE.get(key)
    if model is None:
        model = CNNFieldCausalTransformer(
            size=256, patch=8, width=32, heads=4, memory_grid=4, history_frames=20
        ).to(device).eval()
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        _MODEL_CACHE[key] = model
    return key, model


def _get_graph(key, model, static_inputs, warmup=3):
    """Capture model(static_inputs) into a CUDA graph (cached per weights file).

    Requirements (all satisfied here): fixed shapes, no data-dependent control flow,
    eval mode, and a warmup so cuFFT plans and allocator blocks exist before capture.
    """
    cached = _GRAPH_CACHE.get(key)
    if cached is not None:
        return cached
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(warmup):
                model(static_inputs)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.autocast("cuda", dtype=torch.bfloat16), torch.cuda.graph(graph):
        outputs = model(static_inputs)
    entry = (graph, outputs)
    _GRAPH_CACHE[key] = entry
    return entry

@torch.inference_mode()
def predict(input_path, weights_path, cuda_graph=False):
    with np.load(input_path, allow_pickle=False) as source:
        q = source["q_xy"].astype(np.float32)
        phase = source["trap_phase"].astype(np.float32)
    if q.shape != (21, 96, 2) or phase.shape != (21, 96):
        raise ValueError("Expected q_xy [21, 96, 2] and trap_phase [21, 96]")
    if not np.isfinite(q).all() or not np.isfinite(phase).all():
        raise ValueError("Inputs must be finite")
    device = torch.device("cuda")
    key, model = _get_model(weights_path, device)
    coordinates = torch.from_numpy(q).to(device)
    phase_input = torch.from_numpy(phase).to(device)
    inputs = encode(coordinates, phase_input, 256).unsqueeze(0)

    if not cuda_graph:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            amplitude, focal_phase = model(inputs)
    else:
        static_inputs = _GRAPH_CACHE.get(key + ("buf",))
        if static_inputs is None:
            static_inputs = torch.zeros_like(inputs)
            _GRAPH_CACHE[key + ("buf",)] = static_inputs
        static_inputs.copy_(inputs)              # one 0.5 MB host->device copy
        graph, (amplitude, focal_phase) = _get_graph(key, model, static_inputs)
        graph.replay()

    slm_phase = to_mask(amplitude[0], focal_phase[0]).remainder(2 * torch.pi)
    return slm_phase.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights.pt")
    parser.add_argument("--output", type=Path, default=Path("prediction.npz"))
    parser.add_argument("--cuda-graph", action="store_true",
                        help="capture the 21-frame forward into a CUDA graph (bit-identical, ~3x faster)")
    args = parser.parse_args()
    with args.output.open("xb") as stream:
        try:
            result = predict(args.input, args.weights, cuda_graph=args.cuda_graph)
            np.savez_compressed(stream, slm_phase=result)
        except Exception:
            stream.close()
            args.output.unlink()
            raise
    print(args.output)


if __name__ == "__main__":
    main()
