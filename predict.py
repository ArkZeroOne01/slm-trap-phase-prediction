"""Predict one SLM phase sequence from coordinates and input phases."""
import argparse
from pathlib import Path

import numpy as np
import torch

from encoding import encode, to_mask
from model import CNNFieldCausalTransformer

ROOT = Path(__file__).resolve().parent


@torch.inference_mode()
def predict(input_path, weights_path):
    with np.load(input_path, allow_pickle=False) as source:
        q = source["q_xy"].astype(np.float32)
        phase = source["trap_phase"].astype(np.float32)
    if q.shape != (21, 96, 2) or phase.shape != (21, 96):
        raise ValueError("Expected q_xy [21, 96, 2] and trap_phase [21, 96]")
    if not np.isfinite(q).all() or not np.isfinite(phase).all():
        raise ValueError("Inputs must be finite")
    device = torch.device("cuda")
    model = CNNFieldCausalTransformer(
        size=256, patch=8, width=32, heads=4, memory_grid=4, history_frames=20
    ).to(device).eval()
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    coordinates = torch.from_numpy(q).to(device)
    phase_input = torch.from_numpy(phase).to(device)
    inputs = encode(coordinates, phase_input, 256).unsqueeze(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        amplitude, focal_phase = model(inputs)
    slm_phase = to_mask(amplitude[0], focal_phase[0]).remainder(2 * torch.pi)
    return slm_phase.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights.pt")
    parser.add_argument("--output", type=Path, default=Path("prediction.npz"))
    args = parser.parse_args()
    with args.output.open("xb") as stream:
        try:
            result = predict(args.input, args.weights)
            np.savez_compressed(stream, slm_phase=result)
        except Exception:
            stream.close()
            args.output.unlink()
            raise
    print(args.output)


if __name__ == "__main__":
    main()
