"""Benchmark one 256x256 Mamba frame with CUDA Graph replay."""
from benchmark_single_frame import main

if __name__ == "__main__":
    main(256, use_graph=True)
