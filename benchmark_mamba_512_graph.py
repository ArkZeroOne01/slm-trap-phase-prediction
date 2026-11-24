"""Benchmark one 512x512 Mamba frame with CUDA Graph replay."""
from benchmark_single_frame import main

if __name__ == "__main__":
    main(512, use_graph=True)
