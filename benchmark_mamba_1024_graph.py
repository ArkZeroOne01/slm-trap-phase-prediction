"""Benchmark one 1024x1024 Mamba frame with CUDA Graph replay."""
from benchmark_single_frame import main

if __name__ == "__main__":
    main(1024, use_graph=True)
