# SLM 光阱相位预测

在支持 BF16 的 NVIDIA CUDA 环境中运行。需要 Python、NumPy、PyTorch 和与当前 PyTorch/CUDA 环境兼容的 `mamba-ssm` 扩展。

## 调用权重

在本文件夹运行：

```bash
python predict.py example_input.npz --weights weights.pt --output prediction.npz
```

`--weights` 默认读取脚本同目录下的 `weights.pt`。输出文件路径不可与现有文件相同。也可以在 Python 中调用：

```python
from predict import predict
slm_phase = predict("example_input.npz", "weights.pt")
```

## 加速：CUDA Graph（可选）

推理是固定的 21 帧因果前向，每帧要启动几十个小算子，耗时主要由启动开销决定。加 `--cuda-graph`（或在 Python 中传 `cuda_graph=True`）会把整段前向捕获成一张 CUDA 图，之后每次调用只做一次重放：

```bash
python predict.py example_input.npz --weights weights.pt --output prediction.npz --cuda-graph
```

```python
slm_phase = predict("example_input.npz", "weights.pt", cuda_graph=True)
```

RTX 5090、bf16、21 帧 × 96 阱实测：

| 方式 | 每次调用（21 帧） | 每帧 |
|---|---|---|
| 默认 | 47.6 ms | 2.27 ms |
| `--cuda-graph` | 17.2 ms | 0.82 ms |

输出与默认路径逐位相同（同一输入的 `slm_phase` 最大差为 0）。

注意事项：

- 形状需固定（21 帧、96 阱、256 网格，`predict()` 本来就要求这些）。若改用别的序列长度，会按该形状重新捕获一次。
- 首次带 `cuda_graph=True` 的调用包含捕获开销，之后的调用才会得到上表的数字。
- 计时请在进程内多次调用取最小值：GPU 空闲时时钟频率很低，单次冷调用会明显偏慢。
- 同一份 `weights.pt` 只加载一次模型，重复调用不会重复读权重。

## 输入格式

输入为 NumPy NPZ 文件，包含以下两个 float32 数组：

| 键 | 形状 | 含义 |
|---|---|---|
| `q_xy` | `[21, 96, 2]` | 21 帧、每帧 96 个光阱的焦平面坐标，末维顺序为 x、y |
| `trap_phase` | `[21, 96]` | 每帧各光阱的输入相位，单位为弧度，使用对应坐标的 WPGS 光阱相位 |

帧按时间正序排列，两个数组里同一帧、同一索引必须对应同一个光阱。坐标以焦平面中心为原点，单位为 256×256 傅里叶网格的一格。如果沿用样例标定，1 格为 0.48828125 μm。各坐标及其四邻格须在网格内，同一帧各光阱用于编码的四邻格不能重叠。数组须为有限数值。

## 输出格式

`prediction.npz` 的 `slm_phase` 为 float32 数组，形状 `[21, 256, 256]`，轴顺序为帧、行、列。数值是 SLM 相位，单位为弧度，范围为 `[0, 2π)`。
