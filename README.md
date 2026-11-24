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

## 输入格式

输入为 NumPy NPZ 文件，包含以下两个 float32 数组：

| 键 | 形状 | 含义 |
|---|---|---|
| `q_xy` | `[21, 96, 2]` | 21 帧、每帧 96 个光阱的焦平面坐标，末维顺序为 x、y |
| `trap_phase` | `[21, 96]` | 每帧各光阱的输入相位，单位为弧度，使用对应坐标的 WPGS 光阱相位 |

帧按时间正序排列，两个数组里同一帧、同一索引必须对应同一个光阱。坐标以焦平面中心为原点，单位为 256×256 傅里叶网格的一格。如果沿用样例标定，1 格为 0.48828125 μm。各坐标及其四邻格须在网格内，同一帧各光阱用于编码的四邻格不能重叠。数组须为有限数值。

## 输出格式

`prediction.npz` 的 `slm_phase` 为 float32 数组，形状 `[21, 256, 256]`，轴顺序为帧、行、列。数值是 SLM 相位，单位为弧度，范围为 `[0, 2π)`。

## Mamba 单帧测速

| 输入尺寸 | 普通推理 | CUDA Graph | 权重 |
|---|---|---|---|
| 256×256 | `benchmark_mamba_256.py` | `benchmark_mamba_256_graph.py` | 仓库中的 `weights.pt` |
| 512×512 | `benchmark_mamba_512.py` | `benchmark_mamba_512_graph.py` | 随机初始化 |
| 1024×1024 | `benchmark_mamba_1024.py` | `benchmark_mamba_1024_graph.py` | 随机初始化 |

默认使用仓库的 96 光阱样例，batch size 为 1。同尺寸的 512 或 1024 两个入口使用相同的默认随机种子，也可通过 `--seed` 指定。大尺寸结果用于比较耗时。

默认测第 1 帧，无历史状态。用 `--frame-index 20` 测第 21 帧，脚本会提前构建 20 帧历史状态。两种模式都对相同的输入与历史重复测速。

```bash
python benchmark_mamba_256.py --frame-index 20
python benchmark_mamba_256_graph.py --frame-index 20
```

计时包含一次模型前向和生成 SLM 相位的 FP32 逆 FFT。CUDA Graph 版计时的是预先捕获后的单帧重放。两版都把权重加载、输入编码、历史准备、热身、Graph 捕获、结果传输和写文件放在计时之外。结果报告 CUDA 事件和同步墙钟耗时的中位数。

可用 `--warmup` 和 `--runs` 设置热身、测量次数，`--output single_frame.npz` 保存单张 `[H,W]` 相位图。自定义 `--input` 接受 `q_xy [T,96,2]` 与 `trap_phase [T,96]`。三种尺寸沿用输入的傅里叶网格坐标单位。
