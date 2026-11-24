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

这个分支提供三个入口。三个入口共用同一套 Mamba 结构、权重和计时逻辑，使用支持 BF16 的 NVIDIA CUDA GPU。默认使用仓库内的 96 光阱输入样例。

```bash
python benchmark_mamba_256.py
python benchmark_mamba_512.py
python benchmark_mamba_1024.py
```

默认测第 1 帧，此时没有历史状态。用 `--frame-index 20` 测第 21 帧，可以覆盖有 20 帧历史时的因果前向。脚本先生成历史状态，再对同一帧热身、重复测量。

```bash
python benchmark_mamba_256.py --frame-index 20 --warmup 10 --runs 50 --output single_frame.npz
```

计时包含一次 `model.step()` 和从预测焦平面复光场到 SLM 相位的逆 FFT（BF16 模型前向、FP32 光学运算）。权重加载、输入编码、构建历史、GPU 到 CPU 传输和写文件在计时之外。结果分别报告 CUDA 事件和同步墙钟耗时的中位数。三个入口都是普通推理路径，均使用 batch size 1。

自定义输入可通过 `--input path.npz` 传入。格式为 `q_xy [T,96,2]` 和 `trap_phase [T,96]`，规则同上文。三种尺寸都沿用输入坐标的傅里叶网格单位，不对坐标做缩放。可选输出文件中的 `slm_phase` 是单张 `[H,W]` 相位图。

512 和 1024 入口会按所选尺寸重建位置编码缓冲区，其他参数沿用仓库的 256 权重。这两种尺寸用于测量计算成本，预测精度尚未验证。
