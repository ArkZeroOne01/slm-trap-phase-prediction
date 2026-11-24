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
