# 原版 CNN：1024×1024 随机参数测速

这是独立测速程序，不加载仓库主模型或任何训练权重，不需要数据集或 mamba-ssm。随机初始化使用 seed=123；不能用随机输出评价全息图质量。

## 启动

需要支持 BF16 的 NVIDIA GPU、CUDA 版 PyTorch。原测试环境为 RTX 5090、PyTorch 2.7.1+cu128。所有依赖源文件都在本目录。

在已配置好环境的 GPU 服务器终端执行：

```bash
nvidia-smi
cd cnn_1024_random_benchmark
CUDA_VISIBLE_DEVICES=0 python -u benchmark.py
```

根据空闲情况调整 GPU 编号。脚本自动预热、比较普通推理和 CUDA Graph，终端打印结果，并将完整记录保存至脚本同目录 `results.json`。重新运行会覆盖同名文件，请先备份已有结果。

## 固定配置

- 输入 `[21,1,2,1024,1024]`：21 帧，每帧 batch=1、振幅/相位两通道。
- CNN 输出焦平面振幅和相位，经 IFFT 和取相位生成 `[1,1024,1024]` SLM 相位图。
- BF16 autocast；光学转换保留 FP32；推理时将 BatchNorm 融入卷积。
- 最终输出为 FP32 相位；本程序不是 uint8 SLM 输出测试。
- 原版 CNN 各帧独立计算；各计时配置预热后测量 10 条完整序列。

卷积主干依据 Pan/Lu 论文补充材料图 S6：3 层前端卷积、3 个各含2层卷积的残差块、1 层输出卷积；卷积核3×3，隐藏通道16，输入/输出通道2，融合前19346个可训练参数。论文未明确公布 BN/ReLU 的具体位置，也未披露相同的混合精度和融合部署设置，因此不声称与作者代码逐行一致。

## 计时范围

| scope | 含义 |
| --- | --- |
| gpu_resident | 输入输出留在GPU；包含 CNN、IFFT和取相位 |
| cpu_to_cpu_blocking_per_frame | 加上CPU输入上传和每帧结果回传；CPU缓冲区预分配且锁页 |

`mode=eager` 为普通推理，`mode=cuda_graph` 为图重放。`sequence_ms` 是21帧平均耗时，`per_frame_ms` 为其除以21；`mean_arrivals_ms` 记录各帧回到CPU的时刻。首次捕获开销单独保存为 `capture_ms`。

上述时间排除输入图生成、初始目标相位生成、模型初始化、预热、图捕获、磁盘IO、远程网络和实际SLM刷新。8192×8192是论文WGS训练数据生成网格，不是该CNN的推理输出尺寸。

历史实测及边界说明见 [RESULTS.md](RESULTS.md)，原始记录见 [results.json](results.json)。
