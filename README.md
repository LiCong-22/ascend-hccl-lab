# Ascend HCCL Lab

在单机昇腾多卡环境上复现 HCCL AllReduce。每个进程绑定一张可见 NPU，用不同的 rank 值填充 FP32 张量，在所有 rank 上校验求和结果，并测量不同数据量的延迟。

## 环境

- 同一台 Linux 主机上至少两张可见、支持 HCCL 的昇腾 NPU。
- 已安装相互兼容的昇腾驱动、CANN、PyTorch 和 torch-npu，并已加载 CANN 环境变量。
- 运行前用 `npu-smi info` 确认设备，使用 `torchrun` 拉起一张卡一个进程。

## 运行

```bash
torchrun --standalone --nproc-per-node=2 allreduce.py --sizes 4096 65536 1048576
```

`--sizes` 是每个 rank 的消息字节数，默认 4 KiB、64 KiB、1 MiB。每种大小预热 5 次，测量 40 次。可用 `--warmup` 和 `--iterations` 调整次数。

rank 0 为每种大小输出一行 JSON。`validated` 表示所有 rank 回读的结果都符合预期。脚本从 Host 调用 AllReduce 计到 NPU 同步返回，每轮取所有 rank 中的最长耗时，再计算 p50 和 p95。`algorithm_bandwidth_gb_s` 用每 rank 的消息字节数除以 p50，单位为十进制 GB/s。这个指标包含调用和同步开销。

## 当前验证

已在本地检查参数解析和计时逻辑。双卡 HCCL 真机验证和性能数据待补充。

参考 [HCCL 官方 PyTorch 示例](https://gitcode.com/cann/hccl/tree/master/examples/03_ai_framework/01_pytorch/)。
