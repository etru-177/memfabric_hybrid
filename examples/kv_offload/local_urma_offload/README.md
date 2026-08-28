# local_urma_offload

## 场景

单机多卡 KV offload 实例（本地 URMA 池）：每 rank 在本地 DRAM 上创建 1GB offload 池，底层由 host URMA 与 device URMA 两个 entity 组成——host entity 申请物理内存并导出，device entity 导入并 mmap，二者通过 URMA 协议互通。各 rank 通过 `sparse_copy` 将 host 侧 KV 批量拷贝到 NPU，完成一次写入与读回校验。

## 目标

验证 ACC OFFLOAD 在 `LOCAL_URMA` 场景下的双 entity 建立（create / reserve / alloc / export / 交叉 import / mmap）与稀疏拷贝数据通路是否工作正常。

## 使用能力

- `mf.set_log_level(level)`
- `offload.initialize(config: OffloadConfig)` / `offload.uninitialize()`
- `OffloadConfig`（`device_id`、`reserve_size`、`alloc_size`、`rank_id`、`scene`）
- `offload.Scene` 枚举（`LOCAL`、`SHARED`、`LOCAL_URMA`）
- `offload.empty(sizes, dtype, pin_memory)`（在本地 URMA DRAM 池分配 KV tensor）
- `offload.sparse_copy(srcPtrs, dstPtrs, lenPtrs, sizePtr, deviceId)`（批量稀疏 H2D 拷贝，底层走 entity1 的 `hybm_data_batch_copy`）

## 规模建议

- world_size=4（4 进程 4 卡，spawn 启动）
- 每 rank offload 池 reserve_size=alloc_size=1GB
- KV 维度：K_DIM=512、V_DIM=64，dtype=bfloat16
- 每 rank tokens=4×2048=8192 组 KV

## 必要条件

- 单机至少 4 张可用 NPU（device id 0/1/2/3）
- 芯片需支持 URMA（当前仅 A5 / Ascend950 系列）
- 已安装 **`memfabric_hybrid`**、`torch`、`torch_npu`、`numpy`

## 验收标准

- 各 rank 的 `offload.initialize` 与 `offload.sparse_copy` 均返回 0
- 拷贝完成后 NPU 侧 dst tensor 之和为 0（host 池零值已覆盖 device 上的 ones）
- 4 个子进程全部正常退出（exitcode=0），打印 `local_urma_offload: all ranks OK`

## 运行（Python）

```bash
cd examples/kv_offload/local_urma_offload
python3 local_urma_offload.py
```

成功时打印 `local_urma_offload: all ranks OK`。

## 与 local_dram_offload 的差异

- `scene` 设为 `offload.Scene.LOCAL_URMA`，并补充 `rank_id` 配置（用于 entity 的 `rankId`）
- 底层初始化创建两个 entity：entity1 为 `HYBM_DOP_TYPE_HOST_URMA`（申请内存 + 导出 entity/slice 信息），entity2 为 `HYBM_DOP_TYPE_DEVICE_URMA`（导入 entity1 的信息并 mmap）；两侧 entityInfo 交叉 import 并行执行
- `sparse_copy` 走 entity1 的 `hybm_data_batch_copy`（URMA 通路），而非 launch 库的 AscendC 算子
- 芯片限制收紧为仅 A5（URMA 为 A5 引入特性）

## Q&A
