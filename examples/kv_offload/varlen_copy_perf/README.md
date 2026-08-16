# varlen_copy_perf

## 场景

`varlen_copy`（AIV DRAM->HBM gather）微基准。官方 H2D 工具在 100MiB 粒度下可达约 100 GB/s，而 trans_offload 的 promote 路径峰值仅约 36 GB/s；本脚本将 `varlen_copy` 算子单独隔离出来，对两个最可能的瓶颈做扫描：

- entry 并行度：一个大 entry vs N 个等分 entry（AIV core 扇出）
- swap-in 形状：大量零散小段，模拟 PD DRAM offload 设计中真实的 top-k gather（miss token x layer）

## 目标

定位 `varlen_copy` 在不同 entry 数量、segment 大小分布下的带宽上限，验证各形状下数据拷贝的正确性。

## 使用能力

- `mf.set_log_level(level)`
- `offload.initialize(config: OffloadConfig)` / `offload.uninitialize()`
- `OffloadConfig`（`device_id`、`reserve_size`、`alloc_size`、`world_size`、`rank_id`、`scene=LOCAL`、`flags=OFFLOAD_FLAG_URMA_POOL`）
- `offload.malloc(size, deviceId)`（host DRAM 池分配，返回 hva）
- `offload.get_dva(hva)`（获取设备侧地址）
- `offload.varlen_copy(srcPtrs, dstPtrs, lenPtrs, cnt, deviceId)`（批量变长拷贝）
- `offload.free(hva, deviceId)`

## 模式说明

| 模式 | 内容 |
|------|------|
| `single` | 单大 entry（128MiB/32MiB/4MiB/128KiB）逐个计时 |
| `split` | 128MiB 等分为 x1/x4/x16/x64 个 entry（AIV core 扇出影响） |
| `swapin` | 默认 100000 x 656B 定长段（单 token gather 形状）；可用 `--merge-run` 模拟 host 侧合并，或用 `--segments/--seg-bytes-min/--seg-bytes-max` 自定义段大小分布 |
| `accuracy` | 各形状数据正确性校验（含 >UB 大段、非 32B 对齐长度、batch 边界、混合分布） |
| `all` | single + split + swapin |

多卡运行时每张 NPU 一个进程（spawn），通过本地 TCPStore 屏障保证各卡在同一时刻进入每个配置的计时阶段。

## 必要条件

- 可用 Atlas A5 NPU（支持 AIV DRAM->HBM gather）
- 已安装 **`memfabric_hybrid`**、`torch`、`torch_npu`、`numpy`

## 常用命令

```bash
cd examples/kv_offload/sparse_copy_perf

python3 varlen_copy_perf.py --mode single
python3 varlen_copy_perf.py --mode split                  # split x1/x4/x16/x64 only
python3 varlen_copy_perf.py --mode swapin                 # 100000 x 656B fixed (single-token gather)
python3 varlen_copy_perf.py --mode swapin --merge-run 8   # host-side coalesced shape (8 tokens/seg)
python3 varlen_copy_perf.py --mode swapin --segments 1024 --seg-bytes-min 2048 --seg-bytes-max 8192
python3 varlen_copy_perf.py --mode accuracy               # per-shape data correctness
python3 varlen_copy_perf.py --npu-id 0,2,3 --mode swapin  # multiple NPUs, sequential
python3 varlen_copy_perf.py --mode all
```

## 验收标准

- 启动后 `[verify] 4MiB promote data match: OK`
- `--mode accuracy` 下所有 case 显示 `OK`
- 进程退出码为 0，打印 `[done] varlen_copy perf test finished`

## 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--npu-id` | `0` | NPU id 列表，逗号分隔（如 `0,2,3`），每卡一个进程 |
| `--mode` | `all` | `single`/`split`/`swapin`/`accuracy`/`all` |
| `--rounds` | `10` | 每个配置的计时轮数 |
| `--swapin-rounds` | `50` | swapin 模式计时轮数（0 = 回退到 `--rounds`）；swapin 拷贝短且易抖动，故默认更高 |
| `--segments` | `100000` | [swapin] segment 数量 |
| `--seg-bytes-min/max` | `656` | [swapin] 段字节数上下界；min==max 表示定长 |
| `--merge-run` | `0` | [swapin] 每 N 个连续 token 合并为一段（模拟 host 侧合并），0 关闭 |
| `--tokens` | `100000` | [swapin][--merge-run] 总 token 数 |
| `--item-bytes` | `656` | [swapin][--merge-run] 每 token 每 layer 的 KV 字节数 |
| `--sync-port` | `29531` | 多卡 TCPStore 屏障端口 |

## Q&A
