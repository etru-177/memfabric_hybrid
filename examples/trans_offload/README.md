# trans_offload — PD 分离 KV offload 端到端双机示例(URMA 模式)

本示例验证的架构(offload 分配 + 注册 + URMA 跨机)：

```text
Prefill 节点                                    Decode 节点
  src HBM buffer ── DEVICE_URMA 写 ──►  DRAM 池(offload 分配 + batch_register，HOST 内存
                                          注册时自动建立 device 映射并注册 URMA MR)
  Decode 提升:  offload.sparse_copy(DRAM ──► HBM page)   [AIV 算子]
  校验:         DRAM(HVA 读) == 直写数据 / HBM == 提升数据
```

职责划分(与旧版 trans_malloc 模式的差异)：

- **内存分配**：由 `offload` 组件完成(`offload.initialize` + `offload.malloc`，
  LOCAL 场景单卡 DRAM 池，reserve/alloc 强制 GB 对齐)，池地址经 SetAccess 授权后
  AIV 可直读(提升用 `sparse_copy` 源地址就是池基址)；
- **跨机发布与写入**：DRAM 池地址通过 `engine.batch_register_memory` 注册给
  smem_trans——HOST 内存注册时自动 `HalHostRegister` 建立 device 映射，URMA MR
  优先以 DVA 注册；HBM/DRAM 均经 `DEVICE_URMA` 跨机写；
- **smem_trans 不再有分配接口**(release/1.2 无 `trans_malloc`)，注册是唯一入口。

覆盖的验证点：

- `TransDataOpType.DEVICE_URMA` 双机建组与数据面；
- HOST 内存(offload 池)注册后可被远端 URMA 直写("注册即可达")；
- `offload.initialize/malloc/free` 池生命周期；
- `sparse_copy` 泛化：1 条目(奇数)顺序拷贝，无 K/V 等分假设；
- CPU 经 HVA 读 DRAM 与 AIV 提升后 HBM 数据逐字节一致。

## 平台与构建要求(重要)

| 项 | 要求 |
|---|---|
| 硬件 | **DEVICE_URMA 仅支持 ASCEND_950(A5)**；910C/A3 上 OpenDevice 会直接返回 not supported |
| 构建 | URMA 支持需在构建时启用，见 `docs/installation.md`(`--build_hcom`，必要时 `--build_hcom_ub`) |
| 组网 | DEVICE_URMA 走 UBC_CTP(EID 寻址)，跨机要求两节点 EID 可达；UBOE 变体走 IP |
| 环境 | `MEMFABRIC_HYBRID_EXTEND_LIB_PATH` 指向 `libmf_hybm_accoffload.so` 所在目录 |

## 运行步骤

### 1. Decode 节点(先启动，store server 在 Decode rank0)

单卡用例(`--npu-id 0` + 默认 `--num-writers 1` 即 1:1)：

```bash
python trans_offload_e2e_multi.py \
    --role Decode \
    --store-url tcp://<DECODE_IP>:9900 \
    --src-unique-id <DECODE_IP>:9901 \
    --npu-id 0
```

启动成功后各 rank 布局(region 数/池基址/unique_id)经 meta TCPStore 自动发布，
打印提示后保持进程等待：

```text
[Decode] layouts published, start Prefill now
```

### 2. Prefill 节点(地址自动交换，无需手工粘贴)

```bash
python trans_offload_e2e_multi.py \
    --role Prefill \
    --store-url tcp://<DECODE_IP>:9900 \
    --src-unique-id <PREFILL_IP>:9902 \
    --dst-unique-id <DECODE_IP>:9901 \
    --npu-id 0
```

### 3. 预期输出

Decode 侧依次打印：

```text
[Decode] w0 receive via DRAM write ... OK  (CPU 经 HVA 读 DRAM 校验)
[Decode] promote    128.0 MB: OK,  xx.x ms(avg of 3) => x.xxx GB/s
[Decode] trans_offload_e2e_multi (writers=...): ALL PASS
```

任一环节失败会打印 `FAILED` 并以非零码退出。

## 多对多并发写(N 卡 x M 卡 full-mesh)

`trans_offload_e2e_multi.py` 把上面的单卡用例扩展为 **N 张 Prefill 卡 x M 张
Decode 卡** 的多对多 full-mesh：Decode 侧一条命令按 `--npu-id 0,1` 拉起 M 进程 M 卡，
每卡一个 DRAM 池并按 `--num-writers` 切 region(每 writer 128MiB)，rank0 兼任 config
store 与 meta TCPStore server，各 rank 布局(region 数/池基址/unique_id)自动发布，
**免人工复制地址**；Prefill 侧一条命令按 `--npu-id 0,1,2,3` 拉起 N 进程
N 卡，卡 r 把自己的 region r 经 DEVICE_URMA **并发直写全部 M 张 Decode 卡**；Decode
各 rank 轮询本池全部 region 尾标、逐 region 校验(数据公式按
decode rank+writer rank 区分，写错卡/错位都无法通过)，再做 region0
逐档 `sparse_copy` 提升。

```bash
# Decode 节点(先启动; 2 张卡, 预期 4 个 writer)
python trans_offload_e2e_multi.py --role Decode \
    --store-url tcp://<DECODE_IP>:9900 --src-unique-id <DECODE_IP>:9901 \
    --num-writers 4 --npu-id 0,1

# Prefill 节点(一条命令拉起 4 进程 4 卡; 卡数须等于 --num-writers)
python trans_offload_e2e_multi.py --role Prefill \
    --store-url tcp://<DECODE_IP>:9900 --src-unique-id <PREFILL_IP>:9902 \
    --dst-unique-id <DECODE_IP>:9901 --npu-id 0,1,2,3

# 满配 8x8: Decode --num-writers 8 --npu-id 0,1,2,3,4,5,6,7
#           Prefill            --npu-id 0,1,2,3,4,5,6,7
```

说明：

- 卡数上限：prefill writer 与 decode 接收卡**各 ≤ 8**(编码硬上限 10，启动时 assert
  拒绝超限)；
- 聚合带宽口径(按 decode 卡)：`writers * size / 最慢 writer 耗时`。writer 数据就绪后、
  发送前经 meta TCPStore 计数屏障对齐起跑(不用 HCCL)；逐档计时窗内不再重新对齐，
  聚合值为近似参考；
- 结束屏障：每个 writer 跑完全部计时(最后一发 sync write 返回)后经 meta TCPStore
  计数 +1，Decode 等计数到 `--num-writers` 才进入校验/提升/拆通道，保证拆通道时
  没有任何在途写(对端先退出会让在途 sync write 永久挂死，是多进程下“跑完不退出”
  的根因之一)；
- 单机也可跑(两角色同机不同卡，IP 用 `127.0.0.1`)，但两侧 `--src-unique-id` 的端口
  基址需间隔 ≥ max(卡数)：子进程 unique_id 取 `<ip>:<port+rank>`，如 decode `:9901`
  占 9901..9908、prefill 用 `:9921` 占 9921..9928，避免撞端口；退化形态
  1:1 / N:1 / 1:M 均支持；
- 进程/端口：decode rank0 兼任 config store 与 meta TCPStore server，其余 rank 以
  client 重试接入；meta 端口默认 9950；
- rank0 汇齐全部布局才放行 Prefill，保证开始写时所有 decode rank 均已完成内存注册；
- 退出机制：meta TCPStore(server/client) 均在使用完后**显式释放**；各路径资源显式释放
  后由 `os._exit` 直接退出(绕过解释器关闭阶段的 C++ 析构)；子进程 join 带 300s 兜底，
  卡住的进程打印 pid 并 terminate，避免静默挂死；
- 数据公式编码：variant = 1 + decode 卡号 x 10 + writer 卡号(< 251 不折叠)，
  任意错写(错卡/错位)都无法通过校验。

## 与 device_rdma 模式(旧基线 release/1.1)的对照

| 维度 | release/1.1(trans_malloc 模式) | release/1.2(本示例，注册模式) |
|---|---|---|
| DRAM 分配方 | smem_trans(`trans_malloc`) | offload(`offload.malloc`) |
| device 可见化 | 分配时按 dataOpType 自动 | **注册时**自动(HOST 内存 + URMA → `HalHostRegister`) |
| 跨机协议 | DEVICE_RDMA(MR + rkey) | DEVICE_URMA(HcommMemReg，DVA 优先) |
| AIV 读地址 | `trans_get_dva` 返回的 DVA | offload 池地址本身(SetAccess 授权) |
| trans 接口改动 | 需新增 `trans_get_dva` 等 | **零改动**(注册机制现成) |
