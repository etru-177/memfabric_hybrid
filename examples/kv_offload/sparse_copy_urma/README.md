# sparse_copy_urma examples

01/02 共用生产 `HybmBatchCopy`、固定 route 和
`mf_acc_offload.sparse_copy_urma`，不调用 `offload.initialize()`。建链使用
`bm.BmDataOpType.HOST_DEVICE_URMA`；route 首次发布后不增加内存区间、不替换 peer，调用方保证 entity
在同步拷贝完成前保持存活。

## 01：单机两卡 Device-HBM

需要两张 Ascend 950、可用 CANN/HCOMM/URMA 和已安装的 HYBM AICPU kernel。两个子进程分别绑定 device 0/1：

```bash
python3 examples/kv_offload/sparse_copy_urma/01_single_node_multi_device_urma.py
```

示例在每张卡的 MemFabric HBM 中初始化源数据，然后在对端通过真实框架 HBM tensor 的 `data_ptr()` 接收。
被测数据路径只有 `sparse_copy_urma`；`copy_data(H2G)` 仅用于初始化源 HBM。覆盖 1、999、1000、1001 条、
非零偏移和 range 尾部恰好结束，并打印远端 GVA 与本地 import view。

## 02：Host-DDR → NPU HBM

该示例需要分别在鲲鹏 Host 和 Ascend NPU 进程执行。Host 的 `--eid` 是 32 个十六进制字符，并用于设置
生产 Host manager 要求的 `MF_HOST_URMA_EID`。先启动 Host，再启动 NPU；`--head-ip` 必须是两端可达的
配置存储/控制网络地址：

Host：

```bash
python3 examples/kv_offload/sparse_copy_urma/02_host_device_urma.py \
  --rank 0 --head-ip <host-or-store-ip> --eid <32-hex-host-eid>
```

NPU：

```bash
python3 examples/kv_offload/sparse_copy_urma/02_host_device_urma.py \
  --rank 1 --head-ip <host-or-store-ip>
```

Host 使用固定 GVA DDR 区间并通过一次初始化 `copy_data(H2G)` 写入 pattern；NPU 侧检查 Python 可见的
`exported GVA == import view`，随后只用 `sparse_copy_urma` 读取到真实 HBM tensor。生产 Device manager
同时强制校验 `key.keys[1] == exportDesc.addr == view.addr`。该示例覆盖单条、999/1000/1001 条、非零偏移、
range 尾部和同步 completion。

如果修改 store/control 端口，Host 和 NPU 必须同时传入相同的 `--store-port`、`--ctrl-port`；不要在 route
发布完成后调用 `extend_local_mem`、替换 peer 或重新建链。

`--log-level` 同时控制 MemFabric 和 Python 脚本日志，取值与 `mf.set_log_level()` 一致：`0=DEBUG`、`1=INFO`、
`2=WARN`、`3=ERROR`、`4=OFF`。未指定时，local validation 默认 DEBUG，生产路径默认 INFO；两端建议传入相同值。

## 02：单机 DRAM → HBM 临时验证

该模式只用于同一台机器上的 1 个 Host/DRAM 进程和 1 个 NPU/HBM 进程。先构建打开验证宏的 NPU 版本；默认
构建和不带 `--local-dram-validation` 的跨节点路径不受影响：

```bash
bash script/build_and_pack_run.sh --build_local_dram_validation ON
```

EID 查询工具是 MemFabric 示例目录下的独立临时源文件，不接入 MemFabric 的 CMake、wheel 或 run 包；不依赖
`memfabric_hybrid` Python/C++ 库。以物理卡号查询并保存环境输出：

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  examples/kv_offload/sparse_copy_urma/urma_eid_query.cpp -ldl -o /tmp/mf_urma_eid_query

EID_TOOL=/tmp/mf_urma_eid_query
"${EID_TOOL}" --device-id 5 --format env --no-candidates > /tmp/mf-local-dram-eid.env
python3 examples/kv_offload/sparse_copy_urma/update_env_from_eid.py
```

工具的 `--device-id` 是物理卡号；工具将 EID 映射到 UDMA/逻辑卡，并通过 DCMI 输出该逻辑卡的
`MF_LOCAL_DRAM_AFFINITY_CPUS`（Linux cpulist，例如 `48-63`）。旧版 `libdcmi.so` 不支持查询时输出
`unavailable`，不影响 EID 查询。工具输出的 `MF_LOCAL_DRAM_PHYSICAL_DEVICE_ID`、
`MF_LOCAL_DRAM_LOGICAL_DEVICE_ID` 和 EID 环境变量由更新脚本写入本目录的 `env` 文件。该脚本保留两端公共的
`ASCEND_RT_VISIBLE_DEVICES`、`MEMFABRIC_HYBRID_EXTEND_LIB_PATH` 和 `MF_LOG_LEVEL`；Python 脚本启动时读取
该文件并通过 `os.environ` 设置进程环境，不执行 shell。`MEMFABRIC_HYBRID_EXTEND_LIB_PATH` 应指向安装包的
`lib64` 目录；上例默认按 aarch64 Linux 安装路径填写，其他架构需修改 `env`。脚本根据物理卡号自动计算
ACL/Torch 可见索引；例如 `ASCEND_RT_VISIBLE_DEVICES=5,6` 且目标物理卡为 5 时，runtime index 自动为 `0`。
脚本根据显式的 `--role host|device` 设置 Host/Device 临时角色，覆盖 Host 的
`MF_LOCAL_DRAM_VALIDATION_ROLE=host`，并清理 Device 进程中的该变量；用户无需手动设置或清理环境变量。
`--rank` 仅表示当前协议 rank，不用于推断角色。启动顺序为 Host 后 NPU，两端使用同一个 `env` 文件和相同的
store/control 地址：

终端 1（进程 A，Host/DRAM，rank 0）：

```bash
python3 examples/kv_offload/sparse_copy_urma/02_host_device_urma.py \
  --local-dram-validation --role host --rank 0 --head-ip 127.0.0.1
```

终端 2（进程 B，NPU/HBM，rank 1）：

```bash
python3 examples/kv_offload/sparse_copy_urma/02_host_device_urma.py \
  --local-dram-validation --role device --rank 1 --head-ip 127.0.0.1
```

默认读取 `examples/kv_offload/sparse_copy_urma/env`，也可通过 `--env-file <path>` 指定公共环境文件。
`--physical-device-id`、`--device-id`、`--runtime-device-id`、`--host-eid` 和 `--device-eid` 仍可显式传入，
用于兼容旧命令或做一致性校验；local 模式下不再要求重复传递。`--rank`、`--role`、`--head-ip` 和其他两端
不一致的运行参数继续通过命令行传入。

Host 先初始化固定 8 MiB GVA DRAM 和连续 pattern；两端 `create2` 均使用
`max_dram_size=8 MiB`、`max_hbm_size=1 GiB` 保持相同的全局 GVA 布局。后者满足 Ascend 950 HBM VMM 的
GB 对齐要求，但实际 HBM 分配仍是 `local_hbm_size=8 MiB`；只有 `local_*_size` 按角色决定实际分配和导出。
该 pool 会在传输前注册，因此 Host validation 进程自动将未注册内存中转使用的
`MF_HYBM_RDMA_SWAP_SPACE_SIZE` 设为 `0`。NPU 通过现有 entity/key/endpoint/route 流程取得 Host DRAM，使用
本地 MemFabric HBM pool 的实际 device VA 调用 `sparse_copy_urma`，再用 `copy_data(G2H)` 回读验证。loopback
TCP 只传版本化 JSON 元数据和结果，不传 pattern、HCOMM descriptor 或 key。默认覆盖 1、4096、1 MiB 字节数以及
1/999/1000/1001 个 4 KiB item；可用 `--rounds`、`--sizes`、`--batch-counts` 调整。

EID、物理/逻辑卡映射、尺寸、范围和地址加法在可检查处先失败，HCOMM 返回值和清理错误保留
stage/rank/device/地址/长度上下文。不要为绕过
生产 key/type/address 门禁而设置额外变量。验证完成后删除该临时 Python 分支、构建宏/脚本参数和配套工具，
再以默认 `--build_local_dram_validation OFF` 重新构建。

## 03：AICPU 发起 Host 聚合

该 Demo 只测一条固定 happy path：AICPU 写 Host mailbox，Host busy-poll 后 gather，一次大包写入
NPU HBM，AICPU 轮询 ready 后按固定 stride scatter。TCP 只在计时前做启动屏障，不承载聚合请求。

先使用 local DRAM 验证开关构建并安装 MemFabric 主包，再构建并安装 AICPU kernel：

```bash
bash script/build_and_pack_run.sh --build_local_dram_validation ON
bash script/kernel/build_ops_run.sh
./output/memfabric_hybrid_aicpu_kernel.run --install --force
```

按上文生成 `env` 后启动 Host：

```bash
python3 examples/kv_offload/sparse_copy_urma/03_aicpu_host_aggregate_urma.py \
  --role host --head-ip 127.0.0.1
```

再启动 Device：

```bash
python3 examples/kv_offload/sparse_copy_urma/03_aicpu_host_aggregate_urma.py \
  --role device --head-ip 127.0.0.1
```

默认聚合 `4096 * 2048 B = 8 MiB`。两端可同时传入相同的 `--segments` 和
`--segment-bytes`。Device 输出 AICPU 同一时钟下的 `request_us`、`wait_host_us`、`scatter_us`
和 `e2e_us`；Host 输出 `gather_us`、大包 `write_us` 和 ready 写耗时。该程序没有数据校验、
超时、重试、并发或异常清理，只用于上板穿刺。
