# 环境变量说明

本文档列出 MemFabric_Hybrid 使用的所有环境变量及其用途。

## 运行时环境变量

以下环境变量在程序运行时读取，用于控制运行时行为：

| 环境变量名 | 默认值 | 说明 |
|-----------|-------|------|
| `CUDA_HOME` | 无（必填） | GPU环境下CUDA安装路径，用于加载CUDA动态库。NPU环境无需设置。 |
| `ASCEND_HOME_PATH` | 无（必填） | NPU环境下Ascend安装路径，用于加载CANN动态库。GPU环境无需设置。 |
| `MF_HYBM_USE_HCOMM_AICPU` | false | 设为`true`时绕过SoC判断，在任意SoC自动安装HCOMM AICPU kernel。 |
| `ASCEND_RT_VISIBLE_DEVICES` | 无 | 设备可见性控制，用于将物理设备ID映射为逻辑设备ID。格式如`0,1,2,3`。 |
| `HCOM_MAX_SLICE_SIZE` | NPU: 1MB<br>其他: 1GB | HCOM传输最大切片大小（字节），控制单次传输数据分片上限。适用于HOST_RDMA/HOST_TCP/HOST_URMA传输模式。 |
| `HCOM_RECV_DATA_SIZE` | NPU: 1MB+1KB<br>其他: 1MB+1024 | HCOM接收数据缓冲区大小（字节）。建议设置为`HCOM_MAX_SLICE_SIZE + 1024`。 |
| `MEMFABRIC_HYBRID_EXTEND_LIB_PATH` | 无 | 扩展库路径，用于加载自定义的`libmf_hybm_copy_extend.so`库。 |
| `MF_HYBM_RDMA_SWAP_SPACE_SIZE` | NPU: 1GB<br>其他: 4GB | RDMA交换空间大小（单位MB），用于host_rdma/device_rdma数据传输的中转内存。 |
| `MF_HYBM_URMA_SWAP_SPACE_SIZE` | 0 | URMA交换空间大小（单位MB），用于device_urma数据传输未注册内存的中转内存。设为0时跳过交换空间分配，未注册内存路径将直接返回错误。 |
| `MF_HYBM_RDMA_FORCE_UNREGISTERED` | 0 | 强制RDMA跳过内存注册检查路径。设为非0值时，`BatchDataCopy`直接走未注册路径发起RDMA读写。 |
| `MF_LOG_LEVEL` | 无 | MemFabric日志级别，取值范围0-4（0:DEBUG, 1:INFO, 2:WARN, 3:ERROR, 4:OFF）。**仅Python接口下生效，bm/shm场景不生效。** |
| `MF_CONFIG_STORE_URL` | 无（必填） | MemFabric Store URL，用于Transfer Engine初始化时连接配置存储。格式如`tcp://ip:port`。 |
| `MF_CONFIG_STORE_PORT_START` | 9000 | Config Store可用端口范围起始值，与`MF_CONFIG_STORE_PORT_END`配合使用。TransferEngine在`session_id`未指定端口（如`ip`/`ip:0`）时自动选端口亦使用此范围。 |
| `MF_CONFIG_STORE_PORT_END` | 65535 | Config Store可用端口范围结束值。 |
| `MF_SOCKET_URL` | 无 | 调试用的socket URL，覆盖 config store 的连接地址。 |
| `MF_TRANSPORT_MANAGER` | 无 | 传输管理器选择，用于调试指定传输层实现。 |
| `MF_GROUP_JOIN_MAX_TIMEOUT` | 600 | 集群加入最大超时时间（秒），适用于BM和Transfer模块的Join操作。 |
| `MF_GROUP_RETRY_TIME` | 5 | 集群更新（GroupUpdate）重试次数，适用于BM和Transfer模块。 |
| `MF_ACC_CHECK_PERIOD_HOURS` | 168 | 控制路径SSL证书定期检查周期（小时），有效范围24-720。 |
| `MF_ACC_CERT_CHECK_AHEAD_DAYS` | 30 | 证书提前检查天数，在证书过期前多少天开始告警，有效范围7-180。 |
| `MF_HCOM_CQ_DEPTH` | 无 | HCOM完成队列深度，设置后覆盖默认值。 |
| `MF_HCOM_SQ_SIZE` | 无 | HCOM发送队列大小，设置后覆盖默认值。 |
| `MF_HCOM_RQ_SIZE` | 无 | HCOM接收队列大小，设置后覆盖默认值。 |
| `MF_HCOM_PREPOST_SIZE` | 无 | HCOM预投递大小，设置后覆盖默认值。 |
| `MF_HCOM_MAX_SEND_RECV_DATA_CNT` | 无 | HCOM最大发送接收数据计数，设置后覆盖默认值。 |

## 构建时环境变量

以下环境变量在编译构建时读取，用于控制构建行为：

| 环境变量名 | 默认值 | 说明 |
|-----------|-------|------|
| `MEMFABRIC_VERSION` | 无（必填） | 构建版本号，用于生成whl包版本。**仅构建Python whl包时需要，C++构建不需要。** |
| `BUILD_OPEN_ABI` | OFF | 是否使用开放ABI（`_GLIBCXX_USE_CXX11_ABI`）。ON表示使用C++11 ABI。 |
| `BUILD_MODE` | RELEASE | 构建模式。可选值：RELEASE、DEBUG、ASAN。 |
| `ENABLE_PTRACER` | ON | 是否启用ptracer性能打点工具。 |
| `XPU_TYPE` | NPU | 异构设备类型。可选值：NPU、GPU、NONE。 |

## 安装脚本环境变量

| 环境变量名 | 说明 |
|-----------|------|
| `ASCEND_TOOLKIT_HOME` | Ascend Toolkit安装路径，安装脚本检查用。 |

## 使用示例

### NPU环境运行前设置

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
export MF_LOG_LEVEL=1
export HCOM_MAX_SLICE_SIZE=$((128*1024))
export HCOM_RECV_DATA_SIZE=$((128*1024+128))
export MF_HYBM_RDMA_SWAP_SPACE_SIZE=$((128))
```

### GPU环境运行前设置

```bash
export CUDA_HOME=/usr/local/cuda
export MF_LOG_LEVEL=1
```

### 无卡环境（NONE）运行前设置

```bash
export MF_LOG_LEVEL=1
export HCOM_MAX_SLICE_SIZE=$((128*1024))
export HCOM_RECV_DATA_SIZE=$((128*1024+128))
export MF_HYBM_RDMA_SWAP_SPACE_SIZE=$((128))
```
