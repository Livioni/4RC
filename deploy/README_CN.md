# Stage2 WebSocket 部署

服务默认加载 `checkpoints/TCP4D-task1-stage2/final_checkpoint`，接收历史第三人称 RGB、首帧双臂 TCP query、当前左右腕 RGB、当前关节状态和任务文字，输出未来 32 步双臂 TCP 位姿与夹爪开合。历史长度和预测长度来自 checkpoint，本模型分别为 **4 / 32**。

服务不保存会话历史。每次请求由 client 提交完整窗口；窗口移动后，client 必须更新首帧 query。

## 启动

在仓库根目录执行，使用已有的 `4rc` 模型环境（Python 3.10+）：

```bash
conda activate 4rc
python -m pip install -r deploy/requirements.txt
python -m deploy.server --device cuda:0 --port 8000
```

也支持 `python deploy/server.py`。checkpoint 参数接受实验根目录或含权重的 checkpoint 目录：

```bash
python -m deploy.server \
  --checkpoint checkpoints/TCP4D-task1-stage2 \
  --device cuda:0 --dtype bfloat16 --sampling-steps 4 \
  --host 0.0.0.0 --port 8000
```

参数默认值：`--device auto`、`--dtype auto`、`--seed 42`；采样步数默认读取 checkpoint（本模型为 4）。CUDA 自动选择 BF16 或 FP16，CPU 只使用 FP32。每次请求重置独立采样 generator 到指定 seed，不依赖之前请求的随机状态；相同输入在同一运行环境下可复现。

模型启动时一次性加载，严格检查 `config.json`、`model.safetensors`、`action_norm.json` 和 `state_norm.json`，并核对统计文件与权重 buffers。模型就绪后才监听端口。腕部 DINOv2 架构和权重直接从 checkpoint 恢复；T5 仍需已有缓存，或通过 `--t5-model /path/to/local/t5-base` 指定本地目录。

每个进程只加载一个策略，多个连接的推理串行执行。推理在工作线程中运行，`health` 请求与 WebSocket 心跳仍可处理。服务关闭时等待正在运行的推理结束。

## 输入与训练一致的预处理

| 输入 | 原始尺寸 / 数值范围 | 服务端处理 |
| --- | --- | --- |
| 第三人称历史 RGB | 每帧 **320×240**，`uint8 [0,255]` | 不缩放、不裁剪；左右反射填充各 1、上下各 6，成为 **322×252**，归一化到 `[-1,1]` |
| 当前左右腕 RGB | 每侧 **320×240**，`uint8 [0,255]` | 转为 `[0,1]`，复用原模型的 bicubic 256×192、reflect padding 266×196 和 ImageNet 归一化 |
| 内参、TCP query | **未 padding 的原图**像素坐标 | 内参主点和 query 同步增加 `(1,6)` |
| 当前关节 / 夹爪状态 | 12 个关节弧度、两个夹爪 `[0,1]` | 模型内部使用 checkpoint 的 state buffers 归一化 |

图像数组采用 **RGB、HWC**；OpenCV 采集的 BGR 必须由 client 先转换成 RGB。不接受 JPEG 字节、浮点图像或其他分辨率。不要提前 padding、归一化或将腕图缩至 256×192。

client 按时间从旧到新提供 1～4 帧。短历史重复最早帧：`[0] → [0,0,0,0]`，`[0,1] → [0,0,0,1]`，`[0,1,2] → [0,0,1,2]`。多于 4 帧直接报错。腕图和真实 robot state 始终对应最后一帧。

历史和未来按序号编码，不接收 timestamps。该 checkpoint 使用 `frame_rate=30`、`frame_stride=1`，client 应按约 30 Hz 的连续采样帧组织窗口。未来第 1～32 步是后续采样帧；这不表示服务推理频率为 30 Hz。

## WebSocket 接口（协议版本 1）

地址：`ws://SERVER_IP:8000`。连接建立后，server 首先发送 `type="metadata"` 消息，包含 `protocol_version`、输入尺寸、历史与预测长度、坐标及夹爪约定等。client 应先读取并检查这条消息。

所有消息都通过 `deploy.protocol.pack_message` / `unpack_message` 编解码并作为 WebSocket **binary** 消息发送。client 只需此模块以及 `numpy`、`msgpack`、`websockets`，无需安装 torch 或模型依赖。此协议不是 OpenPI 或 `msgpack-numpy` 的 wire format。

最大消息为 **16 MiB**，连接双方应设置 `max_size=MAX_MESSAGE_BYTES`、`compression=None`。4 帧历史加两张腕图已超过 WebSocket 默认的 1 MiB 限制。单连接按请求顺序返回响应，推荐收到结果后再发送下一次推理请求。

### 预测请求

| 字段 | 类型和形状 | 含义 |
| --- | --- | --- |
| `type` | `"predict"` | 请求类型 |
| `request_id` | 字符串或整数 | 响应原样回传 |
| `images` | `uint8 [T,240,320,3]` | 第三人称历史，`1≤T≤4`，从旧到新 |
| `tcp_query_points` | 数值 `[2,2]` | **首帧**原图中的 `(u,v)`，依次为左、右 TCP；`0≤u<320, 0≤v<240` |
| `intrinsics` | 数值 `[3,3]` | 对应 320×240 的 OpenCV 内参 K；该任务固定相机，各帧使用相同 K |
| `wrist_images` | `uint8 [2,240,320,3]` | 最后一帧的左腕、右腕 RGB |
| `current_robot_state` | 数值 `[14]` | 左六关节、左夹爪、右六关节、右夹爪 |
| `instruction` | 非空字符串 | 当前任务指令 |

小型数值字段可传 NumPy 数组或列表；图像必须传 `uint8` NumPy 数组。内参须可逆、`fx,fy>0`、末行 `[0,0,1]`；数值输入必须有限。

`current_robot_state` 夹爪位于下标 **6 / 13**，必须在 `[0,1]`：0 为合、1 为开。此任务的原始 `observations/joint_position.npy` 使用米制开度，训练转换为：

```python
state = raw_joint_position.astype(np.float32).copy()
state[[6, 13]] = np.clip(state[[6, 13]] / 0.10, 0, 1)
```

该转换由 client 完成，server 不猜测输入单位。无需传历史 TCP 真值、未来图像、深度或相机外参。

### 成功响应

公共字段为 `type="prediction"`、原始 `request_id`、`success=True` 和 `inference_ms`。耗时包含输入预处理、模型执行和结果转回 CPU，不包含排队与网络传输。

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `action_position` | `float32 [32,2,3]` | 未来 TCP 绝对位置，米 |
| `action_rotation` | `float32 [32,2,3,3]` | 未来 TCP 姿态矩阵 |
| `action_gripper` | `int64 [32,2]` | **1=开、0=合** |
| `action_gripper_score` | `float32 [32,2]` | 原始 score，`>=0` 为开；不是概率 |
| `future_step_indices` | `int64 [32]` | 1～32 |
| `history_position` | `float32 [T,2,3]` | 真实输入帧的 TCP 恢复位置 |
| `history_rotation` | `float32 [T,2,3,3]` | 历史 TCP 恢复姿态 |
| `history_confidence` | `float32 [T,2]` | TCP 头原始置信度，不是概率 |
| `history_query_points_px` | `float32 [T,2,2]` | 恢复出的 TCP 在各帧未 padding 原图中的投影 |
| `history_valid` | `bool [T,2]` | 历史投影与条件有效掩码 |
| `history_padding` | 整数 | 服务端重复最早帧的数量，`4-T` |

所有双臂维度都按 **左、右** 排列。

未来位姿在**最后一帧第三人称 OpenCV 相机坐标系**：x 向右、y 向下、z 向前。位置为绝对 TCP 位置，不是 delta；旋转矩阵将 TCP 局部向量变换到相机坐标。机器人基座坐标转换和 IK 由后续 client 完成，TCP 定义须与训练数据相同。

历史位姿在**各自观测帧的相机坐标系**，已去掉重复补齐帧。无效历史条目的位置、姿态、置信度和投影均填 0，必须检查 `history_valid`。投影出画不会被裁到图像边缘。

例如上一请求提交 `[f0,f1,f2,f3]`，下一请求提交 `[f1,f2,f3,f4]`，新 query 可取上一响应的 `history_query_points_px[1]`，前提是 `history_valid[1]` 双臂均为真。不要用未来预测轨迹替代历史恢复结果。无效时由 client 重新定位 TCP，再提交有效首帧 query；server 不自动维护或传播 query。

### 健康检查与错误

发送 `{"type":"health", "request_id":"ping"}`，返回 `{"type":"health", "request_id":"ping", "success":True, "ready":True}`。只有模型已经加载后才接受连接。

可恢复错误响应形如：

```python
{
    "type": "error", "request_id": "observation-1", "success": False,
    "error": {"code": "invalid_request", "message": "..."},
}
```

错误码为 `invalid_request`、`prediction_failed` 或 `inference_error`。错误响应不包含任何 `action_*` 字段，client 必须先检查 `success`；同一连接可继续发送请求。超出 16 MiB 的消息由 WebSocket 层关闭连接。无法解码请求标识时，`request_id` 为 `None`。

### 其他语言实现编解码

根消息是 MessagePack map，键为字符串。NumPy 数组编码为以下 map：

```python
{"__ndarray__": True, "dtype": "<f4", "shape": [2, 2], "data": raw_bytes}
```

`dtype` 使用 NumPy `dtype.str`（例如 `|u1`、`<f4`、`<i8`、`|b1`），`data` 是 C 顺序原始 binary。仅支持 bool、整数和实浮点数组；字节长度必须与形状及 dtype 一致，禁止对象数组或 pickle。

## 最小 Python 调用示例

以下示例在仓库根目录运行，使用本地评估 episode 的真实首帧 TCP 投影初始化 query，便于复现接口；真实 client 可用手动选点或已有标定定位 query。

```python
import asyncio
from pathlib import Path

import numpy as np
from PIL import Image
from websockets.asyncio.client import connect

from deploy.protocol import MAX_MESSAGE_BYTES, pack_message, unpack_message

episode = Path("datasets/place_pen_in_cups/eval_sets/place_pen_in_cups__episode_70")
frame_indices = [0, 1, 2, 3]

def rgb(view, frame):
    with Image.open(episode / "images" / view / f"{frame:06d}.png") as image:
        return np.asarray(image.convert("RGB")).copy()

K = np.load(episode / "intrinsic/third_person.npy").astype(np.float32)
xyz = np.load(episode / "TCP/third_person/state.npy")[frame_indices[0], :, :3]
pixels = xyz @ K.T
queries = pixels[:, :2] / pixels[:, 2:3]
state = np.load(episode / "observations/joint_position.npy")[frame_indices[-1]].copy()
state[[6, 13]] = np.clip(state[[6, 13]] / 0.10, 0, 1)
request = {
    "type": "predict", "request_id": "window-0",
    "images": np.stack([rgb("third_person", i) for i in frame_indices]),
    "tcp_query_points": queries, "intrinsics": K,
    "wrist_images": np.stack([rgb(side, frame_indices[-1]) for side in ("left_view", "right_view")]),
    "current_robot_state": state,
    "instruction": "Use the right arm to place the right pen in the right cup, then the left arm to place the left pen in the left cup.",
}

async def predict():
    async with connect("ws://127.0.0.1:8000", max_size=MAX_MESSAGE_BYTES,
                       compression=None, proxy=None) as ws:
        metadata = unpack_message(await ws.recv())
        assert metadata["protocol_version"] == 1
        await ws.send(pack_message(request))
        result = unpack_message(await ws.recv())
        if not result["success"]:
            raise RuntimeError(result["error"])
        print(result["action_position"].shape)  # (32, 2, 3)
        print(result["action_rotation"].shape)  # (32, 2, 3, 3)
        print(result["action_gripper"])         # (32, 2), 0/1

asyncio.run(predict())
```

## 验证

```bash
python -m pytest deploy/tests -q
```

测试使用轻量假策略覆盖二进制编解码、与现有推理逐值一致的预处理、原生腕图预处理、错误恢复、多连接串行推理和推理期间健康检查。WebSocket 测试需要允许绑定本地 loopback 端口，不加载完整权重或占用 GPU。

真实权重验证可先启动 CUDA server，再运行上面的评估 episode 示例；使用同一输入及 seed 直接调用 `Stage2Predictor.predict(request)` 可比对结果。

2026-11-02 已在 RTX 5090 上用此 checkpoint、BF16 autocast、4 帧历史和 4 步采样完成真实 GPU 验证：直接调用服务 predictor、模型原生 `policy.sample_actions` 和本地 WebSocket 三种路径的未来位置、旋转、夹爪及步序号一致（`rtol=1e-5, atol=1e-6`），返回完整 32 步双臂轨迹，历史双臂有效掩码均为真。此次单进程 GPU 已分配显存峰值约 9015 MiB；该数值不包含 CUDA reserved memory 或其他进程。
