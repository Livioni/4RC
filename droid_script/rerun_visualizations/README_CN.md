# DROID Stage1 Rerun 对照回放

在 `4rc` 环境中完成 Stage1 推理，打开 Rerun Web UI，同步比较所选相机的预测与真值。

```bash
conda run --no-capture-output -n 4rc python -m pip install \
  -r droid_script/rerun_visualizations/requirements.txt

conda run --no-capture-output -n 4rc python \
  droid_script/rerun_visualizations/visualize_stage1.py \
  --input datasets/droid_episodes/AUTOLab__Fri_Aug_18_11:40:54_2023
```

打开终端打印的完整 `Rerun Web viewer` URL，包括 `?url=...` 连接参数。Web 服务在模型加载前启动，推理期间显示加载提示，结果完成后可以回放。程序始终使用 Web viewer，不依赖桌面显示环境。服务器持续运行，按 Ctrl+C 退出。远程访问时，按终端提示同时转发 Web 与 gRPC 两个端口。

默认启动链接显式使用已验证的 WebGL 后端；支持 `--renderer webgpu` 切换。若旧链接出现 `Data source has left unexpectedly / Failed to fetch`，且推理进程仍在运行，可在已有 URL 末尾加 `&renderer=webgl&persist=false` 重新打开；这会用 WebGL 并忽略浏览器保存的旧会话，不需要重新推理。浏览器同样需要能访问数据端口（默认 9876）。

## 界面

- 左侧：预测深度、内参和绝对相机位姿生成的彩色点云；预测 TCP 位姿坐标轴与橙色轨迹。
- 右侧：真实 RGB-D 和标定生成的点云；Franka Panda + Robotiq 2F-85 实测关节回放；真值 TCP 位姿与绿色轨迹。
- 下方：所选相机原始 RGB，橙色/绿色分别标记预测/真值 TCP 的有效投影；白色初始 query 只在起始帧显示。旁边显示当前帧、点数和推理信息。
- 所有内容共用原始帧号与 `episode_time = frame_index / 15`。时间面板可以播放、暂停和拖动；左右初始观察位置相同，可分别旋转和缩放。

两个 3D 视图都使用 robot base 坐标系，单位米。预测侧使用模型预测的相机位姿，真值侧使用数据集真实外参；对照误差包含相机位姿预测误差。显示整段推理帧区间的 TCP 轨迹，当前坐标轴随时间变化。

## 相机、起点和推理参数

省略 `--camera` 时选择 `images/` 中名称排序后的第一路相机。切换相机后重新运行：

```bash
conda run --no-capture-output -n 4rc python \
  droid_script/rerun_visualizations/visualize_stage1.py \
  --input datasets/droid_episodes/AUTOLab__Fri_Aug_18_11:40:54_2023 \
  --camera 24400334 --max-frames 10
```

默认从首个 GT TCP 可见、且至少剩余两帧的位置开始，左右只回放相同的推理帧区间。终端与界面显示跳过的开头帧。可通过 `--start-frame N --tcp-query-point X Y` 指定原始 320×180 RGB 中的初始 query。

默认 checkpoint 为仓库内 `checkpoints/Droid-Stage1/checkpoint-250000/model.safetensors`。推理复用当前 Stage1 的严格权重恢复和滑窗逻辑，后续窗口 query 来自前窗预测。支持 `--model`、`--window-size`（默认 9，范围 2–18）、`--device`、`--dtype`、`--max-frames`（0 表示所有剩余帧）。默认模型和 URDF 路径相对仓库定位，不受启动目录影响。

点云参数：`--max-points 100000`（每帧每侧，0 表示不限制）、`--confidence-percentile 2.5`（预测侧）、`--point-size 0.003`（米）、`--min-depth-m 0.10`、`--max-depth-m 3.0`。两侧均按各自相机坐标的深度过滤。

默认 `--web-port 9090 --grpc-port 9876`；占用时自动选择空闲端口并打印实际地址。

## 真值和机械臂模型

需要所选相机的 `images/`、`depths/`、`intrinsic/<camera>.npy`、`extrinsic/<camera>.npy`、`TCP/<camera>/state.npy`，以及 `observations/joint_position.npy` 和 episode `metadata.json`。深度为 uint16 毫米 PNG，外参方向为 base → camera。按当前 DROID 导出格式读取静态相机标定。

真值 TCP 直接读取已有标签，包括位置和姿态；其工作点定义由数据集决定。机械臂七轴使用实测关节，夹爪使用 `finger_joint = (1 - gripper_open) × 0.8` rad 并同步 mimic joints。

默认模型为 `embodiments/franka-panda-robotiq-2f85/panda_robotiq_2f85.urdf`，支持 `--urdf` 覆盖为同关节命名的自包含 Panda/Robotiq 模型。本目录以标准库和已安装的 trimesh 转换随附的 triangle DAE，绕过其旧格式导入问题，保留原始坐标、URDF scale 与 visual origin；转换资产仅存于临时目录。不调用 Assimp，不导入 Lerobot_Datasets 等外部项目。转换器针对随附无 scene transform 的 triangle DAE；遇到不支持的格式会明确报错。

## 导出录制

```bash
conda run --no-capture-output -n 4rc python \
  droid_script/rerun_visualizations/visualize_stage1.py \
  --input datasets/droid_episodes/AUTOLab__Fri_Aug_18_11:40:54_2023 \
  --output rrd_outputs/droid/stage1_comparison.rrd
```

`--output` 保存含布局、预测、真值和机器人资产的 `.rrd` 后退出；可以用 Rerun 重新打开。源 episode 和 embodiment 不会被写入。
