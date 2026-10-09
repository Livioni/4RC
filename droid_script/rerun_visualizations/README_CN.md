# DROID Stage1 Rerun 对照回放

## 预测机械臂：推理 / IK 与回放分开运行

推荐使用下面的两个脚本。第一个缓存预测点云、相机位姿、完整 TCP 位姿与夹爪开合，并用 cuRobo v2 重建七轴关节；第二个直接加载缓存，在预测侧回放 Panda + Robotiq URDF。调整视图或导出录制无需再次推理，缓存回放无需 GPU、checkpoint 或 cuRobo。

当前 `4rc` 已安装并验证 cuRobo `0.8.0.post1.dev43`、Warp `1.17.0`，保留 PyTorch `2.11.0+cu128`、NumPy `2.4.6` 和 Rerun `0.35.0`。在其他已有 PyTorch CUDA 12 环境复现安装：

```bash
conda run --no-capture-output -n 4rc python -m pip install \
  -r droid_script/rerun_visualizations/requirements.txt
```

固定使用官方 cuRobo 提交 `78fd485fa82d9b9a063fb4985e371814587e666a` 的 **v2 API**，不是旧版 `curobo.wrap.reacher.IKSolver`。采用非 editable 安装，不需要其他项目的 cuRobo 目录。IK 需要 CUDA GPU；官方源码说明见 [安装指南](https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/docs/getting-started/installation.rst)。

第一步，生成预测与机器人关节缓存：

```bash
conda run --no-capture-output -n 4rc python \
  droid_script/rerun_visualizations/infer_stage1_ik.py \
  --input datasets/droid_episodes/AUTOLab__Fri_Aug_18_11:40:54_2023 \
  --camera 22008760
```

默认结果目录为 `outputs/droid/rerun/<episode>/<camera>`，可通过 `--output <目录>` 修改。支持原有的 `--camera`、`--model`、`--urdf`、`--start-frame`、`--tcp-query-point`、`--max-frames`、`--window-size`、`--device`、`--dtype` 和 `--max-points` 推理参数。

第二步，独立回放：

```bash
conda run --no-capture-output -n 4rc python \
  droid_script/rerun_visualizations/visualize_stage1.py \
  --prediction outputs/droid/rerun/AUTOLab__Fri_Aug_18_11:40:54_2023/22008760
```

打开打印的完整 Web viewer URL。左侧新增预测机器人、蓝色 FK TCP 与目标残差线；失败时 FK TCP 变红，并在信息面板显示 **HOLD**。右侧继续使用真值关节。两套机器人使用独立坐标帧，共用原始帧号及 15 Hz 时间轴。

缓存模式仍支持点云过滤、显示大小、端口、`--renderer` 和 `--output <文件.rrd>`。数据集移动后使用 `--episode-root <新的episode目录>`；URDF 移动后可通过 `--urdf` 指定相同内容的文件。相机、query 和帧区间由缓存固定，需要改变时重新生成缓存。

仅调整 IK，不重复神经网络推理：

```bash
conda run --no-capture-output -n 4rc python \
  droid_script/rerun_visualizations/infer_stage1_ik.py \
  --reuse-prediction outputs/droid/rerun/AUTOLab__Fri_Aug_18_11:40:54_2023/22008760 \
  --ik-num-seeds 64
```

默认在源缓存目录更新 IK，也可指定不同的 `--output <目录>`。IK 参数为 `--ik-num-seeds 32`、`--ik-position-tolerance 0.005`（米）、`--ik-rotation-tolerance 0.05`（弧度）；`--ik-no-cuda-graph` 关闭 CUDA Graph。

### 工作点、初始化与失败策略

- 将预测 TCP 通过预测相机位姿变换到 robot base；预测机器人和预测点云使用相同坐标系。
- TCP 工作点读取所选相机 `TCP/<camera>/metadata.json` 的 `tcp_offset_in_robotiq_base_m`。本 episode 为 Robotiq 基座沿 Z 轴 `0.1442549775197502 m`，即闭合夹爪接触面中点。IK 使用临时无网格 URDF 的固定 TCP link，不改动原始 embodiment。
- 用 `observations/joint_position.npy[start_frame]` 初始化。**首帧也求解预测位姿**，真值仅作 IK seed；后续帧只以上一有效解为参考，从有效候选中选择关节变化最小的解。每个候选通过 FK 误差及关节限位复核。
- 不可达或未收敛时保持上一有效七轴关节，首帧失败则保持初始化关节。夹爪始终使用预测 `gripper_open`，同步 mimic joints。残差是实际显示状态的 FK TCP 与原始预测目标之间的误差，不把保持状态标为成功。
- 第一版只进行运动学重建，不执行场景或自碰撞检查，不对目标做平滑或改写。

### 缓存内容

- `metadata.json`：episode、相机、checkpoint 来源、query、滑窗、URDF SHA256、IK 参数、初始化关节及成功统计。
- `prediction.npz`：原始帧号、相机坐标下的 TCP、预测相机到 base 的变换，以及按 offsets 拼接的点云 / 颜色 / 置信度。
- `robot_states.npz`：原始帧号、七轴关节、预测夹爪开合、FK TCP、成功标记、实际状态的位姿残差和每帧耗时。

所有 NPZ 使用数值数组，以 `allow_pickle=False` 读取；加载时检查版本、SHA256、维度和帧号一致性。预测先单独保存，IK 中断后可以 `--reuse-prediction` 恢复计算。回放仍需源 episode 的 RGB-D 与真值数据；URDF 内容改变后必须重新计算 IK。

运行缓存测试及本机 CUDA / episode 回归：

```bash
conda run --no-capture-output -n 4rc python -m unittest \
  droid_script.rerun_visualizations.test_replay -v

FOUR_RC_TEST_IK=1 conda run --no-capture-output -n 4rc python -m unittest \
  droid_script.rerun_visualizations.test_replay -v
```

`comparison.rrd` 可在相同环境通过 `rerun --web-viewer <文件.rrd>` 单独打开，用 `rerun rrd verify <文件.rrd>` 检查录制完整性。

## 原有直接推理入口

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
