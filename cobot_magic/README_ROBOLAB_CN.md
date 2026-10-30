# 四相机 HDF5 → 4d_dataset 风格双臂数据与 Rerun 回放

转换只导出 `cam_third_person` 的 RGB-D，保留每个 episode 的双臂状态、动作、
相机标定和 TCP。回放只读取转换结果，不加载神经网络或运行 IK。

## 环境

在 `4RC` 仓库根目录运行，推荐已有的 `4rc` 环境：

```bash
conda activate 4rc
# 仅在需要补齐依赖时执行；无需安装 cuRobo、Torch 或 CUDA
python -m pip install -r cobot_magic/requirements_4d_dataset.txt
```

## 转换与手动填写

```bash
# 只检查输入，不写文件、不提问
python cobot_magic/convert_to_4d_dataset.py --dry-run

# 默认 datasets/4_cams → datasets/4_cams_4d_dataset
python cobot_magic/convert_to_4d_dataset.py

# 自定义输入根目录，递归处理其中的 .hdf5/.h5
python cobot_magic/convert_to_4d_dataset.py \
  --input /path/to/tasks --output /path/to/converted
```

首次运行会按 **HDF5 所在目录** 询问任务名、instruction、success/failure。
任务名默认目录名，状态默认 `success`；instruction 必须填写。
同目录所有 HDF5 共享这些信息，不同目录分别填写。
每填完一个目录，立即写入 `<output>/task_metadata.json`，以后复用；可直接编辑该文件。

对于默认输入，清单内容形如：

```json
{
  "version": 1,
  "source_root": "/path/to/4_cams",
  "directories": {
    ".": {
      "task": "4_cams",
      "instruction": "请填写实际任务描述",
      "status": "success"
    }
  }
}
```

嵌套目录使用相对路径作键，例如 `pick_box` 或 `day1/pick_box`。
通过 `--metadata-file <文件>` 指定已有清单；自动运行时可用
`--instruction '实际任务描述'` 为此次选中的所有目录提供同一句 instruction。
没有终端、没有已填写的 instruction 时会明确报错，不生成任务描述。

每个源文件输出一个 `<task>/<task>__<源文件stem>/` 目录。
已有 episode 默认报错，不覆盖；要重新转换，可换输出根目录并用
`--metadata-file` 复用原清单。失败的 episode 不发布最终目录，其他 episode 继续处理，
存在失败时最终退出码非零。

```bash
# 转换后检查所有帧、标定、时间轴与 TCP 标签
python cobot_magic/convert_to_4d_dataset.py --validate-only \
  --output datasets/4_cams_4d_dataset
```

替换 `4cam.yaml` 后，已有转换数据可直接重算双臂 TCP，无需再次导出 RGB-D：

```bash
python cobot_magic/convert_to_4d_dataset.py --recompute-tcp \
  --output datasets/4_cams_4d_dataset

# --output 也可以只指定一个已有 episode
python cobot_magic/convert_to_4d_dataset.py --recompute-tcp \
  --output datasets/4_cams_4d_dataset/test/test__episode_0 \
  --calibration datasets/4_cams/4cam_calied.yaml

# 也可用 --input 直接选择已有任务目录，重算该任务下的所有 episode
python cobot_magic/convert_to_4d_dataset.py --recompute-tcp \
  --input datasets/4_cams_4d_dataset/test \
  --calibration cobot_magic/G_106.yaml
```

`--recompute-tcp` 和 `--validate-only` 均支持通过 `--input` 或 `--output` 选择
已有数据集根目录、任务目录或单个 episode；两者同时填写时必须指向同一路径。
重算会原地更新选中的数据。

重算读取已有观测关节，更新内外参、相机坐标 TCP 状态/完整位姿、基座坐标 TCP
和来源元数据。RGB-D 使用硬链接复用，保留 instruction、动作和时间戳。
在临时目录校验成功后才替换原 episode，发布失败时恢复原目录。
默认沿用原 URDF 和局部 TCP 偏移，也支持显式指定新 URDF 或 TCP 偏移。
以前导出的 `.rrd` 包含旧标签，重算后需重新导出录制。

## 几何与数据约定

- 标定默认读取 `cobot_magic/4cam.yaml`。其中矩阵按 **world/footprint → OpenCV camera**
  原样使用；`extrinsic_convention: null` 采用此已确认约定，不自动取逆。
- YAML 内参对应补边后的 322×252。导出 320×240 时减去左 1、上 6 像素补边，
  得到 `fx=fy=244.977, cx=161.4385, cy=106.9735`。
- 输入必须为 640×480 RGB 与已对齐深度。RGB 用面积插值缩小，深度用最近邻；
  深度是 uint16 毫米光轴深度，0 无效。不会再次去畸变或执行 D2C。
- 默认 URDF 为 `embodiments/aloha_new_description/urdf/aloha_tracer2_dabai_dark.urdf`。
  FK 原点为 `footprint`；`base_link` 比它高 0.15 米，不能混用。
- 源 14 维顺序是 `[左6关节, 左夹爪总开距, 右6关节, 右夹爪总开距]`。
  六轴角度直接映射 `fl_joint1–6` / `fr_joint1–6`，单位弧度。
- TCP 是闭合双指接触面面积中心的中点，姿态与对应 `link6` 相同。
  使用 URDF visual origin、scale 和网格场景变换后，在 link6 坐标中筛选
  `|x|<0.1mm` 且法向朝内的接触三角面，计算面积加权中心。
  当前双臂偏移均约 `[0,0,0.13483301]` 米；开合夹爪不会改变这个固定工作点。
- 源夹爪值保留为米制总开距，`gripper_open=clip(g/0.10,0,1)`。
  URDF 回放取 `joint7=+max(g,0)/2`、`joint8=-max(g,0)/2`；不按 URDF 的
  0.08 米总开距限位截断观测到的约 0.1056 米开距。
  平移方向按 `R_joint_origin @ axis` 转到父坐标系，两指沿腕部 X 轴对称分开。
  回放对平移关节直接构造该变换，绕过 Rerun 0.35 漏掉 origin 旋转导致的 Z 轴回缩。
- 底盘保持在 `footprint` 原点，不从 `base_action` 推测里程计。
  未观测的后臂、轮子和相机支架关节默认零，保存到元数据；可重复传入
  `--joint-default camera_stand_1_joint=0.1` 等覆盖未观测关节。
- 时间优先使用第三人称相机时间戳减首帧，保留不均匀间隔，不重采样。
  原始绝对时间另存。没有时间戳时使用 HDF5 采集 FPS；仍缺失则询问帧率，
  也可在清单中填写 `fps` 或传 `--fps`。元数据中的 FPS 是标称值。

支持 `--calibration`、`--urdf` 替换输入，以及
`--tcp-offset-left X Y Z`、`--tcp-offset-right X Y Z` 显式指定腕部局部平移。
替代 URDF 需要保持 `footprint`、前臂关节及 link6–8 的命名。

主要输出：

| 路径 | 含义 |
| --- | --- |
| `images/third_person/000000.png` | RGB，320×240 |
| `depths/third_person/000000.png` | 对齐的 uint16 毫米深度 |
| `intrinsic/third_person.npy` | `[3,3]`，对应导出图像 |
| `extrinsic/third_person.npy` | `[4,4]`，footprint → camera |
| `TCP/third_person/state.npy` | `[T,2,7]`，双臂相机坐标状态 |
| `TCP/third_person/pose.npy` | `[T,2,4,4]`，`T_camera_tcp` |
| `TCP/third_person/metadata.json` | TCP 定义、局部偏移、单位和姿态约定 |
| `observations/joint_position.npy` | `[T,14]`，源 qpos 原样保留，包含两维米制开距 |
| `observations/gripper_position.npy` | `[T,2]`，源夹爪总开距，单位米 |
| `observations/cartesian_position.npy` | `[T,2,6]`，footprint 坐标系 TCP xyz/rpy |
| `observations/camera_timestamp.npy` | `[T]`，源相机绝对时间（存在时） |
| `action/joint_position.npy` | `[T,14]`，源动作原样保留，不错位、不改写 |
| `action/gripper_position.npy` | `[T,2]`，动作中的源夹爪值 |
| `timestamps.npy` | `[T]`，episode 相对时间，秒 |
| `metadata.json` | instruction、状态、URDF/标定来源、源属性和校验统计 |

源 qvel、effort、base_action 存在时分别保存为
`observations/joint_velocity.npy`、`observations/effort.npy`、`action/base_action.npy`。
双臂顺序固定 `[left,right]`；TCP 每臂列为
`[x,y,z,roll,pitch,yaw,gripper_open]`，位置米、角度弧度，
`R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`。

格式标识是 `cobot_magic_4d_dataset_dual_arm_v1`，组织方式参考 4d_dataset，
不直接兼容其仅接受单臂 Panda 和 320×180 图像的现有测评入口。
不生成不存在的仿真状态、环境配置或对象标签。

## Rerun 可视化

```bash
python cobot_magic/visualize_rerun.py \
  --input datasets/4_cams_4d_dataset/test/test__episode_0

# 限制范围/点数
python cobot_magic/visualize_rerun.py \
  --input datasets/4_cams_4d_dataset/4_cams/4_cams__episode_0 \
  --start-frame 50 --max-frames 100 --stride 2 \
  --max-points 30000 --min-depth-m 0.1 --max-depth-m 3

# 导出含网格资产与布局的独立录制，不启动 Web 服务
python cobot_magic/visualize_rerun.py \
  --input datasets/4_cams_4d_dataset/4_cams/4_cams__episode_0 \
  --output outputs/cobot_magic/episode_0.rrd
rerun rrd verify outputs/cobot_magic/episode_0.rrd
rerun --web-viewer outputs/cobot_magic/episode_0.rrd
```

打开终端打印的完整 URL。远程使用时按提示同时转发 Web/gRPC 两个端口。
默认 9090/9876，占用时选空闲端口；默认 WebGL，也可 `--renderer webgpu`。
使用时间面板播放、暂停和拖动，Ctrl+C 停止服务。

上方 3D 视图显示 footprint 坐标系中的彩色点云、完整 ALOHA URDF、
相机视锥和双臂轨迹/姿态轴；橙色为左臂，蓝色为右臂，姿态轴 RGB 对应 XYZ。
下方是同步 RGB（含 TCP 投影）、深度图和 instruction/TCP 数值。
点云筛选不改变深度图或导出标签。`--stride` 后的 `--max-frames` 指显示帧数。

回放会检查 URDF SHA256、完整标签和 FK 一致性。数据移动后会尝试默认 URDF；
也可 `--urdf` 指定内容相同的新路径。DAE 转为临时 GLB，资源嵌入录制，
不修改原始 URDF 和网格。

## 固定相机与机器人，仅校正深度

`refine_depth_to_urdf.py` 使用 episode 当前实际保存的内外参、URDF、关节和时间对应，
将前双臂 visual 网格渲染为同视角光轴深度。使用已有 `4rc` 环境中的 Open3D CPU
射线求交；其他环境需额外安装 `open3d>=0.19`，转换和普通回放不依赖 Open3D。

```bash
python cobot_magic/refine_depth_to_urdf.py \
  --input datasets/4_cams_4d_dataset/test/test__episode_0

python cobot_magic/visualize_rerun.py \
  --input datasets/4_cams_4d_dataset/test/test__episode_0 \
  --depth-source corrected
```

默认均匀抽取 48 帧，以四个连续时间段作留出验证，用 Huber 损失拟合整段共用的
`z_corrected_m = scale * z_raw_m + bias_m`。尺度范围 `[0.9,1.1]`，偏置 ±50 mm。
拟合仅使用有效、平滑、非掠射的机器人内部像素，排除轮廓边缘、深度跳变和
原始差异 ≥80 mm 的对应。每组总体及左右臂都需满足：绝对误差中位数至少降低 20%，
P90 增幅不超过 1 mm。统计基于同一批原始筛选像素，校正误差包括毫米量化。

验证通过后新增 `depths_corrected/third_person/`，保存全部帧的 uint16 毫米深度，
保留原始分辨率、帧名和零值无效区域；有效像素取整并饱和到 `[1,65535]`。
原始深度、RGB、标定、关节与 TCP 均保留。输出目录内还有：

- `metadata.json`：深度格式与来源。
- `calibration.json`：尺度、偏置、拟合范围、配置及源文件/网格 SHA256。
- `report.json`：四组验证、全帧/逐臂/逐连杆误差、局部退化和饱和像素统计。
- `comparison.jpg`：六个代表帧，RGB+URDF 轮廓、原始残差、校正残差；蓝色偏近、
  红色偏远，范围 ±50 mm。展示全部有深度的渲染表面，包括未参与拟合的区域。

验证失败时仅生成 `depth_correction_diagnostics/` 报告，退出码非零；不会发布校正 PNG。
输出在临时目录完整校验后发布，已有结果默认拒绝覆盖。通过 `--sample-frames` 和
`--threads` 可调整拟合帧数（至少 8）和 CPU 求交线程数。

回放默认 `--depth-source raw`；选择 `corrected` 会同时切换点云与深度图，并检查
全部校正帧及来源摘要。修改原始数据、内外参、URDF 或网格后，需重新生成校正结果。
整张图的非零深度都使用统一校正，拟合距离范围以外属于外推；模型与轮廓的局部差异
可能保留或变大，见逐连杆报告。这组参数只描述当前 episode。

## 测试

```bash
python -m pytest cobot_magic/test_depth_refinement.py -q
```
