# Berkeley AUTOLAB UR5 TCP4D 可视化

参考 `cobot_magic/visualize_rerun.py`，显示同步的 RGB-D 点云、UR5e + Robotiq
2F-85 模型、TCP 轨迹与姿态、标定相机投影、原始 RGB 和深度，以及任务说明。
时间轴默认暂停，点击播放或拖动查看任意帧。脚本只读取数据集。

## 运行

在仓库根目录运行，使用本机已有的 `rerun` 环境：

```bash
conda run --no-capture-output -n rerun python visualizations/visualize_berkeley_ur5_rerun.py
```

默认读取：

```text
/home/shengyu/Documents/PENG/datasets/Berkeley_AUTOLAB_UR5/TCP4D_Berkeley_AUTOLAB_UR5/train_set
```

默认选择 `berkeley_autolab_ur5__episode_0`，显示全部帧，使用数据集保存的 5 FPS。
终端会打印 Web 链接，打开即可查看，`Ctrl+C` 停止服务。
默认 Web/gRPC 端口为 9090/9876；端口占用时自动选择可用端口。
远程机器运行时，按终端给出的 SSH 命令同时转发两个实际端口。

选择其他 episode，并抽帧减少加载量：

```bash
conda run --no-capture-output -n rerun python visualizations/visualize_berkeley_ur5_rerun.py \
  --episode 12 --start-frame 5 --stride 2 --max-frames 50
```

`--episode` 支持编号或完整 episode 目录名。
`--input` 支持 `train_set`、包含 `train_set` 的数据集根目录或单个 episode 目录；
直接指定 episode 目录时忽略 `--episode`。
`--max-frames 0` 表示所有剩余抽样帧；正值表示最多记录多少个抽样帧。
抽帧后仍使用源时间戳，不压缩回放时长。

导出可独立打开的 Rerun `.rrd` 文件后退出，不启动 Web 服务：

```bash
conda run --no-capture-output -n rerun python visualizations/visualize_berkeley_ur5_rerun.py \
  --episode 0 --output outputs/visualizations/berkeley_ur5/episode_0.rrd

conda run --no-capture-output -n rerun rerun outputs/visualizations/berkeley_ur5/episode_0.rrd
```

已存在的输出文件会报错，避免覆盖。RRD 包含模型网格、图像、位姿、时间轴和布局，
打开时不需要原数据集或 CalibAll。

## 模型与坐标

- 使用数据集 `provenance` 中的 UR5e/Robotiq URDF 快照并核验其转换哈希。
  视觉网格默认从本仓库同级的 `CalibAll/third_party/urdf` 读取，也会尝试转换记录中的原始位置。
  可通过 `--caliball-root /path/to/CalibAll` 指定其他位置。
- 启动时检查所选 episode 全部帧的关节正运动学与保存的 TCP 位姿是否一致。
  六个机械臂关节使用观测角度；夹爪使用保存的 Robotiq q 和 URDF mimic 关节关系。
  数据集中的夹爪 q 由源开合状态推算，并非实测物理宽度。
- TCP 按每条轨迹 `TCP/third_person/metadata.json` 中的 v2 定义读取：
  闭合夹爪两侧接触面中心点的中点，位置取元数据中的
  `tcp_position_in_gripper_base_m`（当前约 `[0, 0, 0.144255]` 米），
  姿态与 `robotiq_arg2f_base_link` 一致。
  安装偏航和闭合 q 从元数据读取（当前分别为 90°、0.8）。
  实际夹爪张开时，TCP 标记仍遵循对应的固定闭合定义。界面显示实际使用的定义。
- 3D 场景使用 URDF `base_link` 坐标和米。保存的外参为 base-to-camera，
  脚本取逆后把点云和相机系 TCP 变换到基座系。
- 标定相机面板使用真实 K 和外参投影，RGB 面板用橙色圆点显示 TCP。
  红、绿、蓝坐标轴分别为局部 X、Y、Z。
- 深度为 uint16 毫米、光轴深度；0 无效。点云默认只保留 `0.1 < depth <= 3` 米，
  因而排除数据中保存的 65535 mm 值。深度面板保留原始像素，使用相同范围显色。
  通过 `--min-depth-m`、`--max-depth-m` 调整范围。
- TCP 轨迹覆盖所选帧段；每帧点云最多 100000 点，可用 `--max-points` 调整，0 为不限制。
  `--point-size` 控制点直径，单位为米。

没有机器人模型网格时，可使用 `--no-robot` 查看点云、TCP、RGB 和深度。

## 依赖与验证

已有 `rerun` 环境可直接运行；新环境建议 Python 3.10+：

```bash
python -m pip install -r visualizations/requirements_berkeley_ur5_rerun.txt
```

运行不依赖本地数据集的几何测试：

```bash
conda run -n rerun python -m unittest visualizations.test_berkeley_ur5_rerun -v
```
