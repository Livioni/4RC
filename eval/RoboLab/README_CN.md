# RoboLab Stage1 测评

在仓库根目录激活 `4rc` 环境后运行：

```bash
conda activate 4rc
python eval/RoboLab/eval_stage1.py
```

默认模型为 `checkpoints/Droid-Stage1/checkpoint-500000`，输入为
`datasets/robolab_eval_sets/<task>/<episode>/`，输出为
`outputs/robolab/stage1_eval/checkpoint-500000/`。按任务名、episode 名和 camera 名排序，
评估全部任务下的全部 episode/camera，不依赖 Droid 验证名单。
`metadata.status=success/failure` 是采集策略的任务结果，两者都参与测评；
输出中的 `complete/failed/skipped` 则表示本次测评的执行状态。

当前数据集有 120 个任务、120 条轨迹、120 路 `third_person` 相机，共 116,136 帧。
新加入的 episode 会自动被发现。同名 episode 跨任务重复时会报错，避免覆盖输出。

## 命令

```bash
# 只扫描数据，不加载 checkpoint、不创建输出
python eval/RoboLab/eval_stage1.py --dry-run

# GPU 冒烟测试：首个 episode 的全部相机，各最多 18 帧
python eval/RoboLab/eval_stage1.py --max-episodes 1 --max-frames 18 \
  --output-dir outputs/robolab/stage1_eval/smoke

# 全量单卡；再次运行时复用完整缓存并重试失败序列
python eval/RoboLab/eval_stage1.py --device cuda:0 --resume

# 多卡按 camera 分片，每卡独立加载一份模型，rank 0 汇总
torchrun --standalone --nproc_per_node=4 eval/RoboLab/eval_stage1.py --resume

# 筛选 episode：传目录 basename，不传 task/episode 路径
python eval/RoboLab/eval_stage1.py \
  --episodes RubiksCubeTask__20261013T002328Z_d02dc858_0000 \
  --cameras third_person --output-dir outputs/robolab/stage1_eval/rubiks_cube

# 自定义数据、checkpoint 与窗口
python eval/RoboLab/eval_stage1.py --data-root datasets/robolab_eval_sets \
  --model checkpoints/Droid-Stage1/checkpoint-500000 --window-size 9 \
  --dtype bfloat16 --rpe-delta-frames 1 \
  --output-dir outputs/robolab/stage1_eval/custom

# 从预测缓存离线补算 ATE/RPE 和 2D tracking，无需模型或原始数据
python eval/RoboLab/eval_stage1.py \
  --recompute-tcp-metrics-from outputs/robolab/stage1_eval/checkpoint-500000 \
  --output-dir outputs/robolab/stage1_eval/checkpoint-500000_recomputed
```

窗口长度范围为 2–18，默认 9；`--max-episodes 0` 和 `--max-frames 0` 表示不限制。
帧数限制从首次可见 GT TCP 开始计数。默认设备为 CUDA、精度为 auto，RPE 原始帧间隔为 1。
改变模型、数据选择、窗口、精度、RPE 间隔或实现后使用新的输出目录。
续跑校验配置和相机源文件指纹（包含时间戳），跳过匹配的完整/已跳过结果，
重新计算损坏的预测缓存，重试失败序列。
离线补算必须使用新的空输出目录，只支持单进程，不能与筛选、dry-run 或 resume 混用。

## 数据约定

使用 `robolab_droid_like_v2` 导出的 RGB、深度、标定和 TCP，无需读取 HDF5、动作、关节或执行 IK。
RGB 为 320×180；深度为同帧号的 uint16 毫米 PNG，类型为光轴深度
`distance_to_image_plane`；外参为机器人 base → OpenCV camera。
TCP 为 `[x,y,z,roll,pitch,yaw,gripper_open]`，位置单位米、角度单位弧度，
旋转为 `Rz(yaw) @ Ry(pitch) @ Rx(roll)`，固定 TCP 定义为 `droid_generate_tcp_v2`。
TCP 元数据中的 `OpenCV (...)` 和 `OpenCV camera (...)` 视为相同坐标约定。
直接使用 TCP 标签中的开度，不使用 observations 中的夹爪闭合度替换，也不再次反转。

缺失或错误数据记录为失败并继续；无可用于初始化的可见 TCP 时记录为跳过。
两者保留在覆盖率中，仅完整测评序列参与误差汇总。

## 与 Droid 一致的指标

直接复用 [Droid 指标实现](../Droid/metrics.py) 和测评主流程，包含逐帧深度中位数尺度对齐的 `median_align_*` 指标；计算口径见 [Droid Depth 说明](../Droid/README_CN.md#depth)。
完整字段、公式、阈值和边界条件见 [Droid 指标说明](../Droid/README_CN.md#指标与单位)：

- TCP 相机/基座位置与姿态、各轴误差、重投影、夹爪开度/准确率、联合位姿阈值和 confidence。
- 相机 world-to-camera 平移/旋转、相机中心、焦距、主点和 FOV。
- 深度 `within_3m` 与 `all_valid`：MAE、RMSE、AbsRel、SqRel、log RMSE、log10、SILog、δ1/2/3。
- Ray：六分量 L1、方向 L1/角误差、origin L2。
- 每路轨迹全局 Sim(3) 对齐后的 TCP ATE、完整 SE(3) 平移/旋转 RPE。
- TCP 投影 2D tracking：原图和 256×256 归一化 EPE、1/2/4/8/16 px 严格阈值准确率、平均准确率和可见性计数。

首次可见 GT TCP 初始化 query，之后仅传递预测 query；窗口重叠一帧，保留前一窗口结果，
预测出图时按 Droid 规则回退到最后有效预测 query。
绝对误差与 Sim(3) 使用 `GT在图内 & 预测在图内 & GT完整状态有效` 的掩码，包含初始化帧。
Tracking 使用独立 GT 掩码且排除初始化帧；出图/无法投影的预测仍在准确率分母中。
不按 confidence 筛选，不对绝对指标做尺度对齐，不新增任务等权平均。

`camera_macro` 对 episode/camera 等权；`frame_micro`、`pixel_micro`、`ray_micro`
分别合并帧、像素、ray 后计算统计量。RMSE 和分位数不采用逐相机结果的加权平均。
退化轨迹的 Sim(3) 指标为 `null`，绝对误差保留。

## 输出与测试

输出名称和指标字段与 Droid 一致：`run.json`、`summary.json`、
`camera_metrics.csv`、`depth_camera_metrics.csv`、`tracking_camera_metrics.csv`、
`global_frame_metrics.csv`、`global_depth_metrics.csv`、`global_ray_metrics.csv`、
`global_tracking_metrics.csv`，以及 `cameras/<episode>/<camera>/metrics.json` 和 `frames.npz`。
`run.json` 的 `jobs` 保存任务名与源 episode 路径。离线补算保存紧凑的
`trajectory_errors.npz`、`tracking_errors.npz` 并引用原始预测缓存，不修改来源结果。

无需 GPU 的回归测试：

```bash
python -m pytest eval/RoboLab/test_eval_stage1.py -q
```
