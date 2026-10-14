# DROID Stage1 按 camera 评估

在仓库根目录、激活 `4rc` 环境后运行：

```bash
python eval/Droid/eval_stage1.py
```

默认模型为 `checkpoints/Droid-Stage1/checkpoint-500000`，数据根目录为
`datasets/droid_episodes`，验证名单为 `droid_script/splits/val_set.txt`。
只选择这个名单中的 episode，分别评估其全部 camera；同一场景的两台 camera
是两个独立样本，不拼接视角，不跨 camera 传递 query。不依赖训练 SQLite 索引、IK 或机器人关节标签。

默认输出到 `outputs/droid/stage1_eval/checkpoint-500000/`。

## 常用命令

```bash
# 列出验证集 camera 数量和前十个评估任务，不加载模型
python eval/Droid/eval_stage1.py --dry-run

# 快速验证：第一个 episode 的全部 camera，各评估最多 18 帧
python eval/Droid/eval_stage1.py --max-episodes 1 --max-frames 18 \
  --output-dir outputs/droid/stage1_eval/smoke

# 完整验证集，单卡
python eval/Droid/eval_stage1.py --device cuda:0 --dtype bfloat16

# 断点续跑：复用匹配的完整结果，重试失败 camera
python eval/Droid/eval_stage1.py --device cuda:0 --dtype bfloat16 --resume

# 多卡：按 camera 任务分片，每卡各加载一份模型，最后 rank 0 汇总
torchrun --standalone --nproc_per_node=4 eval/Droid/eval_stage1.py --resume

# 指定验证名单内的 episode/camera
python eval/Droid/eval_stage1.py \
  --episodes AUTOLab__Fri_Aug_18_11:41:59_2023 --cameras 22008760 \
  --output-dir outputs/droid/stage1_eval/one_camera

# 修改数据、权重和窗口长度
python eval/Droid/eval_stage1.py --data-root /path/to/droid_episodes \
  --model checkpoints/Droid-Stage1/checkpoint-500000 --window-size 9 \
  --output-dir outputs/droid/stage1_eval/custom

# 从已完成的旧评估离线补算 TCP ATE / RPE 和 2D tracking，不加载模型
python eval/Droid/eval_stage1.py \
  --recompute-tcp-metrics-from outputs/droid/stage1_eval/checkpoint-500000 \
  --output-dir outputs/droid/stage1_eval/checkpoint-500000_tcp_tracking
```

模型路径接受 checkpoint 目录或权重文件。`--window-size` 范围为 2–18，默认 9；
`--max-frames` 从首次可见 TCP 开始计数，0 表示评估剩余全部帧。
改变模型、验证名单、选择范围、窗口长度、精度、RPE 帧间隔或评估实现时，使用新的输出目录。
`--resume` 同样校验每台 camera 的源文件信息，并重新计算损坏的缓存。
增加轨迹或 tracking 指标后，旧推理缓存不能直接用 `--resume`；请使用上面的离线补算入口。

## 初始化、滑窗与越界规则

1. 使用真实内参，将该 camera 的 GT TCP XYZ 投影到未 padding 的 `320×180` RGB。
   首次满足 `Z > 0`、坐标有限、`0 <= u < 320` 且 `0 <= v < 180` 的帧作为起点；
   该帧还需至少保留两帧用于推理。起始 query 只使用这一帧的 GT 像素位置。
   没有满足条件的起点时，该 camera 标为 `skipped`，在总覆盖率中保留。
2. 使用 `infer_4rc_stage1.py` 的预处理与 forward 参数：RGB 四边各 reflect-pad
   1 像素到 `322×182`，输入模型的 query 加 `(1,1)`，每个 window 以第一帧为 reference。
   使用连续视频帧，不随机采样、反转或增强。
3. 窗口共享一帧，例如 `0–8`、`8–16`。下一窗口 query 默认来自上一窗口最后一帧的
   预测 TCP，用 GT 相机内参投影。后续不使用 GT TCP 重置 query，也不传递 3D 状态。
4. 边界预测 TCP 越界时，选择当前窗口中**最后一个投影有效的预测位置**。
   如果整个窗口都没有有效投影，沿用更早窗口的最后有效 query；
   所有历史预测都无效时，保留首次 GT query。回退像素允许来自更早帧，不做边界裁剪。
5. 重叠帧保留上一窗口的预测，及其对应的评估掩码；下一窗口第一帧不重复计分。

原有 TCP/camera/depth/ray 和 Sim(3) 轨迹指标共享逐帧掩码：

```text
metric_mask = GT_TCP在RGB内 & 预测TCP在RGB内 & GT完整状态有效
```

GT 完整状态有效指 XYZ/RPY/夹爪值均有限、夹爪在 `[0,1]`。
任一投影越界、TCP 在相机后方的帧都不计算这些指标。新增 2D tracking 使用下面单独说明的
GT 掩码，保留预测越界帧，避免通过剔除追踪失败改善准确率。
使用回退 query 的窗口中，其他满足掩码的帧仍计分；后续预测回到 RGB 范围内即可恢复计分。
可见性只是几何投影，不判断遮挡。初始帧计分，但起始帧之前的所有帧跳过。

这个筛选口径会排除预测越界导致的失败帧，因此请同时查看
`prediction_out_of_view_frame_count`、`excluded_frame_count`、`scored_frame_count`
和 `scored_fraction_of_inferred`，不要只比较误差数值。

## 指标与单位

原有逐帧 TCP、camera、depth 和 ray 指标使用原始米制尺度，不做尺度、轨迹或
SE(3) 对齐。新增 TCP 轨迹指标单独做每条完整轨迹的全局 Sim(3) 对齐。
所有指标都不按预测 confidence 过滤，也不做异常值百分位截断。

### Depth

GT 来自 `depths/<camera>/<frame>.png`，毫米除以 1000 转为米；
预测去掉四边 padding 后在 `180×320` 原图上评估。
每台 camera 内对所有有效帧的有效像素合并计算：

| 指标 | 定义 |
| --- | --- |
| `mae_m` | `mean(abs(pred-GT))` |
| `rmse_m` | `sqrt(mean((pred-GT)^2))` |
| `abs_rel` | `mean(abs(pred-GT)/GT)` |
| `sq_rel_m` | `mean((pred-GT)^2/GT)` |
| `rmse_log` | 自然对数深度误差的 RMSE |
| `log10` | `mean(abs(log10(pred)-log10(GT)))` |
| `silog` | `100*sqrt(mean(log_error^2)-mean(log_error)^2)` |
| `delta1/2/3` | `max(pred/GT,GT/pred) < 1.25^k` 的比例 |

同时输出 `within_3m`（有限 `0 < GT <= 3m`，对应训练范围）和
`all_valid`（有限 `GT > 0`）。非正预测不从误差中删除，delta 判为失败，
仅对数计算将其截到 `1e-5m`。没有有效像素时，误差为 `null`。

另外默认输出 `median_align_*` 指标（如 `median_align_abs_rel`、
`median_align_rmse_m`、`median_align_delta1/2/3`），覆盖上表所有深度指标。
每个有效帧、每个深度范围独立在同一组 GT 有效像素上计算
`scale = median(GT) / median(pred)`，再用 `pred * scale` 计分。
预测中位数非正时，该帧不参与对齐指标，并计入
`median_align_undefined_frame_count`；无有效像素时不计为对齐失败。
对齐后仍保留非正预测的误差。相机内及 `pixel_micro` 合并对齐后的像素累计量，
`camera_macro` 对各相机指标等权平均。JSON 和现有 depth CSV 均包含新增字段，
`frames.npz` 保存 `depth_median_align_stats_<range>`。
旧缓存没有这些累计量，不能通过 TCP 离线补算获得对齐深度指标，需要重新推理。

### Camera

- `camera_translation_m_*`：预测和 GT 的 **world-to-camera** 平移向量 L2 误差，符合训练监督约定。
- `camera_rotation_deg_*`：world-to-camera 旋转的 SO(3) 测地角误差。
- `camera_center_m_*`：相机在机器人基座中的位置误差，即 camera-to-world 平移向量误差。
- `camera_fx/fy_abs_px`、`camera_fx/fy_abs_rel`：焦距像素绝对误差及相对误差。
- `camera_cx/cy_abs_px`：主点像素绝对误差。解码内参的主点先减去 padding 的 `(1,1)`。
- `camera_fov_h/w_abs_deg`：pose encoding 的 FOV 与 GT FOV 的误差；
  GT 使用训练时的 padded 图像高宽 `182/322` 和原始焦距计算。

位姿误差输出 `mean`、`rmse`、`median`、`p95`。

### TCP

- `tcp_position_m_*`：在 GT 相机坐标系中的 XYZ L2 误差；另有各轴绝对误差 `tcp_x/y/z_abs_m`。
- `tcp_rotation_deg_*`：相机坐标中的 SO(3) 误差，GT RPY 使用 `Rz(yaw) @ Ry(pitch) @ Rx(roll)`。
- `tcp_reprojection_px_*`：预测和 GT TCP 使用同一真实内参投影后的像素距离。
- `tcp_base_position_m_*`、`tcp_base_rotation_deg_*`：预测 TCP 使用**预测相机位姿**转换到基座，
  GT TCP 使用真实外参转换到基座，衡量包含相机误差的最终基座 TCP 性能。
- `tcp_gripper_mae/rmse`：连续开合值误差；`tcp_gripper_accuracy` 使用 0.5 阈值。
- `tcp_pose_le_10mm_5deg`、`tcp_pose_le_20mm_10deg`、`tcp_pose_le_50mm_15deg`：同时满足位置与旋转阈值的比例。
- `tcp_confidence`：有效帧平均 confidence，仅供查看，不参与筛选或加权。

位置、旋转和重投影误差输出 `mean`、`rmse`、`median`、`p95`。

### TCP 投影到 RGB 的 2D tracking

使用原始 TCP XYZ 和真实相机内参生成预测/GT 的 `(u,v)`，在去 padding 的
`320×180` RGB 中评估，不做 Sim(3)、相机位姿对齐或坐标裁剪，不使用预测内参。
它评价 3D TCP 投影后的位置追踪，不使用 TCP 姿态，也不代表独立 2D tracker 的性能。

tracking 使用 `GT在图内 & GT完整状态有效 & 原始帧号 > 首次query帧号` 的独立掩码。
首次 GT query 帧不参与 tracking 计分；原有重投影误差和 3D 指标仍包含初始化帧。
只排除 GT 无效或出图帧，不按预测是否在图内、confidence 或原有 `metric_mask` 剔除。

| 字段 | 定义 |
| --- | --- |
| `tcp_tracking_epe_px_mean/rmse/median/p95` | 原图像素欧氏距离，统计所有可投影的有效 GT 帧，包含预测出图帧 |
| `tcp_tracking_pts_within_1/2/4/8/16px` | 像素距离**严格小于**阈值的帧数 / 全部有效 GT 帧数 |
| `tcp_tracking_average_pts_within_thresh` | 原图五个阈值准确率的算术平均 |
| `tcp_tracking_pts_within_1/2/4/8/16px_256` | 横向误差乘 `256/320`、纵向误差乘 `256/180` 后计算相同阈值准确率 |
| `tcp_tracking_average_pts_within_thresh_256` | 256×256 归一化后五个阈值准确率的平均，参考 TAP-Vid 的位置准确率口径 |
| `tcp_tracking_epe_px_256_mean/rmse/median/p95` | 256×256 归一化后的像素距离统计 |
| `tcp_tracking_in_view_fraction` | 有效 GT 帧中预测 TCP 以正 Z 投影在 RGB 图内的比例，仅表示几何覆盖 |
| `tcp_tracking_frame_count` | 初始化之后 GT 有效且在图内的帧数，作为阈值准确率分母 |
| `tcp_tracking_projectable/unprojectable_frame_count` | 有限正 Z、像素投影有限 / 无法投影的预测帧数 |
| `tcp_tracking_prediction_in/out_of_view_frame_count` | 图内预测帧数 / 出图或无法投影的预测帧数 |
| `tcp_tracking_pts_within_<threshold>px[_256]_count` | 相应阈值的正确帧数，汇总时相加 |

出图预测若仍有有限、正 Z 的投影，使用真实像素距离计算 EPE 和阈值准确率。
例如 GT 在右边缘，预测仅出图 0.5 px，仍可能满足 1 px 误差阈值，因此位置准确率与
图内覆盖率是不同指标。预测在相机后方、Z=0 或像素投影非有限时，在所有阈值上均计失败，
保留在分母中；EPE 不为其虚构有限误差，而是单独报告无法投影帧数。

没有有效 GT 帧时比例和 EPE 为 `null`；有效 GT 帧中所有预测均无法投影时，阈值准确率为 0、
EPE 为 `null`。`camera_macro` 等权平均每条 episode/camera 的定义指标；`frame_micro`
合并样本后重新计算。它不是深度用的 `pixel_micro`，这里每帧只有一个 TCP 追踪点。

阈值、256×256 坐标缩放和排除首次 query 帧的做法参考
[TAP-Vid 官方评测实现](https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid/evaluation_datasets.py)。
当前 GT 可见性仅为几何投影，没有遮挡标注，也没有模型遮挡预测，所以这些是 DROID 上的
投影位置追踪指标，不能当作完整 TAP-Vid benchmark；不输出 AJ 或 Occlusion Accuracy。

### TCP 轨迹：全局 Sim(3) 对齐后的 ATE / RPE

每个 episode 的每路 camera 独立评估，在该固定 OpenCV 相机坐标系下，
使用全部 `metric_mask=True` 的对应 TCP 位置，以 Umeyama 方法拟合**一个**全局
Sim(3) 变换 `(s, R, t)`。不逐帧、逐窗口或跨 episode 拟合。
位置变换为 `p_aligned = s * R @ p_pred + t`，姿态变换为
`R_aligned = R @ R_pred`；尺度只作用于位置。基座 TCP 和相机误差仍按原有未对齐口径计算。

| 字段 | 定义 |
| --- | --- |
| `tcp_ate_sim3_m_mean/rmse/median/p95` | 对齐后的 TCP 位置与 GT 位置的 L2 误差，单位 m；ATE 的主要报告值为 RMSE |
| `tcp_rpe_t_sim3_m_mean/rmse/median/p95` | 相对位姿误差的平移向量范数，单位 m |
| `tcp_rpe_r_sim3_deg_mean/rmse/median/p95` | 相对位姿误差的 SO(3) 旋转角，单位 ° |
| `tcp_ate_sim3_frame_count` | 实际参与 ATE 的帧数 |
| `tcp_rpe_sim3_pair_count` | 实际参与 RPE 的帧对数 |
| `tcp_sim3_alignment_success/undefined_count` | 可计算 / 不可确定 Sim(3) 的完整相机序列数 |
| `tcp_sim3_candidate_frame_count` | 完整相机序列中原始有效帧数，含不可确定对齐的序列 |

设 `Q_i` 为 GT TCP 位姿，`P_i` 为对齐后的预测 TCP 位姿，两者均表示
TCP 局部坐标到相机坐标的变换。帧对 `(i,j)` 的 RPE 使用完整 SE(3) 公式：

```text
E_ij = inverse(inverse(Q_i) @ Q_j) @ (inverse(P_i) @ P_j)
RPE_t = norm(translation(E_ij))
RPE_r = angle(rotation(E_ij))
```

`--rpe-delta-frames` 默认 1。使用所有满足**原始帧号之差**等于该间隔、且两个
端点均有效的重叠帧对，不把剔除无效帧后的相邻样本当作原始相邻帧。
不跨 camera/episode 配对；同一轨迹内跨滑窗边界的有效帧对正常参与。
间隔大于 1 时只要求两个端点有效，不额外要求中间帧有效。

少于 3 个有效位姿、位置协方差秩小于 2（静止、共线等）或尺度无法正值拟合时，
该轨迹的新指标记为 `null`，原有绝对误差继续保留。对齐状态、原因、尺度、旋转和平移
保存在 `metrics.json` 的 `tcp_trajectory_alignment` 中。无有效帧对时 RPE 为 `null`，
ATE 仍可计算。报告结果时应同时说明帧间隔、帧/帧对数及对齐覆盖率。

`camera_macro` 等权平均可定义的每路相机指标；`frame_micro` 先分别对齐每路轨迹，
再合并其 ATE 帧误差和 RPE 帧对误差计算统计量。它不会对合并后的数据再次做 Sim(3) 拟合。
对齐会消除整体尺度、旋转和平移偏差，因此新指标应与原有毫米级绝对误差共同报告。
公式口径可参考 [evo 官方指标说明](https://github.com/MichaelGrupp/evo/wiki/Metrics)。

### 已有结果离线补算

`--recompute-tcp-metrics-from` 读取来源目录的 `summary.json`、`camera_metrics.csv`、
每路相机的 `metrics.json` 和 `frames.npz`，补算 ATE/RPE 和 2D tracking，不加载权重，
也不访问原始 RGB、depth 或标签。也可从先前的补算目录继续导出，此时通过
`metrics.json` 的 `source_frames_npz` 读取原始缓存。
只使用来源目录中的全部结果，不能同时使用 `--resume`、`--dry-run` 或 episode/camera/帧数筛选。
离线补算使用单进程；输出目录必须为空并且不同于来源目录。

输出保留原有全局及逐 camera 指标，并添加新轨迹指标；来源文件不修改。
补算配置保存来源目录、原始推理配置、来源 summary 指纹以及新的指标实现指纹。
每路相机输出 `metrics.json` 和 `trajectory_errors.npz`，后者只含紧凑的 ATE/RPE
误差及原始帧号/帧对；`tracking_errors.npz` 包含 2D tracking 像素误差、投影有效性和
对应原始帧号，不复制整份预测缓存。`source_frames_npz` 指向原始缓存。
正常新推理则把这些数组直接写入自己的 `frames.npz`。

### Ray head

以训练时的 `[world direction, origin]` 表示及采样网格构造 GT，
只保留原图区域映射到原生 ray 分辨率后的像素，不依赖 GT 深度：

- `l1`：6 个分量的平均绝对误差，对应训练 `metric_ray_l1` 的原始口径。
- `direction_l1`：方向 3 个分量的平均绝对误差。
- `direction_angle_deg`：方向向量角误差；零方向预测按 180° 计。
- `origin_l2_m`：ray origin 的米制 L2 误差。

## 输出与汇总

| 文件 | 内容 |
| --- | --- |
| `run.json` | 模型、split、评估配置和实现指纹 |
| `cameras/<episode>/<camera>/metrics.json` | 一台 camera 的所有指标、覆盖计数、初始 query、每个窗口的 query 与回退来源 |
| `cameras/<episode>/<camera>/frames.npz` | 原始帧号、GT/预测 TCP、相机位姿、内参、可见性和指标掩码、逐帧误差、depth/ray 累计量 |
| `cameras/<episode>/<camera>/trajectory_errors.npz` | 仅离线补算：对齐后的 TCP ATE/RPE 误差、ATE 原始帧号和 RPE 原始帧对 |
| `cameras/<episode>/<camera>/tracking_errors.npz` | 仅离线补算：2D tracking 原图/256×256 误差、可投影/图内标记和原始帧号；无法投影误差为 NaN |
| `camera_metrics.csv` | 每台 camera 的 TCP、camera、ray 指标与状态 |
| `depth_camera_metrics.csv` | 每台 camera × 深度范围 |
| `global_frame_metrics.csv` | camera 等权平均和合并有效帧的 TCP/camera 指标 |
| `global_depth_metrics.csv` | camera 等权平均和合并有效像素的 depth 指标 |
| `global_ray_metrics.csv` | camera 等权平均和合并有效 ray 的指标 |
| `tracking_camera_metrics.csv` | 每路 camera 的 2D tracking 指标、计数和状态 |
| `global_tracking_metrics.csv` | 2D tracking 的 camera_macro 和 frame_micro 汇总 |
| `summary.json` | 全局指标、覆盖率、失败/跳过列表及指标定义 |

`camera_macro` 给每台 camera 相同权重；`frame_micro`、`pixel_micro`、`ray_micro`
先合并有效样本再计算指标。例如 pooled RMSE 由总平方误差除以总样本数开根号，
不是平均各 camera 的 RMSE。计数相加；无定义指标为 `null`，不进入 macro 均值。
相同 camera serial 出现在不同 episode 时仍是不同评估样本。

单台 camera 失败会记录原因并继续处理其他 camera；失败的未完成轨迹不进入任何指标汇总。
模型加载失败、缺文件、标签格式错误、有效像素上的非有限 depth/ray 预测都保留失败记录。
几何图只在当前窗口中驻留内存，之后保存累计量，默认不保存整段 depth/ray 图。

运行回归测试：

```bash
python -m pytest eval/Droid/test_tcp_trajectory_metrics.py eval/Droid/test_tcp_tracking_metrics.py -q
```

覆盖已知 Sim(3) 恢复、完整 SE(3) RPE 公式、原始帧号间隔、退化轨迹、宏/微平均、
滑窗边界、旧版缓存和补算目录的离线导出、tracking 严格阈值、查询帧排除、
非等比例分辨率缩放，以及越界/无法投影预测的分母处理。完整评估结果的中文报告见
[REPORT_STAGE1_CHECKPOINT_500000_CN.md](REPORT_STAGE1_CHECKPOINT_500000_CN.md)。
