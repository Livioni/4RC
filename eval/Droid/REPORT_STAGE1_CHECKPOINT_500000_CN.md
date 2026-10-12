# DROID Stage1 checkpoint-500000 中文评估报告

原始评估于 2026-10-10 完成，来源目录为
`outputs/droid/stage1_eval/checkpoint-500000/`，使用 DROID 验证名单和
`Droid-Stage1/checkpoint-500000/model.safetensors` 权重。
原始汇总见 [summary.json](../../outputs/droid/stage1_eval/checkpoint-500000/summary.json)。
本报告记录原始绝对指标，并补充全局 Sim(3) 对齐后的 TCP 轨迹指标及投影到 RGB 的 2D tracking 指标。

## 1. 主要结论

- TCP 相机坐标位置误差平均约 **12.3 mm**，姿态误差平均约 **2.85°**。
- 有效帧中约 **89.1%** 同时满足位置误差不超过 20 mm、姿态误差不超过 10°。
- 全部有效帧位置 RMSE 为 **38.14 mm**，明显高于平均误差，部分序列存在较大误差。
- 训练范围内（GT 深度不超过 3 m）的深度 MAE 约 **26 mm**；加入远距离区域后误差明显增加。
- 原始评估未计算 Sim(3) 对齐后的 ATE、RPE t、RPE r；新版评测已增加这些指标，具体口径见第 7 节。
- 本次离线补算的全部有效样本 **ATE RMSE 为 12.25 mm、RPE t RMSE 为 4.67 mm、RPE r RMSE 为 2.81°**，RPE 原始帧间隔为 1；完整结果见第 8 节。
- 新增 2D tracking 的原图平均像素误差为 **1.65 px**、4 px 阈值准确率为 **94.57%**，256×256 归一化后的跨阈值平均准确率为 **84.80%**；完整结果见第 9 节。
- 所有精度结果都建立在可见性筛选后的帧上，应与 **96.88% 的计分覆盖率**一起解读。

## 2. 评估范围与统计口径

| 项目 | 结果 |
| --- | ---: |
| 验证 episode 数 | 3,337 |
| 相机序列数 | 6,674 |
| 完成评估的相机序列 | 6,672（99.970%） |
| 运行失败的相机序列 | 0 |
| 跳过的相机序列 | 2 |
| 推理帧数 | 1,706,659 |
| 实际计分帧数 | 1,653,337（推理帧的 96.876%） |
| 排除帧数 | 53,322（推理帧的 3.124%） |
| 预测 TCP 投影越界帧数 | 43,483 |
| GT TCP 投影越界帧数 | 49,602 |
| query 回退窗口数 | 5,354 |

预测越界和 GT 越界计数存在重叠，不能直接相加得到排除帧数。
两路跳过的相机均没有满足初始化条件、且至少还剩两帧的可见 GT TCP 起点。

推理使用 9 帧滑窗，相邻窗口共享 1 帧，重叠帧保留前一窗口的预测。
首次查询点来自 GT TCP 的像素投影，后续依靠预测结果传递；不是每个窗口都使用 GT 重置。
起始帧参与计分。可见性只按几何投影判断，不检查遮挡。

`camera_macro` 先计算每条 episode/camera 序列的指标，再对相机序列等权平均。
`frame_micro` 合并全部有效帧后重新计算；深度的 `pixel_micro` 合并有效像素，
Ray 的 `ray_micro` 合并有效射线。宏平均的 median、P95 和 RMSE 分别是各序列统计值的均值，
不能当作整体分位数或整体 RMSE。

## 3. TCP 主要结果与指标说明

以下原始指标保留真实物理尺度，没有进行尺度、旋转或平移对齐。↓ 表示越低越好，↑ 表示越高越好。

| 指标 | 相机等权平均 | 全部有效帧汇总 | 含义 |
| --- | ---: | ---: | --- |
| 相机坐标位置 Mean ↓ | 12.31 mm | 12.25 mm | 三维位置 L2 距离的均值 |
| 相机坐标位置 RMSE ↓ | 14.38 mm | 38.14 mm | `sqrt(mean(position_error²))`，对大误差敏感 |
| 相机坐标旋转 Mean ↓ | 2.85° | 3.04° | 预测和 GT 旋转之间的 SO(3) 测地角 |
| 相机坐标旋转 RMSE ↓ | 3.92° | 8.51° | 旋转角误差的均方根 |
| 基座坐标位置 Mean ↓ | 10.04 mm | 10.59 mm | 预测 TCP 使用预测相机位姿转换到机器人基座后的误差 |
| 基座坐标位置 RMSE ↓ | 12.15 mm | 21.05 mm | 最终基座 TCP 位置误差的均方根 |
| 基座坐标旋转 Mean ↓ | 2.53° | 2.76° | 转换到机器人基座后的姿态误差 |
| 基座坐标旋转 RMSE ↓ | 3.66° | 8.42° | 最终基座 TCP 旋转误差的均方根 |
| TCP 重投影 Mean ↓ | 1.69 px | 1.61 px | 使用真实内参投影后的预测与 GT 像素距离 |
| 夹爪连续开合值 MAE ↓ | 0.0482 | 0.0540 | `[0,1]` 开合值的平均绝对误差 |
| 夹爪准确率 ↑ | 95.55% | 94.99% | 使用 0.5 阈值判断开合状态 |
| 位置 ≤10 mm 且旋转 ≤5° ↑ | 71.91% | 70.88% | 同一有效帧同时满足两个阈值 |
| 位置 ≤20 mm 且旋转 ≤10° ↑ | 89.17% | 89.13% | 同上 |
| 位置 ≤50 mm 且旋转 ≤15° ↑ | 95.70% | 95.87% | 同上 |

整体有效帧的 TCP 位置误差中位数为 **6.24 mm**，P95 为 **33.16 mm**；
旋转误差中位数为 **1.67°**，P95 为 **7.99°**。基座位置误差中位数为 **6.47 mm**，
P95 为 **29.38 mm**。

逐相机统计中，有 185 路序列的平均相机坐标位置误差超过 50 mm，占完成序列的 **2.77%**；
其中 59 路超过 100 mm，占 **0.88%**。因此平均指标不能代替对较大误差序列的排查。
基座误差由相机和 TCP 预测共同决定，不能简单把它与相机坐标误差之差解释为相机误差贡献。

位姿达标率衡量 TCP 估计精度，不是机器人任务执行成功率。
`tcp_confidence` 为模型输出的置信度值，不是准确率，也不参与筛选或加权。

## 4. 深度结果与指标说明

| GT 深度范围 | 汇总方式 | MAE ↓ | RMSE ↓ | AbsRel ↓ | δ1 ↑ | δ2 ↑ | δ3 ↑ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ≤3 m | camera_macro | 0.0258 m | 0.1048 m | 2.61% | 98.08% | 99.08% | 99.46% |
| ≤3 m | pixel_micro | 0.0271 m | 0.1161 m | 2.76% | 97.95% | 99.03% | 99.43% |
| 全部有效深度 | camera_macro | 0.2163 m | 0.7740 m | 5.02% | 93.84% | 95.70% | 96.86% |
| 全部有效深度 | pixel_micro | 0.2338 m | 1.9211 m | 5.28% | 93.55% | 95.53% | 96.74% |

- **MAE**：`mean(abs(pred-GT))`，单位 m。
- **RMSE**：`sqrt(mean((pred-GT)²))`，单位 m，对大误差敏感。
- **AbsRel**：`mean(abs(pred-GT)/GT)`，无量纲，本表转换为百分比展示。
- **δ1/δ2/δ3**：`max(pred/GT, GT/pred) < 1.25^k` 的像素比例，`k=1/2/3`。
- **SqRel**：`mean((pred-GT)²/GT)`；**RMSE log** 为自然对数深度误差的均方根。
- **SILog**：`100 * sqrt(mean(log_error²)-mean(log_error)²)`，描述对数误差的离散程度。

≤3 m 范围是训练监督范围，包含 81,673,699,251 个有效像素；全部范围包含
87,136,856,195 个有效像素。GT>3 m 的像素占 **6.27%**。
根据 pooled 像素数、绝对误差与平方误差反推，GT>3 m 子集的 MAE 约 **3.32 m**、
RMSE 约 **7.66 m**，说明该子集显著影响整体深度误差；这些是由汇总量推导的数值，
不是原始 CSV 独立输出的指标。远距离预测和对应 GT 深度值值得优先核查。

所有深度统计均在 TCP 帧级可见性筛选之后进行，并非对整段视频的全部帧计分。
有效像素只由 GT 决定，不按预测 confidence 或误差大小筛选；本次非正深度预测计数为 0。

## 5. 相机与 Ray 结果

| 指标 | 相机等权平均 | 含义 |
| --- | ---: | --- |
| 相机外参平移误差 | 13.02 mm | world-to-camera 平移向量 L2 误差 |
| 相机外参旋转误差 | 1.18° | 外参旋转的 SO(3) 误差 |
| 相机中心位置误差 | 15.10 mm | 相机在机器人基座中的位置误差 |
| 焦距 fx/fy 绝对误差 | 0.283 / 0.218 px | 两个方向的焦距像素偏差 |
| 焦距 fx/fy 相对误差 | 0.215% / 0.166% | 焦距偏差除以 GT 焦距 |
| 主点 cx/cy 误差 | 1.82 / 1.76 px | 图像主点位置偏差 |
| Ray 方向角误差 | 0.96° | 预测与 GT 射线方向的夹角 |
| Ray 起点位置误差 | 14.95 mm | 射线起点位置 L2 误差 |

Ray 全部有效射线汇总的方向角误差为 **0.95°**，起点位置误差为 **14.84 mm**。
相机 world-to-camera 平移误差与 camera-to-world 相机中心误差定义不同，不能混用。

## 6. 结果解释的边界

原始 3D/depth/ray 和 Sim(3) 指标的计分掩码为 `GT_TCP在RGB内 & 预测TCP在RGB内 & GT完整状态有效`。
预测越界的帧也会被剔除，因此这里的精度是条件于该掩码的精度，不能推广为全部帧精度。
起始帧之前的前缀不推理、不计分。未完成序列不进入误差汇总，但保留在完成率分母中。
本次相机完成率很高，仍应同时报告排除帧比例和大误差序列情况。
新增 2D tracking 使用独立的 GT 掩码，排除首次 query 帧并保留预测出图帧，具体见第 9 节。

## 7. 新增 TCP ATE / RPE 的计算口径

新版 `eval/Droid/eval_stage1.py` 支持以下字段，并同时输出 mean、RMSE、median 和 P95：

| 字段 | 指标 | 主要单位 |
| --- | --- | --- |
| `tcp_ate_sim3_m_*` | 全局 Sim(3) 对齐后的绝对轨迹位置误差 ATE | m |
| `tcp_rpe_t_sim3_m_*` | 对齐后完整相对位姿误差的平移部分 RPE t | m |
| `tcp_rpe_r_sim3_deg_*` | 对齐后完整相对位姿误差的旋转部分 RPE r | ° |

每个 episode/camera 的固定相机坐标 TCP 轨迹独立拟合一次 Umeyama Sim(3)，
使用原有计分掩码的全部有效对应位置。变换为 `p_aligned = s*R@p_pred+t`、
`R_aligned = R@R_pred`，不按窗口分别对齐，不改变旧的绝对 TCP、相机、depth 或 ray 指标。

ATE 的主要报告值为位置误差 RMSE。设 `Q_i` 为 GT TCP 位姿，`P_i` 为对齐后的预测位姿，
RPE 使用 `E_ij = inverse(inverse(Q_i)@Q_j) @ (inverse(P_i)@P_j)`，
平移误差为 `norm(translation(E_ij))`，旋转误差为 `angle(rotation(E_ij))`。
RPE t 包含起始 TCP 姿态误差的影响，不等同于直接相减位置位移。
公式口径参考 [evo 官方指标说明](https://github.com/MichaelGrupp/evo/wiki/Metrics)。

默认 RPE 间隔为 **原始帧号相差 1**，两个端点都须有效，使用全部重叠帧对。
跨滑窗边界可以配对，跨 episode/camera 不能配对。可用 `--rpe-delta-frames` 调整间隔。
有效位姿少于 3 个、轨迹静止/共线等导致位置协方差秩不足时，对齐无法唯一确定，
新指标记为空值，并单独报告数量；原有绝对指标仍保留。

`camera_macro` 对可定义的逐相机指标等权平均；`frame_micro` 在每条轨迹单独对齐后，
合并 ATE 帧误差或 RPE 帧对误差计算，RMSE 不使用各相机 RMSE 的加权平均。
Sim(3) 对齐会消除整体尺度、旋转和平移偏差，不能用其结果替代实际毫米级定位误差。

离线补算命令如下，无需重新加载模型或读取原始图像：

```bash
python eval/Droid/eval_stage1.py \
  --recompute-tcp-metrics-from outputs/droid/stage1_eval/checkpoint-500000 \
  --output-dir outputs/droid/stage1_eval/checkpoint-500000_tcp_sim3
```

旧来源目录保持不变，新目录输出完整汇总、逐相机指标及 `trajectory_errors.npz`。

## 8. 全量离线补算结果（2026-10-11）

补算来源为原始完整评估缓存，输出目录为
`outputs/droid/stage1_eval/checkpoint-500000_tcp_sim3/`。
原始预测不重新运行，使用与原始评估相同的有效帧掩码，RPE 原始帧间隔为 **1 帧**。
汇总见 [补算 summary.json](../../outputs/droid/stage1_eval/checkpoint-500000_tcp_sim3/summary.json)，
逐项统计见 [global_frame_metrics.csv](../../outputs/droid/stage1_eval/checkpoint-500000_tcp_sim3/global_frame_metrics.csv)。

| 覆盖项目 | 数量 |
| --- | ---: |
| 来源相机序列 | 6,674 |
| 完整且成功拟合 Sim(3) 的轨迹 | 6,672 |
| 完整轨迹中不可确定对齐的数量 | 0 |
| 来源跳过的序列 | 2 |
| ATE 有效帧 | 1,653,337 |
| RPE 有效帧对 | 1,645,442 |

| 指标 ↓ | 相机等权平均 camera_macro | 全部有效样本汇总 frame_micro |
| --- | ---: | ---: |
| ATE Mean | 6.66 mm | 7.03 mm |
| **ATE RMSE** | **8.13 mm** | **12.25 mm** |
| ATE Median | 5.54 mm | 4.50 mm |
| ATE P95 | 15.28 mm | 20.15 mm |
| RPE t Mean | 2.98 mm | 2.88 mm |
| **RPE t RMSE** | **4.19 mm** | **4.67 mm** |
| RPE t Median | 2.21 mm | 1.98 mm |
| RPE t P95 | 7.75 mm | 7.86 mm |
| RPE r Mean | 0.887° | 0.889° |
| **RPE r RMSE** | **1.469°** | **2.814°** |
| RPE r Median | 0.615° | 0.549° |
| RPE r P95 | 2.350° | 2.392° |

表中 `camera_macro` 的 Median/P95 是逐相机 Median/P95 的等权平均，
总体分位数应读取 `frame_micro` 列。ATE 与 RPE t 在 JSON/CSV 中以 m 保存，本表转换为 mm。

对齐后的总体 ATE RMSE（12.25 mm）低于原始未对齐 TCP 位置 RMSE（38.14 mm），
表示消除每条轨迹的整体相似变换后，剩余轨迹位置偏差有所降低。
两者评估口径不同，不能据此宣称物理定位精度提高；模型预测没有发生变化。
RPE 描述 1 帧间隔的局部运动一致性，也不能代替长程轨迹误差。
旋转 RPE 的总体 RMSE 明显高于平均值，仍应关注较大局部旋转误差。

补算后与原始 `summary.json` 逐项核对：全部原有逐帧指标、深度指标、Ray 指标、
覆盖计数及失败/跳过列表数值完全一致。新增实现的 **22 项回归测试全部通过**，
覆盖已知 Sim(3) 恢复、完整 SE(3) RPE 公式、原始帧间隔、退化轨迹、
无 RPE 帧对时的 ATE、滑窗边界、正常推理汇总及旧缓存离线补算。

## 9. TCP 投影到 RGB 的 2D tracking 结果（2026-10-11）

本次从已经补算 ATE/RPE 的目录读取原始预测引用，输出到
`outputs/droid/stage1_eval/checkpoint-500000_tcp_tracking/`。
来源预测不重新运行，新目录保留全部原有指标并单独增加 tracking 指标。
结果见 [global_tracking_metrics.csv](../../outputs/droid/stage1_eval/checkpoint-500000_tcp_tracking/global_tracking_metrics.csv)、
[tracking_camera_metrics.csv](../../outputs/droid/stage1_eval/checkpoint-500000_tcp_tracking/tracking_camera_metrics.csv)
及 [summary.json](../../outputs/droid/stage1_eval/checkpoint-500000_tcp_tracking/summary.json)。

使用原始相机坐标 TCP XYZ 和真实相机内参投影，不使用姿态、预测相机内参或 Sim(3) 对齐。
只统计首次 GT query 帧之后、GT 几何投影在 RGB 图内且 GT 完整状态有效的帧。
初始化帧不计分；预测出图帧继续参与像素距离及阈值统计，不按 confidence 筛选。

| 统计项目 | 数量 |
| --- | ---: |
| 完成评估的相机序列 | 6,672 |
| tracking 有效 GT 帧 | 1,650,385 |
| 可投影预测帧 | 1,650,385 |
| 无法投影预测帧 | 0 |
| 图内预测帧 | 1,646,945 |
| 出图预测帧 | 3,440 |

| 指标 | 原图 camera_macro | 原图 frame_micro | 256×256 camera_macro | 256×256 frame_micro |
| --- | ---: | ---: | ---: | ---: |
| EPE MEAN ↓ | 1.723 px | 1.647 px | 2.008 px | 1.919 px |
| EPE RMSE ↓ | 2.071 px | 4.636 px | 2.419 px | 5.502 px |
| EPE MEDIAN ↓ | 1.479 px | 0.826 px | 1.714 px | 0.930 px |
| EPE P95 ↓ | 3.650 px | 4.294 px | 4.291 px | 4.964 px |
| 距离 <1 px 的准确率 ↑ | 60.58% | 59.92% | 54.16% | 53.59% |
| 距离 <2 px 的准确率 ↑ | 85.66% | 85.61% | 81.92% | 81.81% |
| 距离 <4 px 的准确率 ↑ | 94.24% | 94.57% | 93.22% | 93.51% |
| 距离 <8 px 的准确率 ↑ | 96.94% | 97.25% | 96.55% | 96.90% |
| 距离 <16 px 的准确率 ↑ | 98.43% | 98.62% | 97.97% | 98.20% |
| 五个阈值的平均准确率 ↑ | 87.17% | 87.19% | 84.76% | 84.80% |

图内预测比例为 **99.74%（camera_macro）**、
**99.79%（frame_micro）**。分母是上述 tracking 有效 GT 帧，
它表示几何覆盖率，不表示定位准确率，也不是相对于视频全部帧的覆盖率。

EPE 是预测与 GT 像素坐标的欧氏距离。阈值准确率的分母包含全部 tracking 有效 GT 帧，
包括预测出图和无法投影的帧。可投影的出图点使用真实距离比较，例如仅越过边界 0.5 px
仍可能满足 1 px 阈值。落到相机后方、Z=0 或投影非有限时，所有阈值均计失败；
EPE 不为其虚构有限误差，而单独报告无法投影数量。本次该数量为 0。

256×256 指标把横向坐标差乘 `256/320`、纵向坐标差乘 `256/180`，再计算距离，
不是等比例缩放；原图和归一化指标的像素单位对应不同坐标系。阈值均为严格小于 1/2/4/8/16 px，
跨阈值平均准确率是五个准确率的算术平均。`camera_macro` 对序列等权，`frame_micro` 对帧等权；
宏平均的中位数和 P95 是逐序列统计值的平均，总体分位数应读取 micro 列。

阈值和首次 query 帧排除的做法参考
[TAP-Vid 官方评测实现](https://github.com/google-deepmind/tapnet/blob/main/tapnet/tapvid/evaluation_datasets.py)。
这里的 GT 可见性仅按几何投影判断，没有遮挡标注或遮挡预测，因此不计算 AJ/OA，
这些结果也不能作为完整 TAP-Vid benchmark 分数或独立 2D tracker 的评测分数。

新增 tracking 的 EPE mean/RMSE 为 1.647/4.636 px；原先重投影指标为 1.608/4.406 px。
差别来自新 tracking 排除初始化帧并纳入预测出图帧，不能混用这两个统计口径。

补算命令：

```bash
python eval/Droid/eval_stage1.py \
  --recompute-tcp-metrics-from outputs/droid/stage1_eval/checkpoint-500000_tcp_sim3 \
  --output-dir outputs/droid/stage1_eval/checkpoint-500000_tcp_tracking
```

再次补算需另选空输出目录。每条完成序列的新 `tracking_errors.npz` 保存原图和
归一化像素误差、可投影/图内标记，以及对应的原始帧号；正常新推理会把这些数组写入 `frames.npz`。

**33 项回归测试全部通过**。另外由逐相机 CSV 独立复核了全局计数、阈值正确帧数、
宏平均准确率和 pooled EPE RMSE；原有 ATE/RPE、绝对位姿、depth、ray、覆盖率及失败/跳过列表均与来源结果一致。
