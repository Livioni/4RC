# RoboTwin TCP + Depth 滑窗评测

在项目根目录、已经激活的 `4rc` 环境中运行。入口 `eval_tcp.py` 默认同时评测 TCP 和深度。

## 两阶段训练与评测入口

- `train_4rc_stage1.py` 保存原生 `Arc` 权重，与旧版单阶段训练的权重格式一致，
  可直接使用本文的 `eval_tcp.py` 进行完整 episode TCP/Depth 滑窗评测。
- `train_4rc_stage2.py` 保存完整 `TCPActionPolicy`，包含 `arc.*` 重建模块和动作、语言模块。
  该权重不能直接传给当前 `eval_tcp.py --model`；动作验证使用第二阶段的 `--eval-only`。

两个训练入口均支持 `num_train_epochs = None` 配合正整数 `max_train_steps`。
`max_train_steps` 是累计 optimizer 更新目标（包含已完成的步数），不是恢复后追加的步数。
恢复训练会校验模型、optimizer、scheduler 和 `trainer_state.json`，并在第一次更新前
按累计 `global_step`、当前调度目标和 GPU 数量对齐学习率；不会重新开始 warmup。
如果恢复点仍处于原始 warmup 内，则继续剩余 warmup。达到累计目标后不再执行训练更新。
训练参数及数据划分详见 [训练说明](README_TRAIN_CN.md)。

```bash
# 第一阶段：从完整训练状态继续，配置可设置 num_train_epochs = None
accelerate launch train_4rc_stage1.py \
  --config configs/train/4rc-giant-train-mixed.py \
  --resume outputs/4rc-robotwin-mixed-tcp-point-query/final_checkpoint \
  --max-train-steps 100000

# 第二阶段：自动读取 checkpoint/config.json，累计训练到 100000 步
accelerate launch train_4rc_stage2.py \
  --resume outputs/4rc-stage2-action/checkpoint-1000 \
  --max-train-steps 100000
```

若保存配置含有限 `num_train_epochs`，仍会受该上限约束；要取消上限，使用
`--config` 指定设置了 `num_train_epochs = None` 的配置。第二阶段不会在恢复时重载
第一阶段权重；其语言编码器仍需要配置中的 T5 资源可用。


## 运行

```bash
# 单卡，全部 50 个任务、750 个 episode
python eval_tcp.py

# 四卡；每个 GPU 加载一次模型，按 episode 分工
torchrun --standalone --nproc_per_node=4 eval_tcp.py

# 先用一个任务的 clean/random 各一个完整 episode 验证
python eval_tcp.py --tasks click_bell --episode-ids 2 8 \
  --output-dir outputs/tcp_depth_smoke

# 指定任务，仅评测 random
python eval_tcp.py --tasks adjust_bottle click_bell --split random \
  --output-dir outputs/tcp_depth_random

# 同一配置断点续跑；也可切换为 torchrun 继续
python eval_tcp.py --resume

# 指定模型、单卡设备与精度
python eval_tcp.py \
  --model outputs/4rc-robotwin-mixed-tcp-point-query/final_checkpoint/model.safetensors \
  --device cuda:0 --dtype bfloat16
```

默认数据根目录为 `datasets/eval_sets`，默认输出目录为
`outputs/4rc-robotwin-mixed-tcp-point-query/eval_tcp`。可用 `--data-root`、
`--model`、`--output-dir`、`--view`、`--window-size`、`--boundary-merge` 改写。
`--model` 接受具体的 `.safetensors` 文件。`--tasks` 是精确任务名，
`--episode-ids` 是目录末尾的数字编号，例如 `2 8`。

单卡默认自动选择设备和精度；支持 BF16 的 CUDA 使用 BF16。多卡通过
`LOCAL_RANK` 选择 GPU，此时使用 `--device auto` 或 `--device cuda`。
不使用 DDP 包装模型。所有进程需访问同一个输出目录。

默认每个任务期待编号 0–14：0–4 为 clean，5–14 为 random。缺失的预期 episode
会明确记为失败，仍进入完成率分母。筛选后不重新分组。数据 `metadata.json`
内保留的是导出前的原始 episode 编号，不用于 clean/random 划分。

## 推理口径

- 默认第三视角 `third_views`，每窗 9 帧，窗口步长 8，相邻窗重叠 1 帧。
- 只有 episode 首帧使用真值 TCP 投影生成左右查询点。后续窗使用上一窗口末帧
  的预测 TCP 投影继续追踪。不存在中途使用真值重置查询点的操作。
- `--boundary-merge previous` 默认保留前一窗对重叠帧的预测；`next` 保留后一窗；
  `average` 对位置、深度、夹爪概率取平均，对姿态使用原推理脚本的旋转插值。
- 每帧在合并完成后只统计一次，包括 episode 首帧。末尾不足 9 帧的窗口仍处理。
- 每次前向同时提取 TCP 和 depth。深度以逐帧回调累计，不进行点云或相机位姿恢复。
- 保留 checkpoint 中的 TCP 位置均值与标准差。在对应的每帧相机坐标系直接比较
  米制输出和真值，不做位姿、尺度或平移对齐，不使用 confidence 筛选。
- 自动检查 TCP 元数据、单位、旋转约定、二值夹爪及完整帧序列；真值来源为
  `TCP_third/{left,right}_state.npy`（head_view 对应 TCP_head）。

## TCP 指标

每个 episode、每个任务、全局均输出 left、right、both。`both` 将左右臂样本合并，
不要求两臂同时达到阈值；`pose_le_20mm_10deg` 指单个 arm-frame 同时满足位姿阈值。
达标率表示 TCP 估计精度，不是机器人任务执行成功率。

| 字段 | 定义 |
|---|---|
| `position_mm_mean/rmse/median/p95/max` | 三维位置 L2 误差的统计；RMSE = sqrt(mean(L2²)) |
| `x/y/z_mae_mm` | 各坐标轴绝对误差均值 |
| `rotation_deg_mean/rmse/median/p95/max` | 预测与 GT 旋转矩阵之间的 SO(3) 测地角误差 |
| `gripper_accuracy/precision/recall/f1` | 概率 ≥0.5 为 open，open 为正类；另存 TP/FP/FN/TN 计数 |
| `displacement_mm_mean/rmse` | 相邻帧预测位移减去 GT 位移后的向量范数 |
| `velocity_mm_s_mean/rmse` | 位移差范数除以实际时间间隔 |
| `relative_rotation_deg_mean/rmse` | 预测与 GT 的相邻帧相对旋转之间的测地角 |
| `relative_rotation_rate_deg_s_mean/rmse` | 上述相对旋转误差除以时间间隔 |
| `endpoint_position_mm_mean/rmse` | 每个完整 episode 末帧的位置误差 |
| `endpoint_rotation_deg_mean/rmse` | 每个完整 episode 末帧的姿态误差 |
| `position_le_10/20/50mm` | 位置误差 ≤ 对应阈值的样本比例 |
| `rotation_le_5/10/15deg` | 旋转角误差 ≤ 对应阈值的样本比例 |
| `pose_le_20mm_10deg` | 同时满足位置 ≤20 mm、姿态 ≤10° 的比例 |

旋转约定为固定轴 XYZ：`Rz(yaw) @ Ry(pitch) @ Rx(roll)`，不直接相减欧拉角。
时间间隔使用 `Δframe_index / frequency_hz`。运动误差不跨 episode；末帧指标每臂
每 episode 只有一个样本。夹爪指标出现零分母时存为 JSON null / CSV 空值。

## Depth 指标

真值来自 `depths/<view>/<frame>.png`，毫米除以 1000 转为米。预测裁掉上/下各 6、
左/右各 1 像素的 padding，得到 240×320 原图大小，不缩放真值或预测。

- `within_3m`：有限且 `0 < GT ≤ 3m`，与训练 `max_depth=3.0` 对应的主指标。
- `all_valid`：所有有限且 `GT > 0`，包含训练未监督的远处区域，作为参考指标。

两组统计共享同一次预测。有效掩码仅由 GT 决定，不按预测置信度或误差筛选，
不截断过大的深度预测，不使用训练损失中保留 98% 样本的分位数过滤。

设预测为 p、真值为 g，均以米为单位：

| 字段 | 定义 |
|---|---|
| `mae_m` | mean(abs(p−g)) |
| `abs_rel` | mean(abs(p−g)/g) |
| `sq_rel_m` | mean((p−g)²/g) |
| `rmse_m` | sqrt(mean((p−g)²)) |
| `rmse_log` | sqrt(mean((ln(p)−ln(g))²)) |
| `log10` | mean(abs(log10(p)−log10(g))) |
| `delta1/2/3` | max(p/g,g/p) 严格小于 1.25¹/²/³ 的像素比例 |

非正预测保留原始线性误差，delta 判为不达标；对数和比值计算对 p 使用 `1e-5m`
数值下限，并额外报告非正预测数量。有效 GT 上出现 NaN/Inf 预测时，对应深度范围
整个 episode 记为失败，不通过删除坏像素改善指标。

零深度或无效 GT 不参与计算。某帧没有有效 GT 时单独计数；整个 episode 在某范围
都没有有效 GT 时标为 `empty`，指标为空，不作为零误差参与均值。Depth 属于场景级
评测，不分左右臂。报告有效像素数、有效帧数、空帧数和异常预测数。

## 输出与汇总

| 文件 | 内容 |
|---|---|
| `task_metrics.csv` | TCP 每个任务 × clean/random/all × left/right/both × 聚合口径 |
| `episode_metrics.csv` | TCP 每个 episode × 手臂的指标与状态 |
| `depth_task_metrics.csv` | Depth 每个任务 × clean/random/all × 深度范围 × 聚合口径 |
| `depth_episode_metrics.csv` | Depth 每个 episode × 深度范围的指标与状态 |
| `global_tcp_metrics.csv`、`global_depth_metrics.csv` | 全局汇总，便于表格阅读 |
| `summary.json` | 全局指标、完成率、实际配置、指标定义及本次筛选列表 |
| `run_config.json` | 可恢复运行的配置与代码/权重标识 |
| `failures.jsonl` | 本次选中 episode 的失败原因、阶段及状态 |
| `episodes/<task>/<episode>.json/.npz` | 独立结果、窗口信息、原始 TCP 数组及逐帧误差和深度累计统计 |

推荐阅读任务行的 `aggregation=episode_macro`，以及全局的
`aggregation=task_macro`。各统计口径为：

- `episode_macro`：先计算每个 episode 的指标，再对 episode 等权平均。
  例如 P95 是 episode P95 的均值，RMSE 是 episode RMSE 的均值。
- `task_macro`：先获得每个任务的 episode_macro，再对任务等权平均。
- TCP `frame_micro`：合并所有成功 episode 的误差样本重新计算均值、RMSE、分位数
  和达标率；运动指标按帧对合并，末帧指标按 episode 合并。
- Depth `pixel_micro`：累计所有成功 episode 的有效像素统计，RMSE 从总体平方误差
  和除以总体像素数后开方。不会把各 episode 的 RMSE 按像素数加权当作总体 RMSE。
- episode 内的 depth 指标也是按全部有效像素计算，不先等权平均各帧。
- 所有 `*_count` 字段相加；未定义指标不参与宏平均，全部未定义则为空。

NPZ 中 `pred_*` 保留 TCP 原始预测（位置单位为米，姿态为旋转矩阵），
`gt_tcp_state` 为原始 `[T,2,7]` GT，`error_*` 为逐帧/帧对/末帧误差。
`depth_stats_within_3m`、`depth_stats_all_valid` 为 `[T,12]` 的 float64 累计量，
列名保存在 `summary.json` 的 `metric_definitions.depth_stat_columns` 中。
默认不存整幅深度图，避免全量评测产生大量深度视频文件。

## 失败与恢复

滑窗查询点投影出图、预测异常或输入异常会记录该 episode 的失败原因，继续下一个。
若中途滑窗失败，TCP 和 depth 都不使用这段未完成轨迹进入汇总。深度文件单独异常
会保留已经完整推理的 TCP 结果。TCP 与每个深度范围都有独立的计划、成功、失败、
空数据 episode 数及完成率；误差指标只来自该指标族完整成功的 episode。

存在失败时，程序先写出全部汇总，再以状态码 1 退出。模型加载、分布式通信和磁盘写入
等运行基础错误会终止进程，已原子写入的 episode 结果可供恢复。

再次使用已包含运行结果的输出目录必须添加 `--resume`。可在相同配置下更换 GPU 数量，
或更改任务/episode 筛选；本次汇总仅包含本次选中项。匹配的完整结果免推理复用，失败项
重新评测，破损或不完整缓存重新生成。输入文件标识使用路径、大小、纳秒修改时间；
权重同样使用文件标识，相关评测代码使用 SHA-256。代码、配置、权重或已缓存 episode
输入变化时拒绝复用，需换一个输出目录。NPZ 和 JSON 各自原子写入，并以指纹匹配防止混用。

## 第二阶段动作验证

```bash
python train_4rc_stage2.py \
  --resume outputs/4rc-stage2-action/final_checkpoint \
  --output-dir outputs/4rc-stage2-action/eval_action \
  --eval-only --validation-batches 16
```

默认读取 checkpoint 内的 `config.json`，验证数据按训练配置的 episode 留出规则构建。
`--eval-only` 仅加载权重，不要求 optimizer/scheduler 状态，也不执行训练更新；
需要有可用的验证 episode。结果写入指定目录的 `validation/step-00000000.json`。

- `recovered`：使用首帧查询点初始化并从恢复结果投影历史 TCP，作为主要动作预测结果。
- `teacher_forced`：使用历史真值投影，衡量历史定位误差对动作预测的影响。
- `shuffled_instruction`：替换为其他任务指令，保持相同动作初始噪声；只有一个任务时不报告。
- 报告未来动作位置 ADE/FDE、旋转误差、夹爪指标、恢复失败率、重建损失及耗时。

这是历史片段条件下的未来动作验证，与本文完整 episode 的 TCP/Depth 滑窗指标口径不同。
其中重建项是训练损失，不是上表中独立累计的 Depth 指标；也不代表闭环机器人执行成功率。
`stage1_validation_overlap` 默认 `unknown`，需核查第一阶段数据重叠后再解释为未见 episode 泛化。

## 测试

```bash
python -m pytest tests/test_training_resume.py tests/test_action_policy.py tests/test_action_dataset.py -q
```

覆盖累计训练步数、恢复学习率对齐、第二阶段 checkpoint 恢复与动作验证，以及动作模型和数据处理。
这些测试使用小模型和合成数据，不替代真实权重的完整 episode TCP/Depth GPU 评测。
