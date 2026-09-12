# 4RC 在 RoboTwin 上的 Geometry + TCP 训练说明

本文档说明如何联合训练 4RC geometry 分支和双臂 TCP 轨迹分支。

当前训练范围：

- RGB 默认使用 RoboTwin `third_views`，TCP 标签读取 `TCP_third`；
- 首帧 TCP 只用于生成二维投影与 loss 标签，不会作为 7D query 输入模型；
- 训练 backbone、DualDPT、共享 motion decoder、视觉 query encoder 和 absolute TCP head；
- camera decoder 与 dense track head 默认冻结；
- 损失由 depth、ray、TCP pose、temporal velocity 和 gripper 组成。

## 1. 代码布局

```text
4RC/
├── arc/
│   ├── datasets/
│   │   ├── robotwin.py          # RoboTwin dataloader、时序采样、collate
│   │   └── utils/geometry.py    # 尺度归一化、相对外参、ray GT
│   └── loss/
│       ├── geometry.py          # depth、gradient、ray loss
│       └── tcp_tracking.py      # TCP pose、temporal、gripper loss
├── configs/
│   └── train/
│       ├── 4rc-giant-train.py   # 默认联合训练参数
│       └── 4rc-tcp-recovery.py  # 修复后恢复 TCP position 的微调参数
└── train_4rc_stage1.py                 # Accelerate 训练入口
```

配置目录和参数写法参考 Depth-Anything-Next：训练配置放在 `configs/train/`，每个参数直接定义为 Python 顶层变量。

## 2. 环境准备

```bash
conda activate 4rc
pip install -r requirements.txt
pip install -e .
```

训练入口依赖 `torch`、`torchvision`、`accelerate`、`tensorboard` 和 `wandb`（已列入 `requirements.txt`）。默认同时记录 TensorBoard 和 W&B，首次在线训练前运行 `wandb login`。离线记录可在启动命令前设置 `WANDB_MODE=offline`；仅使用 TensorBoard 时将配置中的 `report_to` 改为 `"tensorboard"`。

## 3. RoboTwin 数据目录

默认数据根目录为 `datasets/RoboTwin`，需要满足：

```text
datasets/RoboTwin/
└── <task>/
    └── <episode>/
        ├── metadata.json
        ├── images/third_views/000000.png
        ├── depths/third_views/000000.png
        ├── intrinsics/third_views.npy
        ├── extrinsics/third_views.npy
        └── TCP_third/
            ├── metadata.json
            ├── left_state.npy
            └── right_state.npy
```

- RGB 必须是 `320×240`。
- depth 是 uint16 毫米值，读取后转换为米；0 或超过默认 `3m` 的像素无效并置零。
- extrinsics 是逐帧 OpenCV world-to-camera，形状为 `[T,3,4]`。
- intrinsics 形状为 `[3,3]`。
- episode `metadata.json` 提供真实 `frequency_hz`（当前数据均为 15 Hz）。
- TCP state 为 `[T,7] = xyz(m), rpy(rad), gripper_open(0/1)`。

原图不会 resize 或 crop。为满足 patch size 14，dataloader 会左右各反射填充 1 像素、上下各反射填充 6 像素，模型实际输入为 `322×252`；填充区不参与损失。

## 4. 修改训练参数

编辑：

```text
configs/train/4rc-giant-train.py
```

### 数据和采样

```python
data_root = "datasets/RoboTwin"
view = "third_views"
max_depth = 3.0  # 设为 None 可关闭最远深度过滤
min_views = 2
max_views = 18
min_interval = 1
max_interval = 5
reverse_probability = 0.5
frame_rate = None  # 默认读取每个 episode 的 metadata.json
max_tcp_linear_speed = 3.0
max_tcp_angular_speed = 4.0 * 3.141592653589793
max_episodes = None
augment = True
num_workers = 8
train_batch_images = 18
scene_counts = (1, 2, 3, 6, 9)
batches_per_epoch = None
recent_buffer_size = 10_000
```

`TCP_third` 的首帧相机坐标会通过内参投影到 padded 图像。模型从每只手臂
投影点周围采样 `3×3` 4RC patch tokens，并用真实 patch 坐标作为 MotionDecoder
的二维 RoPE。训练前 10% step 以 80% 精确投影、20% jitter 开始，随后过渡到
25% 精确投影、75% 最大一 patch jitter。

默认参考 Depth-Anything-Next 的固定图像预算 sampler：每张 GPU 每个
DataLoader batch 始终输入 18 张图。每步从 `scene_counts` 均匀选择场景数
`B`，每个场景采样 `S = 18 // B` 帧，因此可能得到：

```text
1×18, 2×9, 3×6, 6×3, 9×2
```

张量形状为 `[B,S,3,252,322]`，不会把不同 episode 当成同一个 18 帧
序列。每个场景仍使用 4RC 时序采样：随机选择 `min_interval～max_interval`
的固定时间间隔，再随机选择合法起点；采样完成后以
`reverse_probability` 的概率将整个 clip 倒序（默认正序、倒序各 50%）。RGB、
depth 和相机参数会同步倒序；外参不参与帧选择。设为 `0` 可保持仅正序，设为
`1` 则始终倒序。

`batches_per_epoch=None` 表示每个 epoch 产生 `len(dataset)` 个逻辑
batch。`recent_buffer_size` 用于降低 episode 在相邻 batch 中被重复选择的
概率。短 episode 或较小的 `max_episodes` 会让不可行的组合自动退出候选池。

单 batch 调试时可以设置：

```python
train_batch_images = 2
scene_counts = (1,)
min_views = 2
max_views = 2
max_episodes = 1
batches_per_epoch = 1
num_workers = 0
```

此时形状固定为 `[1,2,3,252,322]`。如果保留默认
`train_batch_images=18`，则 `max_episodes=1` 时会自动使用 `1×18`。

### 冻结模块

```python
train_backbone = True
train_geometry_head = True
train_camera_decoder = False
train_motion_decoder = False
train_tcp_tracker = True
```

`train_tcp_tracker=True` 会训练共享 motion decoder、视觉 query encoder 和 absolute
TCP head；`train_motion_decoder=False` 仅避免启用未使用的 dense track head。训练
前向会跳过 camera decoder 和 dense tracking 分支。

`find_unused_parameters=True` 必须保持开启：DualDPT 为兼容预训练权重保留了
ray 金字塔各层的预测模块，而当前几何目标只监督最终 ray 层。关闭该选项会让
DDP 在第二次同步反传后报 `Expected to have finished reduction`。

等价的模型接口为：

```python
model.configure_trainable_modules(
    backbone=True,
    geometry_head=True,
    camera_decoder=False,
    motion_decoder=False,
    tcp_tracker=True,
)
```

### 优化器和学习率

```python
num_train_epochs = 50
gradient_accumulation_steps = 2
mixed_precision = "bf16"
max_grad_norm = 1.0

lr_backbone = 1e-5
lr_head = 2e-5
lr_motion_decoder = 1e-5
lr_tcp = 1e-4
weight_decay = 0.01
warmup_steps = 1000
eta_min_factor = 0.1
```

backbone、DualDPT、共享 motion decoder 和 TCP 模块分别使用配置中的学习率。scheduler 在 warmup 后使用 cosine decay，最低降至初始学习率乘以 `eta_min_factor`。

### 损失权重

```python
depth_loss_weight = 1.0
ray_loss_weight = 1.0
loss_gamma = 1.0
loss_alpha = 0.2
depth_valid_range = 0.98
gradient_scales = 4

tcp_loss_weight = 1.0
tcp_point_scale = 0.1
tcp_virtual_point_radius = 0.03
tcp_rotation_weight = 0.5
tcp_temporal_weight = 0.2
tcp_gripper_weight = 0.2
tcp_velocity_scale = 1.0

# 首帧视觉 query
tcp_query_window_size = 3
tcp_query_initial_exact_ratio = 0.80
tcp_query_exact_ratio = 0.25
tcp_query_max_jitter_patches = 1.0
tcp_query_curriculum_warmup_ratio = 0.10
tcp_query_curriculum_transition_ratio = 0.20
```

总损失为 geometry objective 加权 TCP pose、temporal velocity 与 gripper
objective。TCP head 直接预测每帧相机坐标系的绝对位置与绝对旋转；position 使用
训练集左右臂 mean/std 标准化。`tcp_point_scale` 只归一化 loss 中的米制误差。

raw depth/ray L1 和 TCP metrics 只用于日志，不会再次加入 objective。

### 日志和 checkpoint

```python
output_dir = "outputs/4rc-robotwin-tcp-debug"
log_every_steps = 10
visualize_every_steps = 1000
checkpointing_steps = 5000
save_each_epoch = False
report_to = ["tensorboard", "wandb"]
```

两个日志后端使用相同的 `global_step`，在每次梯度累积完成后记录相同的 geometry/TCP loss 与指标、各参数组学习率以及各数据源累计 batch 数。W&B 项目名为 `4RC-RoboTwin-TCP`，同时记录训练配置。`log_every_steps` 控制终端进度条刷新频率；深度预览图仍按 `visualize_every_steps` 保存在本地 `visuals/` 目录。

checkpoint 保存模型、AdamW、scheduler、随机数状态、epoch、batch 位置和 global step，可以精确恢复到下一个 batch。

## 5. 启动训练

单卡：

```bash
conda activate 4rc
accelerate launch --num_processes 4 train_4rc_stage1.py \
  --config configs/train/4rc-giant-train.py \
  --data-root datasets/RoboTwin
```

多卡：

```bash
accelerate config
accelerate launch train_4rc_stage1.py \
  --config configs/train/4rc-giant-train.py
```

本地预训练权重：

```bash
accelerate launch train_4rc_stage1.py \
  --config configs/train/4rc-giant-train.py \
  --pretrained-model /path/to/model.safetensors
```

不指定 `--pretrained-model` 时，读取配置里的 `pretrained_model = "Luo-Yihang/4RC"`。如果权重未缓存，会由 Hugging Face 下载。

## 6. 快速检查

先用一条 episode 和一个 optimizer step 检查完整流水线：

```bash
accelerate launch --num_processes 1 train_4rc_stage1.py \
  --config configs/train/4rc-giant-train.py \
  --max-episodes 1 \
  --max-train-steps 1
```

也可以运行单元测试：

```bash
pytest -q tests/test_tcp_pipeline.py
```

## 7. 恢复训练

混合数据训练配置支持取消 epoch 上限，只按累计 optimizer step 结束：

```python
resume = "outputs/4rc-robotwin-mixed-tcp-point-query/final_checkpoint"
num_train_epochs = None
max_train_steps = 100_000
```

在已激活的 `4rc` 环境中，保持与原训练相同的四卡配置：

```bash
accelerate launch --multi_gpu --num_processes 4 train_4rc_stage1.py \
  --config configs/train/4rc-giant-train-mixed.py
```

`max_train_steps` 是包含已完成步数的累计目标，不是额外训练步数。例如从
`global_step=42050, epoch=50, batch_in_epoch=0` 恢复，会从第 51 个 epoch
继续，最多再进行 57,950 次 optimizer 更新。`num_train_epochs=None` 时必须设置
正整数 `max_train_steps`；两个上限均为整数时，仍然在任一上限达到时停止。
恢复会加载模型、optimizer、scheduler 和随机数状态，不需要先下载基础预训练模型。
输出目录维持配置中的路径；结束时会更新该目录下的 `final_checkpoint`。

恢复后会以 `trainer_state.json` 的 `global_step` 显式校准 scheduler 计数和
optimizer 的实际 LR（包含 Accelerate 的多进程步数倍率）。`warmup_steps=1000`
仍表示整个训练最初的 1,000 步；从 42,050 步恢复直接进入 cosine 阶段，
不会重新 warmup。启动日志会打印 `Resume LR aligned`、warmup 剩余步数及每组 LR。
保持当前 100,000 步总目标时，42,050 步对应 backbone LR 约 `6.69225e-6`、
geometry head LR 约 `1.33845e-5`、TCP LR 约 `6.69225e-5`。

从完整 Accelerate 状态继续训练：

```bash
accelerate launch train_4rc_stage1.py \
  --config configs/train/4rc-giant-train.py \
  --resume outputs/4rc-robotwin-tcp-debug/checkpoint-5000
```

从旧模型复用 backbone/geometry/motion 权重并重新训练视觉 TCP 模块：

```bash
accelerate launch train_4rc_stage1.py \
  --config configs/train/4rc-tcp-recovery.py
```

命令行中的 `--data-root`、`--pretrained-model`、`--resume`、`--output-dir`、`--max-episodes`、`--num-train-epochs` 和 `--max-train-steps` 会覆盖配置文件。其他参数直接修改配置文件。

## 8. 固定外参和显存注意事项

RoboTwin 默认使用 `third_views` 标定。超过 3 米的像素只会退出 depth 与 gradient 监督；ray 仍在完整非 padding 区域学习相机模型。日志中的 `metric_ray_temporal_std` 可用于观察 ray 的时序变化。

camera decoder 与 dense track head 默认冻结；共享 motion decoder 会随视觉 TCP tracker 一起训练。

真实 `N=2` backward 已验证可运行，但 AdamW 首次 step 会额外创建 optimizer states，长序列也会增加激活显存。如果单卡显存不足，应降低调试配置的 `max_views`，正式复现则使用 Accelerate FSDP/多卡，不要静默改变最终的 `2～18` 采样范围。

## 9. 导出 RoboTwin GT 可视化

“arc/datasets/robotwin.py” 参考 Depth-Anything-Next 的数据集调试入口，
可以把采样到的 RGB、深度、内参和外参反投影为带颜色的点云并导出 GLB。
导出的内容是数据集 GT，不需要加载 4RC 模型。

在仓库根目录运行：

```bash
python arc/datasets/robotwin.py \
  --data-root datasets/RoboTwin \
  --index 0 \
  --num-views 4 \
  --max-points-per-view 50000 \
  --output outputs/robotwin-sample.glb
```

参数说明：

- “--index”：episode 索引。
- “--num-views”：按 4RC 时序采样规则抽取的帧数。
- “--max-points-per-view”：每帧最多写入多少个点，用于控制 GLB 大小。
- “--camera-size”：GLB 中相机锥体的大小，默认 0.05。
- “--no-cameras”：只导出点云，不显示相机。
- “--seed”：控制帧采样和点云下采样，固定后可重复导出。

也可以在 Python 中直接调用：

```python
from arc.datasets import RoboTwin4RC, visualize_scene

dataset = RoboTwin4RC(
    "datasets/RoboTwin",
    min_views=4,
    max_views=4,
    augment=False,
)
visualize_scene(
    dataset,
    index=0,
    output_path="outputs/robotwin-sample.glb",
    max_points_per_view=50_000,
)
```

数据仍使用原生 320×240 图像，并按训练数据流反射 padding 到 322×252。
可视化使用同步平移后的主点 “(cx+1, cy+6)”，不会对图像或深度另行缩放。

## 10. 上传 RoboTwin 到 Hugging Face

“scripts/upload_robotwin_to_hf.py” 默认上传到 dataset
“HarrisonPENG/4RC-Action”。由于当前 RoboTwin 约 118 GB、包含约 470 万个
小文件，脚本不会把这些文件直接提交到 Hub，而是为每个 task 创建一个未压缩
tar，上传到：

```text
RoboTwin/<task>.tar
```

每个 tar 内保留 “<task></task>/<episode></episode>/...” 原始路径。归档逐个生成和上传；
上传成功后默认删除对应临时 tar，因此只需要容纳最大单个 task 的临时空间。

先检查任务列表，不创建归档也不连接 Hub：

```bash
python scripts/upload_robotwin_to_hf.py --dry-run
```

安全地输入并显式传递具有 dataset 写权限的 token：

```bash
read -rsp "HF token: " HF_TOKEN
export HF_TOKEN
python scripts/upload_robotwin_to_hf.py --token "$HF_TOKEN"
unset HF_TOKEN
```

只上传一个 task：

```bash
python scripts/upload_robotwin_to_hf.py \
  --token "$HF_TOKEN" \
  --task adjust_bottle
```

如果仓库尚不存在，脚本会创建 dataset repo；添加 “--private” 可将新仓库
设为私有。中断后直接重新执行即可：远端已经存在的 task 会跳过，上传失败时
完整的本地 tar 会保留并在下次复用。若默认 staging 磁盘空间不足，可指定：

```bash
python scripts/upload_robotwin_to_hf.py \
  --token "$HF_TOKEN" \
  --staging-dir /path/to/large/disk/hf-upload-robotwin
```

“--keep-archives” 会保留上传成功的 tar；“--overwrite” 会重新上传远端已存在
的 task；“--rebuild-archives” 会重新创建 staging 中已有的 tar。


## 11. 第二阶段：重建条件下的 TCP 动作生成

训练入口分为 train_4rc_stage1.py 和 train_4rc_stage2.py。第一阶段继续使用
现有 geometry + TCP recovery 配置；第二阶段使用
configs/train/4rc-stage2-action.py，联合优化历史重建与未来动作生成。

### 启动、初始化与恢复

~~~bash
# 第一阶段
accelerate launch train_4rc_stage1.py \
  --config configs/train/4rc-giant-train-mixed.py

# 使用第一阶段 checkpoint 初始化第二阶段
accelerate launch train_4rc_stage2.py \
  --stage1-checkpoint outputs/4rc-robotwin-mixed-tcp-point-query/final_checkpoint \
  --batch-size 1

# 恢复第二阶段
accelerate launch train_4rc_stage2.py \
  --resume outputs/4rc-stage2-action/checkpoint-1000
~~~

stage1_checkpoint 在新训练时必须提供，支持权重文件或 Accelerate checkpoint
目录。也可直接使用本地 checkpoints/RoboTwin-TCP-Tracking/model.safetensors。
阶段切换只加载模型，重新创建优化器和 scheduler；恢复第二阶段则恢复完整
训练状态，未指定 --config 时默认读取 checkpoint 内保存的配置。

已有 TCP recovery head 的 position mean/std 保留 checkpoint 的值，新
action head 使用单独的训练集统计量。Safetensors 加载会正确恢复 DualDPT
共享 LayerNorm 的别名，真实缺失或形状不匹配的参数仍会报错。

### 历史长度与每卡 batch size

~~~python
history_frames = 8
prediction_horizon = 16
batch_size = 1
~~~

batch_size 是每张 GPU 的 clip 数。脚本自动计算
train_batch_images = batch_size * history_frames，并固定 scene_counts=(batch_size,)。

| 每卡 batch size | RGB batch | 历史图像数 |
|---|---|---|
| 1 | [1,8,3,252,322] | 8 |
| 2 | [2,8,3,252,322] | 16 |

有效 batch size 为每卡 clip 数 × GPU 数 × gradient accumulation steps。
历史窗口固定为配置长度；未来只读取 TCP 标签和坐标转换所需外参，不读取
未来 RGB/depth。窗口正序、连续且不跨异常轨迹分段，默认需要至少 9 个连续
有效帧（8 帧历史 + 至少 1 步未来标签）。不足 16 步的未来动作重复最后一个
有效动作补齐，并通过 future_action_valid 屏蔽 padding 的 attention 和 loss；
归一化统计及验证指标仅计真实未来标签，FDE 取最后一个有效步。未来时间仍
按采样频率递增，补齐不会跨越异常轨迹分段。--history-frames 和 --prediction-horizon 可用于配置新实验。

### 历史 token、编码与 DiT

每次前向先恢复历史 geometry/TCP，再从各历史帧的最后一层 backbone 全局
特征中，在左右 TCP 中心分别采样 3×3 patch，复用现有 visual query encoder。

邻域特征投影到 768 维后，以中心 patch 为 query、9 个 patch 为 key/value，
做 attention pooling，再加中心残差和 LayerNorm。每帧保留左右臂各一个
token，8 帧共 16 个历史 tokens。

池化结果加入 TCP 中心的二维位置、左右臂身份、历史/未来类型、投影有效性
及相对当前帧的物理时间编码。二维位置使用归一化 padded 图像坐标的
sin/cos 编码加 MLP；物理时间使用独立 sin/cos 编码加 MLP，以 1/15 秒为
单位。连续 15 Hz 数据的历史编码为 [-7,...,0]，未来为 [1,...,16]。

Stage2 配置默认使用 20 层、768 维、12 heads 的 DiT，约 297M 参数（不含条件编码模块）。主序列为 16 个历史 tokens 加 16 个未来
动作 tokens。历史只能读取历史；未来能读取全部历史和未来。仅未来动作
加噪、接受生成时间 tau 的 AdaLN-Zero 调制，并计算 flow matching loss。
物理时间和生成时间使用独立编码模块。

T5-base encoder 参数冻结且保持 eval。逐 token 输出经可训练 768→768
投影进入每层 cross-attention，并传递文本 padding mask。

每步每臂输出 xyz（3）+ rotation 6D（6）+ gripper（1），双臂共 20 维。
位置使用 action 统计量标准化；旋转输出转成矩阵；夹爪训练为关闭 −1、
打开 +1，推理以零为阈值。未来位姿统一表示在当前观测相机坐标系中。

### Teacher forcing 与学习率

训练历史采样中心按 clip 在 GT 与当前模型预测 TCP 投影之间随机选择，整段
历史使用同一种来源。GT 概率从 history_tcp_gt_initial_ratio=1.0 线性下降到
history_tcp_gt_final_ratio=0.5：首个更新 100% GT，最后一个更新 50% GT /
50% 预测，中途约 75% GT。比例按累计 optimizer step 计算，续训不会重新开始；
梯度累积期间概率相同，每个 micro-batch 独立选择。单步训练使用初始比例。
初始恢复 query 与逐帧生成采样点分别管理，此课程不改变初始 query。
GT 历史完全无效的窗口仍不训练；预测投影无效时使用缺失 token，不回退到 GT。
日志 condition/history_gt_probability 记录计划概率，history_gt_fraction 记录
当前 micro-batch 实际 GT 比例，history_valid_fraction 记录混合后的有效投影比例。
本课程不额外加入位置扰动。

推理用首帧左右两个二维点启动 TCP 恢复，随后根据恢复位置与标定内参投影。
所有点和内参使用 padded 图像坐标：原图点增加 (1,6)，内参主点增加同样
偏移；不要重复添加 padding。

联合损失包括 geometry、TCP recovery 和 action flow matching，三个外部
权重默认均为 1。生成损失更新池化、共享 query encoder 和 backbone；
GT 与 detach 后的预测采样中心均不提供通向 TCP 位置预测 head 的梯度，该 head 由恢复损失更新。

| 参数组 | 训练开关 | 学习率 |
|---|---|---|
| backbone | train_backbone | lr_backbone |
| geometry head | train_geometry_head | lr_head |
| sparse motion decoder | train_motion_decoder | lr_motion_decoder |
| 共享 visual query encoder | train_query_encoder | lr_query_encoder |
| TCP recovery head | train_tcp_head | lr_tcp_head |
| 历史池化、位置/时间编码 | train_history_pool | lr_history_pool |
| DiT | train_action_head | lr_action_head |
| 文本投影 | train_language_projection | lr_language_projection |

旧模块学习率继承 mixed 配置，新模块默认 1e-4。关闭开关或把该组学习率设为
0 会冻结参数。camera decoder 和 dense track head 不参与第二阶段训练。

### 验证与推理接口

默认按完整 episode 确定性留出 10% 验证数据，划分写入 data_manifest.json。
stage1_validation_overlap 默认 unknown；需审计第一阶段是否使用过这些
episode，才能将结果解释为未见 episode 泛化。

~~~bash
python train_4rc_stage2.py \
  --resume outputs/4rc-stage2-action/final_checkpoint \
  --eval-only --validation-batches 16
~~~

验证结果写入 validation/step-XXXXXXXX.json，包括：

- recovered：首帧 query 初始化后使用恢复投影，为主要推理结果；
- teacher_forced：使用全部历史 GT 投影，量化定位误差带来的差距；
- shuffled_instruction：使用其他任务的 instruction，保持相同动作初始噪声；
- 位置 ADE/FDE、旋转角误差、夹爪 F1/准确率、恢复失败率、重建指标和耗时。

只有一个任务时不报告跨任务指令打乱结果。小规模调试没有验证 episode 时
仍可训练，但 --eval-only 要求存在验证数据。sampling_steps 默认 8，可通过
配置比较 4/8/16 步。

TCPActionPolicy.sample_actions 接收 images、instructions、initial_query_points、
frame_times、intrinsics。images 为 padded 的 [B,K,3,H,W]、范围 [-1,1]；
frame_times 为 [B,K]；intrinsics 为 [B,K,3,3]。不接收历史 GT 轨迹。

输出包括 success、action_position [B,16,2,3]、action_rotation [B,16,2,3,3]、
action_gripper [B,16,2] 和未来时间。失败样本的 success 为 false，
位置/旋转为 NaN，夹爪为 −1。重建、池化和 T5 编码在一次生成中只执行一次。

~~~bash
python -m pytest tests/test_action_policy.py tests/test_action_dataset.py -q
~~~
