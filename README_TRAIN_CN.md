# 4RC 在 RoboTwin 上的 Geometry 训练说明

本文档说明如何使用从 Depth-Anything-Next 移植的训练框架，训练 4RC 的 ViT backbone 和完整 DualDPT geometry head。

当前训练范围：

- 数据集仅使用 RoboTwin `head_view`；
- 训练 backbone、DualDPT depth 分支和 ray 分支；
- camera decoder、motion decoder、track head 默认冻结并跳过计算；
- 损失为 `Ldepth + Lray`；
- 暂不包含 point、camera、motion loss。

## 1. 代码布局

```text
4RC/
├── arc/
│   ├── datasets/
│   │   ├── robotwin.py          # RoboTwin dataloader、时序采样、collate
│   │   └── utils/geometry.py    # 尺度归一化、相对外参、ray GT
│   └── loss/
│       └── geometry.py          # depth、gradient、ray loss
├── configs/
│   └── train/
│       └── 4rc-giant-train.py   # 默认训练参数
└── train_4rc.py                 # Accelerate 训练入口
```

配置目录和参数写法参考 Depth-Anything-Next：训练配置放在 `configs/train/`，每个参数直接定义为 Python 顶层变量。

## 2. 环境准备

```bash
conda activate 4rc
pip install -r requirements.txt
pip install -e .
```

训练入口依赖 `torch`、`torchvision`、`accelerate` 和 `tensorboard`。如需使用 WandB，可另外把配置中的 `report_to` 改为 `wandb` 并登录 WandB。

## 3. RoboTwin 数据目录

默认数据根目录为 `datasets/RoboTwin`，需要满足：

```text
datasets/RoboTwin/
└── <task>/
    └── <episode>/
        ├── images/head_view/000000.png
        ├── depths/head_view/000000.png
        ├── intrinsics/head_view.npy
        └── extrinsics/head_view.npy
```

- RGB 必须是 `320×240`。
- depth 是 uint16 毫米值，读取后转换为米；0 表示无效。
- extrinsics 是逐帧 OpenCV world-to-camera，形状为 `[T,3,4]`。
- intrinsics 形状为 `[3,3]`。

原图不会 resize 或 crop。为满足 patch size 14，dataloader 会左右各反射填充 1 像素、上下各反射填充 6 像素，模型实际输入为 `322×252`；填充区不参与损失。

## 4. 修改训练参数

编辑：

```text
configs/train/4rc-giant-train.py
```

### 数据和采样

```python
data_root = "datasets/RoboTwin"
view = "head_view"
min_views = 2
max_views = 18
min_interval = 1
max_interval = 5
reverse_probability = 0.5
max_episodes = None
augment = True
num_workers = 8
train_batch_images = 18
scene_counts = (1, 2, 3, 6, 9)
batches_per_epoch = None
recent_buffer_size = 10_000
```

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
```

`train_motion_decoder=False` 会同时冻结 motion decoder 和 track head。训练前向还会跳过 camera/motion decoder，因此不只是关闭梯度，也能节省计算和显存。

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
weight_decay = 0.01
warmup_steps = 1000
eta_min_factor = 0.1
```

backbone 和 DualDPT 使用不同学习率。scheduler 在 warmup 后使用 cosine decay，最低降至初始学习率乘以 `eta_min_factor`。

### 损失权重

```python
depth_loss_weight = 1.0
ray_loss_weight = 1.0
loss_gamma = 1.0
loss_alpha = 0.2
depth_valid_range = 0.98
gradient_scales = 4
```

总损失为：

```text
objective = depth_loss_weight × (depth uncertainty loss + spatial gradient loss)
          + ray_loss_weight × ray uncertainty loss
```

raw depth/ray L1 只用于日志，不会再次加入 objective。

### 日志和 checkpoint

```python
output_dir = "outputs/4rc-robotwin-geometry"
log_every_steps = 10
visualize_every_steps = 1000
checkpointing_steps = 5000
save_each_epoch = False
report_to = "tensorboard"
```

checkpoint 保存模型、AdamW、scheduler、随机数状态、epoch、batch 位置和 global step，可以精确恢复到下一个 batch。

## 5. 启动训练

单卡：

```bash
conda activate 4rc
accelerate launch --num_processes 1 train_4rc.py \
  --config configs/train/4rc-giant-train.py \
  --data-root datasets/RoboTwin
```

多卡：

```bash
accelerate config
accelerate launch train_4rc.py \
  --config configs/train/4rc-giant-train.py
```

本地预训练权重：

```bash
accelerate launch train_4rc.py \
  --config configs/train/4rc-giant-train.py \
  --pretrained-model /path/to/model.safetensors
```

不指定 `--pretrained-model` 时，读取配置里的 `pretrained_model = "Luo-Yihang/4RC"`。如果权重未缓存，会由 Hugging Face 下载。

## 6. 快速检查

先用一条 episode 和一个 optimizer step 检查完整流水线：

```bash
accelerate launch --num_processes 1 train_4rc.py \
  --config configs/train/4rc-giant-train.py \
  --max-episodes 1 \
  --max-train-steps 1
```

也可以运行单元测试：

```bash
pytest -q tests/test_4rc_training.py
```

## 7. 恢复训练

```bash
accelerate launch train_4rc.py \
  --config configs/train/4rc-giant-train.py \
  --resume outputs/4rc-robotwin-geometry/checkpoint-5000
```

命令行中的 `--data-root`、`--pretrained-model`、`--resume`、`--output-dir`、`--max-episodes`、`--num-train-epochs` 和 `--max-train-steps` 会覆盖配置文件。其他参数直接修改配置文件。

## 8. 固定外参和显存注意事项

RoboTwin `head_view` 的内外参固定，因此 depth 分支仍能从变化的 RGB 和深度监督中学习，但 ray 分支可能收敛为固定相机的空间模板。日志中的 `metric_ray_temporal_std` 可用于观察这种特化。该训练结果适合相同 head camera，不应直接当作通用变相机模型。

虽然 camera/motion decoder 权重被冻结，但 backbone 会更新；以后重新启用这些 decoder 时，仍建议联合微调以适配新的 backbone 特征分布。

真实 `N=2` backward 已验证可运行，但 AdamW 首次 step 会额外创建 optimizer states，长序列也会增加激活显存。如果单卡显存不足，应降低调试配置的 `max_views`，正式复现则使用 Accelerate FSDP/多卡，不要静默改变最终的 `2～18` 采样范围。

## 9. 导出 RoboTwin GT 可视化

“arc/datasets/robotwin.py” 参考 Depth-Anything-Next 的数据集调试入口，
可以把采样到的 RGB、深度、内参和外参反投影为带颜色的点云并导出 GLB。
导出的内容是数据集 GT，不需要加载 4RC 模型。

在仓库根目录运行：

~~~bash
python arc/datasets/robotwin.py \
  --data-root datasets/RoboTwin \
  --index 0 \
  --num-views 4 \
  --max-points-per-view 50000 \
  --output outputs/robotwin-sample.glb
~~~

参数说明：

- “--index”：episode 索引。
- “--num-views”：按 4RC 时序采样规则抽取的帧数。
- “--max-points-per-view”：每帧最多写入多少个点，用于控制 GLB 大小。
- “--camera-size”：GLB 中相机锥体的大小，默认 0.05。
- “--no-cameras”：只导出点云，不显示相机。
- “--seed”：控制帧采样和点云下采样，固定后可重复导出。

也可以在 Python 中直接调用：

~~~python
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
~~~

数据仍使用原生 320×240 图像，并按训练数据流反射 padding 到 322×252。
可视化使用同步平移后的主点 “(cx+1, cy+6)”，不会对图像或深度另行缩放。

## 10. 上传 RoboTwin 到 Hugging Face

“scripts/upload_robotwin_to_hf.py” 默认上传到 dataset
“HarrisonPENG/4RC-Action”。由于当前 RoboTwin 约 118 GB、包含约 470 万个
小文件，脚本不会把这些文件直接提交到 Hub，而是为每个 task 创建一个未压缩
tar，上传到：

~~~text
RoboTwin/<task>.tar
~~~

每个 tar 内保留 “<task>/<episode>/...” 原始路径。归档逐个生成和上传；
上传成功后默认删除对应临时 tar，因此只需要容纳最大单个 task 的临时空间。

先检查任务列表，不创建归档也不连接 Hub：

~~~bash
python scripts/upload_robotwin_to_hf.py --dry-run
~~~

安全地输入并显式传递具有 dataset 写权限的 token：

~~~bash
read -rsp "HF token: " HF_TOKEN
export HF_TOKEN
python scripts/upload_robotwin_to_hf.py --token "$HF_TOKEN"
unset HF_TOKEN
~~~

只上传一个 task：

~~~bash
python scripts/upload_robotwin_to_hf.py \
  --token "$HF_TOKEN" \
  --task adjust_bottle
~~~

如果仓库尚不存在，脚本会创建 dataset repo；添加 “--private” 可将新仓库
设为私有。中断后直接重新执行即可：远端已经存在的 task 会跳过，上传失败时
完整的本地 tar 会保留并在下次复用。若默认 staging 磁盘空间不足，可指定：

~~~bash
python scripts/upload_robotwin_to_hf.py \
  --token "$HF_TOKEN" \
  --staging-dir /path/to/large/disk/hf-upload-robotwin
~~~

“--keep-archives” 会保留上传成功的 tar；“--overwrite” 会重新上传远端已存在
的 task；“--rebuild-archives” 会重新创建 staging 中已有的 tar。
