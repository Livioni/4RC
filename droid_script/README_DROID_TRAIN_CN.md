# 4RC 在 DROID 上的单目、单臂两阶段训练

本文面向第一次使用本项目的同学，包含环境安装、数据划分、索引准备、训练、恢复和验证。**所有命令均在仓库根目录 `4RC/` 执行，使用 `4rc` Conda 环境。**

本功能需要这份包含 DROID 修改的完整仓库；仅克隆上游原始 4RC 不包含下面的脚本。共享模型也做了单臂兼容修改，分享时请一并提供 `arc/`、`configs/` 和 `droid_script/`。

## 1. 两个阶段训练什么

| 项目 | 第一阶段 | 第二阶段 |
|---|---|---|
| 输入 | 单台相机的时序 RGB、首帧一个 TCP 二维 query | 单台相机的历史 RGB、首帧一个 TCP 二维 query、instruction |
| 默认时间窗口 | 2–18 帧，间隔 1–5，可反向 | 连续 8 帧历史，预测未来 16 帧 |
| 训练目标 | 深度、ray、绝对相机外参、单臂 TCP 恢复 | 继续训练上述目标，并训练语言条件下的未来 TCP 生成 |
| 相机外参 | 机器人基座到相机的绝对变换 `world-to-camera` | 同左 |
| TCP/未来动作坐标 | 所选相机的 OpenCV 坐标 | 所选相机的 OpenCV 坐标 |

每个 episode 有两台第三人称相机，数据加载器将它们作为两条独立的单目序列采样。一个 clip 内始终使用同一台相机，不把两路图像拼成双目输入，也不把它们当成两只手臂。

**这份 DROID 导出固定为 15 fps。** 时间直接使用 `frame_index / 15`；反向片段的时间也相应递减。加载器不读取相机时间戳来估计帧率。

外参和 ray 都在机器人基座坐标下监督，不将首帧外参归一化为单位矩阵。相机 head 内部沿用原模型的 `camera-to-world` 编码，计算损失时转换成 `world-to-camera`。TCP 标签则直接读取对应相机的 `TCP/<camera>/state.npy`，避免重复坐标变换。

## 2. 代码位置

```text
configs/train/
├── 4rc-stage1-droid.py          # 完整 stage1 配置，布局与原 stage1 一致
└── 4rc-stage2-droid.py          # 完整 stage2 配置，布局与原 stage2 一致
arc/datasets/
├── droid.py                    # 单目 clip、单臂动作窗口与缓存统计
└── droid_index.py              # SQLite 索引、数据校验、TXT 划分
arc/loss/
└── droid_geometry.py           # 基座坐标 ray 与绝对相机监督
droid_script/
├── train_4rc_stage1.py
├── train_4rc_stage2.py
├── prepare_droid_dataset.py    # 准备索引与划分
├── check_data.py               # 不加载模型的数据检查
├── checkpoints.py             # 单臂权重迁移、配置与划分快照
├── requirements.txt
├── README_DROID_TRAIN_CN.md
├── splits/
│   ├── train_set.txt
│   ├── val_set.txt
│   └── split_report.json
├── tests/
├── cache/                     # 自动生成，不提交/分享缓存数据库
└── outputs/                   # 日志、预览、checkpoint
```

仓库根目录原来的两个训练入口仍用于原 RoboTwin 配置。**训练 DROID 请使用 `droid_script/` 下的入口。**

## 3. 安装 4RC 环境

### 3.1 获取代码并创建环境

先取得包含本次修改的仓库副本并进入根目录。上游地址为 <https://github.com/Luo-Yihang/4RC>，但还需要应用本次 DROID 修改。

沿用原 4RC README 的 Python 3.11 Conda 环境：

```bash
conda create -n 4rc python=3.11 cmake=3.14.0 -y
conda activate 4rc
python -m pip install --upgrade pip
```

已有 `4rc` 环境时只需激活，不要重新创建。用 `which python` 确认当前解释器来自该环境。

### 3.2 安装 PyTorch

GPU 训练先运行 `nvidia-smi` 确认驱动可用。下面沿用原项目的 PyTorch 2.8 / CUDA 12.6 示例，并固定匹配的 torchvision 版本：

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

其他 CUDA 构建请参照 [PyTorch 官方版本配对和安装命令](https://pytorch.org/get-started/previous-versions/#v280)，不要任意混装 torch/torchvision。

只有 CPU、仅检查数据和运行单元测试时，可改用：

```bash
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

CPU 构建不能用于 GPU 训练。之后切换 GPU 时需重新安装对应 CUDA 构建。

### 3.3 安装依赖和本项目

仅运行本文训练与测试，使用已经收集的训练依赖即可：

```bash
python -m pip install -r droid_script/requirements.txt
python -m pip install -e .
```

如果还需要原 4RC 的完整 demo、可视化或评测组件，原项目的安装步骤是：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

完整依赖包含较重的可视化及训练组件。用于 DROID 时仍需保证 `addict` 已安装，并使用本文列出的 Accelerate/Transformers 版本。训练依赖选择的是 headless OpenCV；已有 `opencv-python` 时无需同时安装两种 OpenCV 发行包。

检查关键模块：

```bash
python -c "import torch, torchvision, sqlite3, accelerate, transformers, addict; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python droid_script/train_4rc_stage1.py --help
python droid_script/train_4rc_stage2.py --help
```

## 4. 数据目录、单位与标签

默认根目录是 `datasets/droid_episodes`：

```text
datasets/droid_episodes/
└── <episode_name>/
    ├── metadata.json                  # frame_count、三个 language_instruction 字段
    ├── images/<camera_serial>/000000.png
    ├── depths/metadata.json
    ├── depths/<camera_serial>/000000.png
    ├── intrinsic/<camera_serial>.npy  # [3,3]
    ├── extrinsic/<camera_serial>.npy  # [4,4]，静态 world-to-camera
    └── TCP/<camera_serial>/
        ├── metadata.json
        └── state.npy                 # [T,7]
```

每个 episode 对应两个 `<camera_serial>`。相机编号在不同 episode 中可以变化，不写死编号。`observations/` 和 `action/` 可以保留，但本训练的动作目标来自已经转换好的 TCP 轨迹，不直接使用 `action/cartesian_position.npy`。

- RGB 原生为 320×180，不 resize 或 crop；四边各 pad 1 像素得到 322×182，使尺寸能被 patch size 14 整除。内参主点同步平移，padding 不参与几何监督。
- 深度保存为毫米，读取后除以 1000 转为米。默认屏蔽 0、非有限值和超过 `max_depth=3.0` 的像素；将 `max_depth=None` 可关闭距离上限。
- TCP 列为 `x,y,z,roll,pitch,yaw,gripper_open`，位置单位米、角度单位弧度；姿态使用 `Rz(yaw) @ Ry(pitch) @ Rx(roll)`。
- TCP 为 OpenCV 相机坐标：x 向右、y 向下、z 向前。夹爪是 `[0,1]` 连续开度，不提前二值化。
- 首帧 TCP 只用于生成二维 query 和监督标签，模型没有接收首帧完整 7D TCP 状态。

### 4.1 TCP 投影在相机画面外时如何处理

TCP 投影越界时，按相机和采样窗口分别处理，无需直接删除整个 episode。下面的“首帧”指**采样 clip 的第一帧**，不是 episode 的第 0 帧；第一阶段反向采样时，也以反向后的第一帧为准。

| 情况 | 当前处理 |
|---|---|
| Stage1 首帧 TCP 投影在画面外 | 设置 `tcp_query_valid=False`，该 clip 的 TCP 位姿、夹爪和时序损失全部屏蔽；深度、ray 和绝对外参仍正常训练。 |
| Stage1 / Stage2 首帧投影有效，后续 TCP 移出画面 | 仍使用有效的三维 TCP 标签监督轨迹恢复，不因后续帧投影越界而屏蔽该帧的 TCP 恢复损失。 |
| Stage2 历史首帧 TCP 投影在画面外 | 不采样这个窗口；某台相机没有任何合格窗口时跳过该相机序列，另一台相机独立判断。 |
| Stage2 后续历史帧用于动作条件的 TCP 投影越界 | 对应局部视觉特征替换为表示缺失的 `missing_token`，保留时间和有效性信息。有效性依据当前使用的真值轨迹或模型恢复轨迹判断。 |
| 未来动作的 TCP 投影在画面外 | 仍参与动作训练，因为相机坐标系下的三维标签仍有效。`future_action_valid` 屏蔽的是超出有效轨迹段的补齐部分，不按未来投影是否在图内过滤。 |

投影有效性检查包括坐标有限、相机坐标的 z 为正、二维投影落在原始图像区域内；四周 padding 不算有效区域。无效投影的二维坐标置零仅用于占位，配合有效性 mask 使用，不将越界点夹到图像边缘后当成有效标签。

**当前 Stage1 尚未强制首帧投影有效**，因此部分 clip 只贡献几何监督。若希望提高 TCP 训练效率，可进一步改为优先从投影有效的帧开始采样，将完全没有有效投影的相机序列保留用于几何训练；这项优先采样策略目前尚未启用。

这里的“投影有效”只表示点落在相机画面内，**没有判断 TCP 是否被物体遮挡**，不能将其等同于视觉上无遮挡。

对应实现见 [DROID 采样](../arc/datasets/droid.py)、[投影有效性检查](../arc/action.py)、[TCP 损失 mask](../arc/loss/tcp_tracking.py) 和 [历史条件缺失处理](../arc/models/arc/arc_action.py)。

## 5. TXT 划分与大数据索引

本次生成的完整清单包含 **30,629 个训练 episode、3,337 个验证 episode**，共 33,966 个。默认 seed 为 42，稳定哈希按约 90%/10% 划分；具体计数见 `splits/split_report.json`。

每个 TXT 一行一个相对于数据根目录的 episode 名称，不包含机器绝对路径、相机或帧号。两台相机一起划分，两个训练阶段共用同一对 TXT。

第一次在新机器上准备索引：

```bash
python droid_script/prepare_droid_dataset.py \
  --data-root datasets/droid_episodes
```

已有 TXT 时保持原清单；已有且匹配的索引也会直接复用。SQLite 保存到 `droid_script/cache/droid.sqlite`。索引逐 episode 校验标注、相机对应关系和首尾帧文件，不解码全部 PNG。完整数据的中间帧损坏会在采样到该帧时报告具体位置。

数据换了目录或修改了标注时，显式重建索引；已有 TXT 仍保持不变：

```bash
python droid_script/prepare_droid_dataset.py \
  --data-root /your/path/droid_episodes --rebuild-index
```

全新划分数据时才使用 `--overwrite-splits`，例如：

```bash
python droid_script/prepare_droid_dataset.py \
  --seed 42 --validation-fraction 0.1 --overwrite-splits
```

可以手动修改 TXT。加载时会检查重名、训练验证交叉、不存在的目录，以及清单与索引不一致的问题。修改后，训练统计会按新清单重新计算；**不能用改变后的划分恢复原训练 checkpoint**。纯空行和以 `#` 开头的注释行允许存在。

速度过滤在准备索引时执行，默认线速度上限 3 m/s、角速度上限 4π rad/s，异常跳变处切段。修改配置里的速度阈值时，应使用相同参数重建索引，例如关闭两项过滤：

```bash
python droid_script/prepare_droid_dataset.py --rebuild-index \
  --max-tcp-linear-speed none --max-tcp-angular-speed none
```

同时将两个训练配置中的对应阈值改为 `None`。索引参数和配置不一致会报错。

检查真实数据，不加载模型权重、不需要 GPU：

```bash
python droid_script/check_data.py --max-episodes 8 --samples 2
```

检查覆盖两个阶段的训练/验证样本，并打印图像和 TCP 形状、相机编号、有效深度和未来标签数量。

## 6. 第一阶段训练

修改 `configs/train/4rc-stage1-droid.py`。文件的分区与原 stage1 一致，包括 Common、Dataset、Model、Training、Optimizer、Scheduler、Loss、TCP curriculum 和 Logging。

默认从 `Luo-Yihang/4RC` 初始化，首次使用会下载权重。也可通过 `--pretrained-model` 指定本地 `model.safetensors`、PyTorch 权重文件或含权重的目录。

```bash
accelerate launch --num_processes 1 droid_script/train_4rc_stage1.py \
  --config configs/train/4rc-stage1-droid.py
```

四卡示例：

```bash
accelerate launch --multi_gpu --num_processes 4 droid_script/train_4rc_stage1.py \
  --config configs/train/4rc-stage1-droid.py
```


默认训练 backbone、geometry head、camera decoder、共享 motion decoder 和单臂 TCP 模块。`train_motion_decoder=False` 沿用原 stage1 的语义：关闭 dense tracking；共享 motion decoder 仍随 TCP tracker 训练。

## 7. 第二阶段训练与验证

修改 `configs/train/4rc-stage2-droid.py`，布局与原 stage2 一致，包括独立的 Validation 和 History TCP Condition Curriculum 分区。

第二阶段必须加载本文第一阶段生成的**单臂 DROID checkpoint**。脚本会检查坐标约定、臂数及 TXT 摘要，不会静默加载不匹配的双臂权重。

```bash
accelerate launch --num_processes 1 droid_script/train_4rc_stage2.py \
  --config configs/train/4rc-stage2-droid.py \
  --stage1-checkpoint droid_script/outputs/stage1/final_checkpoint \
  --batch-size 1
```

多卡时同样添加 `--multi_gpu --num_processes 4`。`batch_size` 是每张 GPU 的 clip 数，输入图像数为 `batch_size × history_frames`，不会再乘相机数。

默认使用冻结的 `google-t5/t5-base` 编码 instruction，首次运行会下载模型。离线使用时将配置中的 `t5_model` 改为包含权重和 tokenizer 的本地目录。

第一阶段 debug 跑通后，可以衔接第二阶段 debug：

```bash
accelerate launch --num_processes 1 droid_script/train_4rc_stage2.py \
  --config configs/train/4rc-stage2-droid.py \
  --stage1-checkpoint droid_script/outputs/stage1_debug/final_checkpoint \
  --max-episodes 8 --max-train-steps 2 --num-workers 0 --validation-batches 1 \
  --output-dir droid_script/outputs/stage2_debug
```

第二阶段只读取历史 RGB/深度，未来只读 TCP 标签。未来不足 16 帧时重复末状态用于存储，补齐部分不参与动作损失。窗口不能跨越轨迹异常切段点；首帧 TCP 必须能投影到图像内。

DiT 输入为 `[最后观测帧的整图 patch tokens | TCP 局部历史 tokens | 未来动作 tokens]`。
全局分支保留最后一层 backbone 全局特征的全部 patch，投影后加入 patch 中心二维位置、
相对时间 0 及独立可学习类型 embedding。三类类型 ID 为 TCP 历史 0、未来动作 1、全局视觉 2。
默认 182×322 图像对应 13×23 = 299 个全局 tokens，加 8 个单臂历史和 16 个未来动作，
共 323 个 tokens。全局/TCP 条件互相可见且不能读取未来动作；仅未来动作接受加噪和生成时间调制。
新增全局编码器与历史池化共用 train_history_pool / lr_history_pool；冻结开关也同时生效。
训练、验证和推理使用相同的条件构造逻辑，TCP 全无效时的失败判定保持不变。

新增全局编码器和第三类 embedding 后，旧 Stage2 checkpoint 无法直接加载。
请从 Stage1 开始新的 Stage2 实验并使用新的输出目录；新结构支持正常断点恢复。

动作内部为每步 10 维：位置 3、旋转 6D、连续夹爪 1。接口保留臂维度，未来标签为 `[B,H,1,10]`，输出位置为 `[B,H,1,3]`。连续夹爪输出使用 `action_gripper_open`，范围 `[0,1]`；`action_gripper` 仍提供二值开闭结果。

仅验证已有第二阶段 checkpoint：

```bash
python droid_script/train_4rc_stage2.py \
  --resume droid_script/outputs/stage2/final_checkpoint \
  --eval-only --validation-batches 16
```

验证记录 recovered、teacher-forced 和 shuffled-instruction 条件下的动作指标，包括位置 ADE/FDE、旋转误差、夹爪 MAE，以及历史深度、ray、TCP 和绝对相机误差。第一阶段入口沿用原训练流程记录训练指标，不额外执行完整验证循环。

## 8. 恢复、日志与大数据参数

恢复第一阶段：

```bash
accelerate launch --num_processes 1 droid_script/train_4rc_stage1.py \
  --resume droid_script/outputs/stage1/checkpoint-5000
```

恢复第二阶段：

```bash
accelerate launch --num_processes 1 droid_script/train_4rc_stage2.py \
  --resume droid_script/outputs/stage2/checkpoint-1000
```

恢复时自动读取 checkpoint 中的 `config.json`。`max_train_steps` 表示累计优化步数，可通过命令行提高目标，不会重新从第 0 步 warmup。为复现原采样顺序，保持 GPU 数、batch、采样窗口、seed 和梯度累积设置一致。

checkpoint 保存模型、优化器、调度器、RNG、训练位置、配置、TXT 副本和索引标识。重建索引后标识会变化，完整训练恢复会拒绝混用；需要开始新实验时使用权重初始化和新的输出目录。

| 参数 | 默认值 / 用途 |
|---|---|
| `num_workers` | 8；SSD 较慢或主机内存紧张时降低 |
| `prefetch_factor` | 每个 worker 预取 2 个 batch |
| `cache_size` | 每个 worker 最多缓存 8 条 mmap TCP 轨迹 |
| `batches_per_epoch` | 2,000，表示采样周期，不是完整遍历所有帧 |
| `max_episodes` | 默认无限制；调试时分别限制所选 train/val 清单，不修改 TXT |
| `max_train_steps` | stage1 100,000；stage2 500,000 |
| `mixed_precision` | 默认 bf16；CPU 测试使用 `no` |
| `camera_loss_weight` / `lr_camera` | 默认 1.0 / 2e-5，两个阶段都启用 |

数据加载使用 `spawn` worker，SQLite 连接按进程独立打开。只解码采样窗口，未来动作不加载图片。训练统计在首次使用相应窗口/划分时逐相机计算并缓存，内存不会随全数据集图片总量增长。

查看日志：

```bash
tensorboard --logdir droid_script/outputs
```

使用 W&B 时将 `report_to` 改为 `["tensorboard", "wandb"]`，再执行 `wandb login`；离线可以设置 `WANDB_MODE=offline`。`visuals/` 保存深度预览，`validation/` 保存第二阶段指标 JSON。

## 9. 测试与已验证范围

所有测试均使用 `4rc` 环境：

```bash
conda activate 4rc
python -m pytest -q droid_script/tests tests/test_action_dataset.py \
  tests/test_action_policy.py tests/test_training_resume.py \
  tests/test_stage2_inference.py -k 'not interactive_selection_without_truth' \
  -o cache_dir=droid_script/.pytest_cache
```

在已准备本地真实数据和索引的机器上，可额外运行：

```bash
DROID_RUN_REAL_TESTS=1 python -m pytest -q \
  droid_script/tests/test_real_droid.py \
  -o cache_dir=droid_script/.pytest_cache
```

上述回归及真实数据测试共 **85 项通过**；旧 Gradio 界面的交互测试未纳入本次训练适配验证。

真实数据测试使用小型 CPU backbone，以及缩小维度的生产 MotionDecoder、TCP、相机和动作模块，实际执行两个阶段的短训练、保存、恢复和验证，不下载 giant/T5 权重。它验证训练链路和数据契约，不代表完整 giant 模型的 GPU 吞吐、显存或收敛结果。

本次测试机器的 NVIDIA 驱动不可用，因此使用 `4rc` 中的 PyTorch 2.8.0 CPU 构建。完整 giant 模型 GPU 训练仍需在驱动正常的机器上运行。

## 10. 常见问题

- **找不到索引/TXT**：先运行数据准备脚本，确认工作目录是仓库根目录。
- **数据根目录不是默认位置**：准备索引和训练时都传相同的 `--data-root`，或修改两份配置的 `data_sources[0]["options"]["root"]`。
- **只有部分 DROID 数据**：自行提供只包含本机 episode 的 train/val TXT，两个阶段使用同一对文件；不要沿用包含未下载 episode 的完整清单。
- **某一划分没有合格窗口**：检查清单、instruction、episode 长度、首帧可见性和速度过滤；增加调试用的 `max_episodes`。
- **共享内存/worker 问题**：先使用 `--num-workers 0` 检查单进程链路，再检查主机共享内存和运行容器的进程通信权限。
- **CUDA 不可用**：先检查 `nvidia-smi`，再检查是否装了 CPU 版 torch。更换训练脚本不能修复驱动。
- **checkpoint 形状不匹配**：DROID 是原生单臂；双臂权重只能通过第一阶段的显式初始化迁移路径使用，不能作为第二阶段 DROID checkpoint 或完整 resume。
- **TXT 被修改后无法 resume**：使用 checkpoint 中 `splits/` 的原始清单，或创建新实验，避免混淆验证集合。

原有双臂滑窗推理应用没有在本次改动中扩展为 DROID 界面；本文覆盖训练、数据检查及训练入口提供的验证流程。
