## 下载数据和模型
```bash
mkdir checkpoints
hf download HarrisonPENG/4RC-Action-RoboTwin-Stage1 --local-dir checkpoints/RoboTwin-Stage1
```

```bash
mkdir datasets
hf download HarrisonPENG/4RC-Action --local-dir datasets --repo-type dataset
```

删除 RoboTwin/put_object_cabinet.tar 这个任务 然后解压其他的tar包

## 预下载 DINOv2 Giant 腕部权重

以下命令均在 4RC 仓库根目录执行。这里的 DINOv2 是左右腕部共享的独立冻结
encoder；第三人称 4RC backbone 的权重仍由 `stage1_checkpoint` 加载。

### 方式一：下载到独立本地目录（推荐）

在有网络的机器上提前下载官方 `facebook/dinov2-giant`：

```bash
conda activate 4rc
mkdir -p checkpoints/dinov2-giant
hf download facebook/dinov2-giant config.json model.safetensors \
  --local-dir checkpoints/dinov2-giant \
  --max-workers 2
```

下载后的目录应包含：

```text
checkpoints/dinov2-giant/
├── config.json
└── model.safetensors
```

权重文件约 4.55 GB；网络中断后可以重新执行下载命令。需要迁移到训练服务器时，
复制整个 `checkpoints/dinov2-giant` 目录，保留配置文件与权重文件。

在 `configs/train/4rc-stage2-action-wrist.py` 中设置：

```python
use_wrist_condition = True
wrist_model = "checkpoints/dinov2-giant"
wrist_cache_dir = ".cache/huggingface"
```

`wrist_model` 也可以填写本地目录的绝对路径；多卡进程需要能访问同一目录。
使用本地目录时，DINOv2 直接从该目录读取，`wrist_cache_dir` 不需要改为模型目录。
当前接口要求 Transformers 格式的模型目录，不直接接受单个官方 `.pth` 文件，
也不能将 4RC Stage1 的 `model.safetensors` 当作这套独立腕部权重。

### 方式二：提前下载到 Hugging Face 缓存

如果希望配置中保留模型 ID，使用与 `wrist_cache_dir` 一致的缓存路径：

```bash
conda activate 4rc
hf download facebook/dinov2-giant config.json model.safetensors \
  --cache-dir .cache/huggingface \
  --max-workers 2
```

对应配置为：

```python
use_wrist_condition = True
wrist_model = "facebook/dinov2-giant"
wrist_cache_dir = ".cache/huggingface"
```

`--cache-dir` 使用 Hub 的 `blobs`、`snapshots`、`refs` 目录结构，
与方式一的 `--local-dir` 不同。迁移缓存时应完整复制 `.cache/huggingface`，
保留符号链接，且保持配置中的缓存路径一致。

### 完全离线训练还需要预下载 T5

Stage2 的文本分支也会加载 T5-base 与 tokenizer。若训练服务器没有外网，
可以在有网络的机器上下载以下文件，再复制 `checkpoints/t5-base` 整个目录：

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google-t5/t5-base",
    local_dir="checkpoints/t5-base",
    allow_patterns=["*.json", "*.safetensors", "spiece.model"],
    max_workers=2,
)
PY
```

随后在训练配置中设置：

```python
t5_model = "checkpoints/t5-base"
```

Stage1 checkpoint 和数据集也应按上面的步骤提前下载到本地。
下载参数参考 [Hugging Face CLI 文档](https://huggingface.co/docs/huggingface_hub/guides/cli)
及 [文件下载文档](https://huggingface.co/docs/huggingface_hub/guides/download)。


## 修改训练配置

在[4rc-stage2-action-wrist](configs/train/4rc-stage2-action-wrist.py) 中 修改下面的参数

- batch_size = 4  # Per GPU; each sample contains history_frames input images. 我这默认4，我也不知道H20能开多少，可以试下；
- output_dir = "outputs/4rc-stage2-action-clean-wrist" 这里是输出logs， checkpoints保存放的地方


## 启动训练

使用 accelerate 启动训练，下面以单卡为例：

```bash
conda activate 4rc
accelerate launch --num_processes 4 train_4rc_stage2.py \
  --config configs/train/4rc-stage2-action-wrist.py
```


`--num_processes 4` 表示使用张 GPU；单机多卡时改为实际使用的 GPU 数量。
首次从 Stage1 启动需要上述 DINOv2 预训练文件。新腕部 Stage2 checkpoint 已保存
冻结 DINOv2 的权重和架构，使用其初始化或 resume 时不会重新下载 DINOv2；
T5 模型及 tokenizer 仍需本地目录或完整缓存。
