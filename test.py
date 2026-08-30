import torch

from arc.models.arc.arc import Arc
from arc.dust3r.inference_multiview import inference
from arc.dust3r.utils.image import load_images

# 优先使用 GPU；没有可用的 CUDA 设备时自动退回 CPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 从 Hugging Face 加载预训练的 4RC 权重，并将模型移动到计算设备。
model = Arc.from_pretrained("Luo-Yihang/4RC").to(device)
# 切换到评估模式，关闭训练阶段才需要的行为（例如 Dropout）。
model.eval()

example_dir = "examples/robot_arm"
# 读取目录中的多视角图像，将其缩放到推理尺寸并整理为模型需要的输入字典。
# patch_size=14 需要与 Arc.PATCH_SIZE 保持一致。
images = load_images(example_dir, size=512, patch_size=14, verbose=True)

# 推理时不构建反向传播所需的计算图，以降低显存占用和计算开销。
# inference() 本身也带有 @torch.no_grad()，这里保留外层上下文可明确表达用途。
with torch.no_grad():
    # 推理过程大致为：
    # 1. 将多张图像合并成一个 batch，并移动到 device；
    # 2. 主干网络提取多视角特征；
    # 3. 深度、相机和运动分支分别预测几何信息与跨帧轨迹；
    # 4. 后处理得到每个视角的三维点、置信度、轨迹及相机内外参。
    predictions, profiling = inference(
        images,
        model,
        device,
        # 使用 bfloat16 自动混合精度；部分数值敏感的模块仍会以 float32 运行。
        dtype="bf16-mixed",
        # 开启性能统计。当前实现返回模型 forward 总耗时，不会改变预测内容。
        profiling=True,
        # 打印图像加载和推理过程的提示信息。
        verbose=True,
        # False 表示保持原始视角顺序，并默认以第一张图像作为参考视角。
        # True 则会临时把中间图像移动到首位作为参考，推理后再恢复原顺序。
        use_center_as_anchor=False,
    )

raise
# predictions 是按视角组织的预测结果，主要包含三维点、置信度、运动轨迹和相机参数。
# profiling 是性能统计字典；当前仅包含 total_time，单位为秒。
