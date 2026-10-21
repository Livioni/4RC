# Stage 1 深度、PCA 和点云可视化

在仓库根目录运行：

```bash
conda run --no-capture-output -n 4rc python visualizations/visualize_droid_stage1.py
```

默认输入为 `datasets/droid_eval_sets/AUTOLab__Mon_Aug_21_14:20:05_2023`，
相机为 `22008760`，权重为 `checkpoints/Droid-Stage1/checkpoint-500000`。
使用当前 `4rc` 环境中的 PyTorch、NumPy、Pillow、matplotlib 和 scikit-learn。

连续最多 32 帧为一组，组间不重叠，末尾不足 32 帧也会处理。该数据共 158 帧，
默认输出 5 组、158 套图片、316 个逐帧 PLY，以及 20 个合并 PLY。
逐帧文件为 `groups/<起始帧>_<结束帧>/gt_<帧号六位>.ply` 和 `pred_<帧号六位>.ply`，
包含当帧场景点云、TCP 位置标记及姿态坐标轴。
使用 CUDA 时默认自动选择 bfloat16（设备支持时），ray 解算使用 float32。

## 输出

默认目录：`outputs/visualizations/droid_stage1/<episode>/22008760/`。

```text
rgb/000000.png
gt_depth/000000.png
pred_depth/000000.png
pca/000000.png
comparison/000000.png
groups/000000_000031/
    arrays.npz
    pca_parameters.npz
    gt_stride4.ply
    gt_stride8.ply
    pred_stride4.ply
    pred_stride8.ply
manifest.json
```

- `comparison` 每张图片横向排列 RGB、GT 深度、预测深度、PCA，并注明帧号和深度色标。
- 两种深度共享 0–3 米色标。无效深度为黑色，超范围值饱和显示；原始米制浮点深度保存在 `arrays.npz`，不按显示范围截断。
- PCA 通过 forward hook 取得 `model.backbone.pretrained.patch_embed` 的输出：每帧单独执行 14×14、步长 14 的卷积投影，得到 `299×1536` 的 patch 特征。捕获位置在添加位置编码、CLS/register/time token 以及所有 Transformer 注意力层之前。使用指定 checkpoint 的 patch embedding 权重及模型的正常输入预处理，不额外施加 LayerNorm。每帧的这些特征只由自身图像决定；深度、ray 和 TCP 由 32 帧联合推理得到。
- 每个窗口（最多 32 帧）独立拟合三维 PCA，并独立计算各通道 1%–99% 分位数显色范围。窗口内所有帧共享投影与色标，不同窗口的颜色不能直接比较。每个窗口将 `当前组帧数×299` 个 patch 的 1536 维特征中心化，投影到前三个主成分，再分别映射为 RGB。13×23 特征图双线性放大至 182×322，裁掉四周 1 像素后与 RGB 对齐。各组目录的 `pca_parameters.npz` 包含本窗口的均值、投影矩阵、解释方差比例和颜色范围。
- Patchify 特征主要反映局部颜色、边缘与纹理；其原始空间分辨率只有 13×23，放大显示不会增加小物体细节。
- GT 点云使用真实内外参；预测点云的内外参均来自预测 `ray` 和 `ray_conf`，不使用 camera decoder 或 GT 标定替代。两者使用米制机器人基座坐标系，顶点颜色取原始 RGB。
- PLY 默认保留 `0 < depth <= 3m` 的有限点，不做额外置信度过滤、随机抽样、ICP 或体素融合。运动物体在多帧合并后可能出现多个位置。
- `stride4` 合并组内 0、4、…、28 帧（8 帧）；`stride8` 合并 0、8、16、24 帧（4 帧）。抽帧仅影响场景点云，两种文件共享同一次 32 帧推理。TCP 轨迹始终使用当前组全部帧，包含最后一帧。
- `arrays.npz` 包含原始帧号、GT/预测深度、预测深度置信度、GT/预测内参及 camera-to-base 外参。内参均对应裁剪后的原始 320×180 图像。另存 `gt_tcp_camera`、`pred_tcp_camera`、`gt_tcp_base`、`pred_tcp_base`、`pred_tcp_confidence` 和原始图像坐标下的 `tcp_query_pixel`。
- TCP 姿态保存为 `gt_tcp_rotation_camera`、`gt_tcp_rotation_base`、`pred_tcp_rotation_camera`、`pred_tcp_rotation_base`，形状均为 `[当前组帧数,3,3]`；各列分别表示局部 X/Y/Z 轴在对应坐标系中的方向。
- `manifest.json` 记录运行状态、参数、跳过的帧及每个 PLY 的实际帧号和点数。只有 `status=complete` 表示全部输出已完成。

## PLY 中的 TCP 轨迹与姿态

- 每个窗口将首帧 GT TCP 通过真实内参投影为二维查询点（输入模型时补上 padding 偏移），由训练好的 TCP head 预测全部 32 帧的位置。GT 只提供初始查询；后续位置来自模型。首帧 TCP 不在画面内时会明确报错。
- 预测 TCP 原始位置在相机坐标系，逐帧用 ray 解算的 camera-to-base 外参变换，与该帧预测点云使用同一坐标系。GT TCP 使用真实外参变换。
- `pred_stride4/8.ply` 加入橙色预测 TCP 轨迹；`gt_stride4/8.ply` 加入紫色 GT TCP 轨迹。
- 仅在 stride 选中的帧画姿态坐标架：stride 4 为组内 `0,4,…,28`，stride 8 为 `0,8,16,24`。X 轴红色、Y 轴蓝色、Z 轴绿色，默认轴长 3 cm、轴杆半径 1 mm，端部带圆锥箭头。坐标架是彩色点，普通 PLY 点云查看器即可显示。
- 预测旋转直接取 TCP head 的 `tcp_rotation`，按 `R_base_tcp = R_base_camera @ R_camera_tcp` 转换到基座系。GT 使用标注的弧度 RPY，按 `Rz(yaw) @ Ry(pitch) @ Rx(roll)` 转为相机系旋转，再乘真实 camera-to-base 旋转。轴原点为对应帧的 TCP 位置，方向为旋转矩阵各列。
- 相邻帧位置做分段线性插值，默认最大采样间距 2 mm。插值中心线周围生成半径 3 mm 的管状点，每帧原始位置增加半径 4.5 mm 的球面点标记。它们都是 PLY 的 XYZ/RGB 顶点，无需查看器支持线段或额外叠加文件。
- 线性插值用于显示采样点之间的路径，不增加真实运动观测；不连接无效 GT 点两侧的轨迹。静止点也能正常显示。
- PLY 按场景点、轨迹和标记点、姿态坐标架点的顺序写入。`manifest.json` 分别记录 `scene_vertex_count`、`tcp_vertex_count`、`pose_axis_vertex_count`、全部轨迹帧号和 `pose_frame_indices`，便于区分这些部分。
- 通过 `--tcp-spacing` 调节插值间距，通过 `--tcp-radius` 调节粗细，单位均为米。使用点云查看器时启用 RGB 顶点颜色。
- `--tcp-axis-length` 控制轴长，`--tcp-axis-radius` 控制轴杆半径，单位为米；半径需小于轴长的 1/6。

## 单独导出 TCP 点云叠加层

已有完整 PLY 结果后，无需重新推理：

```bash
conda run --no-capture-output -n 4rc python visualizations/export_tcp_only.py
```

默认写入当前结果目录的 `tcp_only/<起始帧>_<结束帧>/`，每组包含：

- `gt_trajectory.ply`、`pred_trajectory.ply`：仅插值轨迹及位置标记。
- `gt_tcp_stride4.ply`、`gt_tcp_stride8.ply`、`pred_tcp_stride4.ply`、`pred_tcp_stride8.ply`：完整轨迹加对应 stride 帧的姿态坐标架。

4 组共 24 个 PLY，不含场景点。顶点直接从原始 PLY 分离，XYZ 和 RGB 完全一致；
均为米制机器人基座坐标系，与同组场景加载后即可叠加，不需要平移、缩放或重新配准。
GT 轨迹紫色，预测轨迹橙色，XYZ 轴依次为红、蓝、绿。
可以通过 `--input` 选择其他完整结果目录，通过 `--output` 选择空输出目录。

另存不含轨迹、位置标记或坐标架的纯场景 PLY：

```bash
conda run --no-capture-output -n 4rc python visualizations/export_tcp_only.py --scene-only
```

结果在 `scene_only/<起始帧>_<结束帧>/`，每组为 `gt_stride4/8.ply` 和
`pred_stride4/8.ply`，文件数为组数的 4 倍。把同组、同来源的纯场景与 `tcp_only` 文件
共同加载即可叠加；使用 `*_trajectory.ply` 只显示轨迹，使用 `*_tcp_stride4/8.ply`
则同时显示轨迹和相应帧的姿态坐标架。原始合并版 PLY 保留在 `groups/`。

## 参数与重复运行

输出目录必须为空，防止混入不同帧段或不同 PCA 投影的图片。重复运行请指定新的输出目录：

```bash
conda run --no-capture-output -n 4rc python visualizations/visualize_droid_stage1.py \
  --input datasets/droid_eval_sets/AUTOLab__Mon_Aug_21_14:20:05_2023 \
  --camera 22008760 \
  --checkpoint checkpoints/Droid-Stage1/checkpoint-500000 \
  --output outputs/visualizations/droid_stage1_trial \
  --start-frame 0 --max-frames 32 \
  --device cuda:0 --dtype bfloat16 --max-depth 3 --seed 0 \
  --tcp-spacing 0.002 --tcp-radius 0.003 \
  --tcp-axis-length 0.03 --tcp-axis-radius 0.001
```

`--max-frames 0` 表示选择全部剩余帧；不足 32 帧的尾组也会处理并逐帧导出。
`--max-depth` 同时控制深度色标上限和点云深度过滤上限。
固定 seed 控制 PCA 和 ray 解算的随机采样；每个窗口单独拟合，
更改其他窗口不会影响当前窗口的 PCA，改变窗口内的帧则可能改变颜色。

运行针对几何和导出逻辑的测试：

```bash
conda run -n 4rc python -m pytest visualizations/test_stage1_visualization.py -q
```
