# Stage2 整段滑窗推理

`stage2_sliding_window_inference.py` 独立加载完整 Stage2 策略，不导入 Stage1 推理脚本或训练入口。
默认 checkpoint 为 `checkpoints/RoboTwin-Stage2/90000`，模型结构、历史帧数和预测长度来自该目录的 `config.json`。
当前 policy 使用最后观测帧全局 patch 前缀和三类 token embedding。旧的仅含 TCP 历史条件的 Stage2 checkpoint 与此结构不兼容；请通过 `--checkpoint` 指定新结构训练得到的 checkpoint。

## 运行

```bash
conda activate 4rc

# 打开 Gradio：依次点击左、右 TCP，也可点击“使用首帧真值投影”。
# 确认任务文字后运行；结果通过 Viser 展示。
python stage2_sliding_window_inference.py \
  --input datasets/eval_sets/place_dual_shoes/episode_0000092 \
  --interactive --fps 5 --point-size 0.0016

# 非交互：首帧用真值投影初始化，推理结束打开 Viser。
python stage2_sliding_window_inference.py \
  --input datasets/eval_sets/place_dual_shoes/episode_0000092

# 无界面保存；改为每帧移动窗口，覆盖自定义指令。
python stage2_sliding_window_inference.py \
  --input datasets/eval_sets/place_dual_shoes/episode_0000092 \
  --window-stride 1 --instruction "Place both shoes into the box." \
  --headless --output-dir outputs/stage2_shoes
```

输入 episode 需要 `images/<view>/`、`intrinsics/<view>.npy`、`extrinsics/<view>.npy`、`metadata.json`。
RGB 必须为 320×240，文件名以连续帧号结尾。元数据提供正数 `frequency_hz`；任务文字默认取
`instructions` 第一条非空字符串，也可用 `--instruction` 覆盖。脚本不读取真实 depth。

`TCP_third/{left,right}_state.npy`（head_view 对应 `TCP_head`）用于首帧真值选点和未来真值对比。
没有 TCP 标注时，使用 `--interactive` 手动选点，仍可预测。真值按钮不可用时，可以检查首帧 TCP 是否出画。

常用选项：

| 参数 | 默认值 / 含义 |
| --- | --- |
| `--checkpoint` | `checkpoints/RoboTwin-Stage2/90000`，包含完整权重与配置 |
| `--window-stride` | 7；范围 1～历史长度减一 |
| `--sampling-steps` | checkpoint 配置，当前为 8 |
| `--seed` | 42；第 i 个窗口使用 seed+i |
| `--device` / `--dtype` | auto；CUDA 自动选择 BF16/FP16，CPU 使用 FP32 |
| `--t5-model` | 可指定本地 T5-base 目录；默认使用配置中的模型及本地缓存 |
| `--max-windows` | 调试时限制窗口数，未跑完整段时 JSON 标记 incomplete |
| `--headless` | 仅保存，不启动 Viser；可与交互选点组合 |
| `--host` / `--port` | Viser：127.0.0.1 / 8020 |
| `--ui-host` / `--ui-port` | Gradio：127.0.0.1 / 7860 |
| `--fps` | 初始播放 FPS，默认 5；页面滑块范围 0.25～30 |
| `--point-size` | 初始点大小 0.0016；页面 Point size 滑块可实时调整 |
| `--max-depth` | 点云显示深度上限 3 米；0 关闭限制 |
| `--confidence-percentile` | 过滤置信度最低的 2.5% 像素 |
| `--max-points` | 每个显示帧最多 100000 点；0 不限 |

交互页面提示 `Cannot find empty port in range: 7860-7860` 时，添加 `--ui-port 8097`（或其他空闲端口）。`--server-port` 和 `--server_port` 是 `--ui-port` 的别名。`--port` 只设置 Viser 端口，不会更改 Gradio 端口；两个服务应使用不同端口。

通过远程机器浏览时，可转发 7860、8020 端口。页面中包含 Viser 的直接链接。

## 时间、坐标和真值

窗口和预测长度来自 checkpoint 配置，新训练默认输入 8 帧、预测后续 16 条动作记录。
默认窗口如 `[0,8)`、`[7,15)`；最后一个窗口向前对齐到末帧，保持完整 8 帧，
不填充历史。少于 8 帧或历史断帧会报错；这是整段推理的窗口规则。

Stage2 统一只编码历史和未来的序号，不规定动作执行频率。返回
`future_step_indices`（1～16）及用于真值对齐的 `future_frame_indices`，不返回
预测目标秒数 `future_frame_times`。JSON `format_version=2`，采集频率记录为
`source_frequency_hz`，不表示动作执行频率。末尾仍预测完整长度，缺失真值使用有效掩码。

加载 checkpoint 时始终使用序号条件，不再切回旧物理时间模式。已有权重可通过
训练配置的 `stage2_checkpoint` 初始化后继续训练，再使用新 checkpoint 推理。

只有第一个窗口由交互或真值选点初始化。后续窗口使用上一窗口对下一首帧恢复出的 TCP，
按相机内参投影传递；不会用预测的未来动作替代恢复结果，也不会每窗重新读取真值作为条件。
传递位置出画、非有限或在相机后方时停止，并保存已完成的窗口和错误原因。

**每个窗口的全部可视化及保存的 TCP 坐标都位于该窗口最后一帧的 OpenCV 相机坐标系**：
x 向右、y 向下、z 向前，位置单位米，旋转为 3×3 矩阵。历史点云由预测深度和标定内参反投影，
通过 `T_anchor @ inverse(T_frame)` 对齐。未来动作本身已在 anchor 相机坐标系。
不同窗口独立保存，不能直接连接不同 anchor 相机坐标下的轨迹。

未来 `action_gripper_score >= 0` 表示打开，`action_gripper` 为 1/0，失败时为 -1。
该 score **不是概率**；历史 `history_gripper_probability` 则是 sigmoid 后的概率。
预测失败由 `success=false` 表示，JSON 中无效浮点数写为 `null`。

## 可视化和输出

Viser 使用与 Stage1 参考脚本相同的逐帧播放方式：

- **Frame / Previous / Next**：按 episode 原始顺序选帧；重叠帧仅播放一次，循环回到首帧。
- **Play / FPS**：播放、暂停及实时调速，范围 0.25～30 FPS，默认 5；播放期间禁用手动跳帧。
- **Video export → Export episode MP4 (current view)**：固定点击时的相机视角，以当前 FPS 和显示设置导出完整 episode 的三维场景（不含 GUI/RGB 面板）。保持视口比例，最高 1920×1080；完成后点击浏览器下载通知。导出期间保持页面打开且可见，结束后恢复原帧和播放状态。TCP 滑窗脚本也支持此按钮。
- **Point size**：实时调整点云中每个点的显示大小，无需重新推理。
- **Future steps**：控制显示的未来步数；另有历史/预测/真值显隐、TCP 姿态轴和置信度过滤。

默认从第一帧开始播放。点云和当前 TCP 跟随 Frame 更新；未来轨迹来自覆盖该帧的第一个推理窗口，
预测起点仍是该窗口最后一帧，页面标明窗口历史范围与当前帧。未来只画 TCP 轨迹，不生成未来场景点云。
左臂橙色、右臂蓝色，真实未来轨迹为黄色/绿色虚线。ADE/FDE 只统计存在真实标签的未来步。

默认保存到 `outputs/stage2_inference/<task>/<episode>/`；重复运行相同输出目录会覆盖同名结果。

- `predictions.json`：指令、采样配置、坐标约定、窗口帧号、选点来源、历史恢复、未来预测、真值有效掩码及位置误差。
不再保存 `.npy` / `.npz` 文件。每窗深度和置信度保存在内存中供播放，结束程序后释放。

轨迹 JSON 逐窗写入；RGB 从输入 episode 按需读取。`--headless` 会直接丢弃几何数组以节省内存。
仅凭保存的 JSON 无法恢复点云播放，需要重新运行推理。
`complete` 表示是否遍历完全部窗口，不代表所有窗口的 `success` 均为真。

## Python 接口和测试

`load_stage2_policy` 加载策略；`load_episode` 读取 episode；`infer_episode_sliding_windows` 处理整段。
`infer_stage2_window` 接受 padded RGB `[1,8,3,252,322]`、padded 内参 `[1,8,3,3]`、
padded 首帧选点 `[1,2,2]` 和任务文字，不接收历史真值或未来图像。
接口不接收 `frame_times` 和 `frequency_hz`，历史和未来均按序号编码。
返回深度、历史 TCP、未来位置 `[16,2,3]`、旋转 `[16,2,3,3]`、夹爪、动作序号及成功状态。

```python
prediction = infer_stage2_window(policy, images, intrinsics, query_points, instruction="Lift the cup.")
```
单窗口接口的历史 TCP 仍在各帧自身相机坐标；整段接口负责统一到 anchor 相机。
整段返回值的 `_geometry` 保存内存中的深度与置信度，不进入 JSON；设置 `keep_geometry=False` 可跳过缓存。

```bash
python -m pytest tests/test_stage2_inference.py tests/test_action_policy.py tests/test_action_dataset.py -q
```
