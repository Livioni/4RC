"""Stage two: recover eight observations and generate sixteen future actions."""

# ======================================================
# 4RC Geometry + TCP Tracking + Action Configuration
# ======================================================

# == Common Configuration ==
output_dir = "outputs/4rc-stage2-action-global-bs4x3"
wandb_run_name = "4rc-stage2-action-global-bs4x3"
logging_dir = "logs"
stage1_checkpoint = "outputs/4rc-robotwin-mixed-tcp-point-query/final_checkpoint/model.safetensors"  # Required for a new run; file or stage-one checkpoint directory.
resume = None  # Stage-two checkpoint directory when resuming.

# == Dataset Configuration ==
# Relative sampling weights are normalized over enabled sources. A zero weight
# skips the source entirely, so the default only needs datasets/RoboTwin.
# To mix standard/randomized RoboTwin at 1:2, set their weights to 1.0 / 2.0.
# Stage two currently supports only sources with RoboTwin action labels.
data_sources = (
    {
        "name": "robotwin",
        "type": "robotwin",
        "weight": 0.33,
        "options": {"root": "datasets/RoboTwin"},
    },
    {
        "name": "robotwin_random",
        "type": "robotwin",
        "weight": 0.66,
        "options": {"root": "datasets/RoboTwin_random"},
    },
)
view = "third_views"
history_frames = 8
prediction_horizon = 16  # Short futures repeat the last valid action; padding is masked.
batch_size = 3  # Per GPU; each sample contains history_frames input images.
# The action adapter requires forward, contiguous windows. The runner derives
# these image budgets and intervals again from batch_size / history_frames.
train_batch_images = batch_size * history_frames
scene_counts = (batch_size,)
min_views = history_frames
max_views = history_frames
min_interval = 1
max_interval = 1
reverse_probability = 0.0
# Read the saved frequency from each episode's metadata.json.
frame_rate = None
max_depth = 3.0  # Metric z-depth cap; None disables the cap.
max_tcp_linear_speed = 3.0  # Metres per second; None disables the check.
max_tcp_angular_speed = 4.0 * 3.141592653589793  # Radians per second.
max_episodes = None
augment = True
normalize_geometry = False  # Action targets require metric geometry.
num_workers = 8
batches_per_epoch = None
recent_buffer_size = 10_000

# == Validation Configuration ==
validation_fraction = 0.1
validation_batches = 16
validation_batch_size = 1
validate_every_steps = 1000
# Stage-one overlap must be audited before claiming unseen-episode generalization.
stage1_validation_overlap = "unknown"

# == Model Configuration ==
train_backbone = True
train_geometry_head = True
train_motion_decoder = True  # Shared sparse TCP decoder; dense tracking is not run.
train_query_encoder = True
train_tcp_head = True
train_history_pool = True  # Global visual encoder, TCP pooling, and time/type embeddings.
train_action_head = True
train_language_projection = True
# Camera decoder and dense track head are frozen by the stage-two optimizer.
tcp_query_window_size = 3

t5_model = "google-t5/t5-base"  # Frozen encoder; a local pretrained directory also works.
text_max_length = 128
# About 297M parameters in the Action DiT (excluding conditioning modules).
action_dim = 768
action_depth = 20
action_heads = 12
time_unit_seconds = 1.0 / 15
sampling_steps = 8

# == History TCP Condition Curriculum ==
# Choose GT/recovered history per clip. Linearly interpolate over optimizer
# updates from 100% GT to 50% GT / 50% recovered; resume uses cumulative steps.
history_tcp_gt_initial_ratio = 1.0
history_tcp_gt_final_ratio = 0.5

# == Training Configuration ==
seed = 42
num_train_epochs = None  # No epoch limit; stop at the cumulative max_train_steps.
max_train_steps = 500_000
gradient_accumulation_steps = 2
mixed_precision = "bf16"
max_grad_norm = 1.0
find_unused_parameters = True

# == Optimizer Configuration ==
# Each module has an independent rate. A False train flag or zero LR freezes it.
lr_backbone = 1e-5
lr_head = 2e-5
lr_motion_decoder = 1e-5
lr_query_encoder = 1e-4
lr_tcp_head = 1e-4
lr_history_pool = 1e-4
lr_action_head = 1e-4
lr_language_projection = 1e-4
adam_beta1 = 0.9
adam_beta2 = 0.95
adam_epsilon = 1e-8
weight_decay = 0.01

# == Learning Rate Scheduler Configuration ==
warmup_steps = 1000
eta_min_factor = 0.1

# == Loss Configuration ==
geometry_loss_weight = 1.0
tcp_loss_weight = 1.0
action_loss_weight = 1.0

depth_loss_weight = 1.0
ray_loss_weight = 1.0
loss_gamma = 1.0
loss_alpha = 0.2
depth_valid_range = 0.98
gradient_scales = 4

tcp_point_scale = 0.1
tcp_virtual_point_radius = 0.03
tcp_rotation_weight = 0.5
tcp_temporal_weight = 0.2
tcp_gripper_weight = 0.2
tcp_velocity_scale = 1.0

action_position_weight = 1.0
action_rotation_weight = 1.0
action_gripper_weight = 1.0

# == Logging and Checkpoint Configuration ==
log_every_steps = 10
visualize_every_steps = 5_000
checkpointing_steps = 10_000
save_each_epoch = False
report_to = ["tensorboard", "wandb"]
