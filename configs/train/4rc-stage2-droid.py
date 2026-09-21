"""DROID stage two: eight monocular observations and sixteen single-arm future actions."""

# ======================================================
# 4RC Geometry + TCP Tracking + Action Configuration
# ======================================================

# == Common Configuration ==
output_dir = "droid_script/outputs/stage2"
wandb_run_name = "4rc-droid-stage2"
logging_dir = "logs"
stage1_checkpoint = "droid_script/outputs/stage1/final_checkpoint"  # Native single-arm DROID checkpoint.
resume = None  # Stage-two checkpoint directory when resuming.

# == Dataset Configuration ==
# Both stages use the same episode TXT split. The two cameras are sampled separately.
data_sources = (
    {
        "name": "droid",
        "type": "droid",
        "weight": 1.0,
        "options": {"root": "datasets/droid_episodes"},
    },
)
index_path = "droid_script/cache/droid.sqlite"
train_set = "droid_script/splits/train_set.txt"
val_set = "droid_script/splits/val_set.txt"
history_frames = 8
prediction_horizon = 16  # Short futures repeat the last valid action; padding is masked.
batch_size = 1  # Per GPU; each sample contains history_frames input images.
# The action adapter requires forward, contiguous windows. The runner derives
# these image budgets and intervals again from batch_size / history_frames.
train_batch_images = batch_size * history_frames
scene_counts = (batch_size,)
min_views = history_frames
max_views = history_frames
min_interval = 1
max_interval = 1
reverse_probability = 0.0
# This export has fixed 15 fps; timestamps are not read.
frame_rate = 15
max_depth = 3.0  # Metric z-depth cap; None disables the cap.
max_tcp_linear_speed = 3.0  # Metres per second; None disables the check.
max_tcp_angular_speed = 4.0 * 3.141592653589793  # Radians per second.
max_episodes = None
augment = True
normalize_geometry = False  # Action targets require metric geometry.
geometry_frame = "robot_base"
tcp_frame = "camera"
gripper_encoding = "continuous"
padding = (1, 1, 1, 1)  # Native 320x180 -> 322x182; no resizing.
cache_size = 8  # Maximum mmap trajectories per worker.
prefetch_factor = 2
num_workers = 8
batches_per_epoch = 2000  # Sampling epoch length, independent of total frame count.
recent_buffer_size = 10_000

# == Validation Configuration ==
# Split membership comes only from train_set/val_set TXT; preparation defaults to 90/10.
validation_batches = 16
validation_batch_size = 1
validate_every_steps = 1000
# Both stages enforce identical split file hashes.
stage1_validation_overlap = "none: shared DROID episode TXT splits"

# == Model Configuration ==
num_arms = 1
train_backbone = True
train_geometry_head = True
train_motion_decoder = True  # Shared sparse TCP decoder; dense tracking is not run.
train_query_encoder = True
train_tcp_head = True
train_history_pool = True  # Global visual encoder, TCP pooling, and time/type embeddings.
train_action_head = True
train_language_projection = True
train_camera_decoder = True  # Absolute robot-base camera supervision.
# Dense track head remains frozen.
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
lr_backbone = 1e-6
lr_head = 2e-5
lr_camera = 2e-5
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
camera_loss_weight = 1.0
camera_translation_weight = 1.0
camera_rotation_weight = 1.0
camera_fov_weight = 0.1
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
visualize_every_steps = 1000
checkpointing_steps = 1000
save_each_epoch = False
report_to = ["tensorboard"]  # Add "wandb" after login or configuring offline mode.
