"""Single-stage geometry and sparse TCP tracking configuration for RoboTwin."""

# ======================================================
# 4RC Geometry + TCP Tracking Configuration
# ======================================================

# == Common Configuration ==
output_dir = "outputs/4rc-robotwin-tcp-debug"
logging_dir = "logs"
pretrained_model = "Luo-Yihang/4RC"
resume = None

# == Dataset Configuration ==
data_root = "datasets/RoboTwin"
view = "head_view"
min_views = 2
max_views = 18
min_interval = 1
max_interval = 5
# Each clip's first sampled frame is its query. Random starts and reversal make
# both forward and reverse queries arbitrary episode frames.
reverse_probability = 0.5
# Read the saved frequency from each episode's metadata.json (15 Hz in the
# current RoboTwin export). Set a positive number only to override metadata.
frame_rate = None
# Split trajectories at physically implausible source discontinuities. Set
# either threshold to None to disable that check.
max_tcp_linear_speed = 3.0  # metres per second
max_tcp_angular_speed = 4.0 * 3.141592653589793  # radians per second
max_episodes = None
augment = True
# Keep RoboTwin depth and camera translations in meters. Set to True to use
# the legacy per-scene unit-mean-distance normalization.
normalize_geometry = False
num_workers = 8
train_batch_images = 18
scene_counts = (1, 2, 3, 6, 9)
batches_per_epoch = None
recent_buffer_size = 10_000

# == Model Configuration ==
train_backbone = True
train_geometry_head = True
train_camera_decoder = False
train_motion_decoder = False
train_tcp_tracker = True

# == Training Configuration ==
seed = 42
num_train_epochs = 50
max_train_steps = 50_000
gradient_accumulation_steps = 2
mixed_precision = "bf16"
max_grad_norm = 1.0
# DualDPT retains intermediate ray-pyramid prediction layers, but this
# geometry-only objective supervises only the final ray level.
find_unused_parameters = True

# == Optimizer Configuration ==
lr_backbone = 1e-5
lr_head = 2e-5
lr_motion_decoder = 1e-5
lr_tcp = 1e-4
adam_beta1 = 0.9
adam_beta2 = 0.95
adam_epsilon = 1e-8
weight_decay = 0.01

# == Learning Rate Scheduler Configuration ==
warmup_steps = 1000
eta_min_factor = 0.1

# == Loss Configuration ==
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

# == Logging and Checkpoint Configuration ==
log_every_steps = 10
visualize_every_steps = 1000
checkpointing_steps = 5000
save_each_epoch = False
report_to = "tensorboard"
