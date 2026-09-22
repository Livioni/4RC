"""DROID stage one: monocular metric geometry, absolute cameras and single-arm TCP tracking."""

# ======================================================
# 4RC Geometry + TCP Tracking Configuration
# ======================================================

# == Common Configuration ==
output_dir = "outputs/droid/stage1"
logging_dir = "logs"
pretrained_model = "Luo-Yihang/4RC"
resume = None

# == Dataset Configuration ==
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
min_views = 2
max_views = 18
min_interval = 1
max_interval = 5
# Each clip's first sampled frame is its query. Random starts and reversal make
# both forward and reverse queries arbitrary episode frames.
reverse_probability = 0.5
# This export has fixed 15 fps; timestamps are not read.
frame_rate = 15
# Ignore dense depth targets beyond this metric z-depth. None disables the cap.
max_depth = 3.0
# Split trajectories at physically implausible source discontinuities. Set
# either threshold to None to disable that check.
max_tcp_linear_speed = 3.0  # metres per second
max_tcp_angular_speed = 4.0 * 3.141592653589793  # radians per second
max_episodes = None
augment = True
# Rays/cameras use the robot base; TCP labels remain in the selected camera.
geometry_frame = "robot_base"
tcp_frame = "camera"
gripper_encoding = "continuous"
padding = (1, 1, 1, 1)  # Native 320x180 -> 322x182; no resizing.
normalize_geometry = False
cache_size = 8  # Maximum mmap trajectories per worker.
prefetch_factor = 2
num_workers = 8
train_batch_images = 18
scene_counts = (1, 2, 3, 6, 9)
batches_per_epoch = 2000  # Sampling epoch length, independent of total frame count.
recent_buffer_size = 10_000

# == Model Configuration ==
num_arms = 1
train_backbone = True
train_geometry_head = True
train_camera_decoder = True
train_motion_decoder = False
train_tcp_tracker = True
# Sample a 3x3 local patch neighborhood around each projected TCP point.
tcp_query_window_size = 3

# == Training Configuration ==
seed = 42
num_train_epochs = None  # Stop at cumulative max_train_steps.
max_train_steps = 500_000
gradient_accumulation_steps = 2
mixed_precision = "bf16"
max_grad_norm = 1.0
# DualDPT retains intermediate ray-pyramid prediction layers, but this
# geometry-only objective supervises only the final ray level.
find_unused_parameters = True

# == Optimizer Configuration ==
lr_backbone = 1e-5
lr_head = 2e-5
lr_camera = 2e-5
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
camera_loss_weight = 1.0
camera_translation_weight = 1.0
camera_rotation_weight = 1.0
camera_fov_weight = 0.1
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

# == TCP visual-query curriculum ==
tcp_query_initial_exact_ratio = 0.80
tcp_query_exact_ratio = 0.25
tcp_query_max_jitter_patches = 1.0
tcp_query_curriculum_warmup_ratio = 0.10
tcp_query_curriculum_transition_ratio = 0.20

# == Logging and Checkpoint Configuration ==
log_every_steps = 10
visualize_every_steps = 1000
checkpointing_steps = 10_000
save_each_epoch = False
report_to = ["tensorboard"]  # Add "wandb" after login or configuring offline mode.
