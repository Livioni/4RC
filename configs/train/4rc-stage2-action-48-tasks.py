# ======================================================
# 4RC TCP Tracking + Action Configuration (no geometry decoding)
# ======================================================

# == Common Configuration ==
output_dir = "outputs/4rc-stage2-gr00t-48-tasks"
wandb_run_name = "4rc-stage2-gr00t-48-tasks"
logging_dir = "logs"
stage1_checkpoint = "outputs/4rc-robotwin-stage1/final_checkpoint/model.safetensors"  # Shared visual/TCP initialization for every new run.
resume = None  # Full-state resume of a V2 run only.

# == Dataset Configuration ==
# Applies to every enabled source, for training and validation.
tasks = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]
# Relative sampling weights are normalized over enabled sources. A zero weight
# skips the source entirely, including task-directory checks.
# To mix standard/randomized RoboTwin at 1:2, set their weights to 1.0 / 2.0.
# Stage two currently supports only sources with RoboTwin action labels.
data_sources = (
    {
        "name": "robotwin",
        "type": "robotwin",
        "weight": 1.0,
        "options": {"root": "datasets/RoboTwin"},
    },
    {
        "name": "robotwin_random",
        "type": "robotwin",
        "weight": 0.0,
        "options": {"root": "datasets/RoboTwin_random"},
    },
)
view = "third_views"
history_frames = 4  # Repeat the oldest observed frame if history is short.
prediction_horizon = 32  # Future padding is masked in attention and loss.
batch_size = 8  # Per GPU; each sample contains history_frames input images.
# Step indices encode observation order; repeated frames keep their source timestamps.
# The action adapter requires forward windows within one valid segment. The runner derives
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
color_augment = False  # Keep blur; disable jitter and grayscale.
normalize_geometry = False  # Action targets require metric geometry.
num_workers = 8
batches_per_epoch = None
recent_buffer_size = 10_000

# == Validation Configuration ==
enable_validation = False
validation_fraction = 0
validation_batches = 16
validation_batch_size = 1
validate_every_steps = 0
# Stage-one overlap must be audited before claiming unseen-episode generalization.
stage1_validation_overlap = "unknown"

# == Model Configuration ==
action_head_type = "groot_tcp_v1"
state_normalization_version = 1
global_token_grid = (14, 19)
condition_dim = 2048
action_output_dim = 1024
action_dropout = 0.2
action_gradient_checkpointing = True
wrist_model = "facebook/dinov2-large"
wrist_cache_dir = ".cache/huggingface"
action_normalization = "componentwise"
action_normalization_version = 2
train_backbone = True
train_geometry_head = False
decode_geometry = False  # Skip the geometry head in training and validation.
train_motion_decoder = True  # Shared sparse TCP decoder; dense tracking is not run.
train_query_encoder = True
train_tcp_head = True
train_condition_encoders = True  # State, unpooled TCP, image and type/position encoders.
train_action_head = True
train_language_projection = True
# Camera decoder and dense track head are frozen by the stage-two optimizer.
tcp_query_window_size = 3

t5_model = "google-t5/t5-base"  # Frozen encoder; a local pretrained directory also works.
text_max_length = 128
# About 294M parameters in DiT; encoders and action decoder are counted separately.
action_dim = 1280
action_depth = 12
action_heads = 32
# Observation indices [-3..0], future positions [1..32]; flow time is a separate condition.
sampling_steps = 4

# == History TCP Condition Curriculum ==
# Choose GT/recovered history per clip. Linearly interpolate over optimizer
# updates from 100% GT to 50% GT / 50% recovered; resume uses cumulative steps.
history_tcp_gt_initial_ratio = 0.5
history_tcp_gt_final_ratio = 0.5

# == Training Configuration ==
seed = 42
num_train_epochs = None  # No epoch limit; stop at the cumulative max_train_steps.
max_train_steps = 500_000
gradient_accumulation_steps = 2
mixed_precision = "bf16"
max_grad_norm = 1.0
find_unused_parameters = False

# == Optimizer Configuration ==
# Each module has an independent rate. A False train flag or zero LR freezes it.
lr_backbone = 1e-4
lr_head = 2e-5
lr_motion_decoder = 1e-5
lr_query_encoder = 1e-4
lr_tcp_head = 1e-4
lr_condition_encoders = 1e-4
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
geometry_loss_weight = 0.0
tcp_loss_weight = 0.2
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
tcp_temporal_weight = 0.0  # Skip physical TCP velocity loss; flow-matching supervision remains enabled.
tcp_gripper_weight = 0.2

action_position_weight = 1.0
action_rotation_weight = 1.0
action_gripper_weight = 1.0

# == Logging and Checkpoint Configuration ==
log_every_steps = 10
visualize_every_steps = 0  # No decoded depth to visualize.
checkpointing_steps = 10_000
save_each_epoch = False
report_to = ["tensorboard"]
