"""Check real DROID samples without loading model weights or requiring a GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from arc.datasets.droid import DroidDataset
from arc.datasets.mixture import validate_training_sample
from arc.datasets.droid_index import DEFAULT_INDEX, DEFAULT_TRAIN, DEFAULT_VAL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="datasets/droid_episodes")
    parser.add_argument("--index-path", default=DEFAULT_INDEX)
    parser.add_argument("--train-set", default=DEFAULT_TRAIN)
    parser.add_argument("--val-set", default=DEFAULT_VAL)
    parser.add_argument("--max-episodes", type=int, default=8)
    parser.add_argument("--samples", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(2)
    summary = []
    for stage in (1, 2):
        for split in ("train", "validation"):
            started = time.monotonic()
            dataset = DroidDataset(args.data_root, index_path=args.index_path, train_set=args.train_set,
                                    val_set=args.val_set, max_episodes=args.max_episodes,
                                    stage=stage, split=split, augment=False)
            for i in range(min(args.samples, len(dataset))):
                sample = dataset.get_sample(i, 3 if stage == 1 else 8, 42 + i)
                validate_training_sample(sample, "droid")
                if not torch.isfinite(sample["tcp_state"]).all():
                    raise ValueError("Non-finite TCP labels")
                summary.append({"stage": stage, "split": split, "episode": sample["episode"],
                    "camera": sample["camera_id"], "images": list(sample["images"].shape),
                    "tcp": list(sample["tcp_state"].shape), "fps": sample["frame_rate"],
                    "valid_depth_pixels": int(sample["valid_mask"].sum()),
                    "query_valid": sample["tcp_query_valid"].tolist(),
                    "future_frames": int(sample["future_action_valid"].sum()) if stage == 2 else None,
                    "elapsed_seconds": round(time.monotonic() - started, 3)})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
