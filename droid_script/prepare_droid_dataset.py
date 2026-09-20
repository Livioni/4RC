"""Prepare the DROID index and portable episode split files."""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
import sqlite3
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arc.datasets.droid_index import DEFAULT_INDEX, DEFAULT_TRAIN, DEFAULT_VAL, ensure_index, write_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="datasets/droid_episodes")
    parser.add_argument("--index-path", default=DEFAULT_INDEX)
    parser.add_argument("--train-set", default=DEFAULT_TRAIN)
    parser.add_argument("--val-set", default=DEFAULT_VAL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--max-tcp-linear-speed", type=lambda v: None if v.lower() == "none" else float(v), default=3.0)
    parser.add_argument("--max-tcp-angular-speed", type=lambda v: None if v.lower() == "none" else float(v), default=4 * math.pi)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--overwrite-splits", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    reused_splits = Path(args.train_set).is_file() and Path(args.val_set).is_file() and not args.overwrite_splits
    path = ensure_index(args.data_root, args.index_path, rebuild=args.rebuild_index, max_episodes=args.max_episodes,
                        max_tcp_linear_speed=args.max_tcp_linear_speed, max_tcp_angular_speed=args.max_tcp_angular_speed)
    splits = write_splits(args.data_root, path, args.train_set, args.val_set, seed=args.seed,
                          validation_fraction=args.validation_fraction, overwrite=args.overwrite_splits)
    with sqlite3.connect(path) as connection:
        excluded = list(connection.execute("SELECT episode,reason FROM excluded ORDER BY episode"))
    report = {"fps": 15, "seed": None if reused_splits else args.seed,
              "validation_fraction": None if reused_splits else args.validation_fraction,
              "split_mode": "existing_txt" if reused_splits else "generated_hash",
              "train_episodes": len(splits["train"]), "val_episodes": len(splits["validation"]),
              "excluded": [{"episode": name, "reason": reason} for name, reason in excluded]}
    report_path = Path(args.train_set).with_name("split_report.json")
    if reused_splits and report_path.is_file():
        previous = json.loads(report_path.read_text())
        for key in ("seed", "validation_fraction"):
            report[key] = previous.get(key)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "excluded"}, ensure_ascii=False))
    print(f"Excluded episodes: {len(excluded)}; index: {path}")


if __name__ == "__main__":
    main()
