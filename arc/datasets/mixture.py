"""Extensible dataset adapters and weighted homogeneous-batch mixing."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, default_collate

from .robotwin import RoboTwin4RC


REQUIRED_TRAINING_SAMPLE_KEYS = frozenset(
    {
        "images",
        "depth",
        "valid_mask",
        "original_mask",
        "intrinsics",
        "extrinsics",
        "frame_times",
        "tcp_state",
        "tcp_query_points",
        "tcp_query_valid",
        "source_size",
        "padding",
        "task",
        "episode",
    }
)


@runtime_checkable
class SequenceDatasetAdapter(Protocol):
    """Contract required by the generic 4RC training data pipeline."""

    min_views: int
    max_views: int
    tcp_position_mean: torch.Tensor
    tcp_position_std: torch.Tensor

    def __len__(self) -> int: ...

    def set_epoch(self, epoch: int) -> None: ...

    def eligible_indices(self, num_views: int) -> np.ndarray: ...

    def get_sample(
        self, index: int, num_views: int, sample_seed: int
    ) -> dict[str, Any]: ...


DatasetFactory = Callable[
    [Mapping[str, Any], Mapping[str, Any]], SequenceDatasetAdapter
]
_DATASET_ADAPTER_REGISTRY: dict[str, DatasetFactory] = {}


def register_dataset_adapter(
    type_name: str,
    factory: DatasetFactory,
    *,
    replace: bool = False,
) -> None:
    """Register a config ``type`` with a dataset-adapter factory."""
    normalized = str(type_name).strip()
    if not normalized:
        raise ValueError("Dataset adapter type cannot be empty")
    if normalized in _DATASET_ADAPTER_REGISTRY and not replace:
        raise ValueError(f"Dataset adapter type {normalized!r} is already registered")
    if not callable(factory):
        raise TypeError("Dataset adapter factory must be callable")
    _DATASET_ADAPTER_REGISTRY[normalized] = factory


@dataclass(frozen=True, slots=True)
class DatasetSource:
    """One named dataset adapter and its requested sampling weight."""

    name: str
    type_name: str
    dataset: SequenceDatasetAdapter
    weight: float


@dataclass(frozen=True, slots=True)
class MixtureSampleRequest:
    """Index routed by the batch sampler to one source-local sample."""

    source_index: int
    sample_index: int
    num_views: int
    sample_seed: int


def _validate_adapter(source: DatasetSource) -> None:
    dataset = source.dataset
    missing = [
        attribute
        for attribute in (
            "min_views",
            "max_views",
            "tcp_position_mean",
            "tcp_position_std",
            "set_epoch",
            "eligible_indices",
            "get_sample",
        )
        if not hasattr(dataset, attribute)
    ]
    if missing:
        raise TypeError(
            f"Dataset source {source.name!r} does not implement the adapter "
            f"contract; missing: {', '.join(missing)}"
        )
    if len(dataset) < 1:
        raise ValueError(f"Dataset source {source.name!r} has no valid samples")
    if dataset.min_views < 2 or dataset.max_views < dataset.min_views:
        raise ValueError(
            f"Dataset source {source.name!r} has invalid view range "
            f"[{dataset.min_views}, {dataset.max_views}]"
        )
    for label, value in (
        ("tcp_position_mean", dataset.tcp_position_mean),
        ("tcp_position_std", dataset.tcp_position_std),
    ):
        if not isinstance(value, torch.Tensor) or value.shape != (2, 3):
            raise ValueError(
                f"Dataset source {source.name!r} {label} must have shape [2,3]"
            )
        if not torch.isfinite(value).all():
            raise ValueError(
                f"Dataset source {source.name!r} {label} must be finite"
            )
    if torch.any(dataset.tcp_position_std <= 0):
        raise ValueError(
            f"Dataset source {source.name!r} tcp_position_std must be positive"
        )


def validate_training_sample(sample: Mapping[str, Any], source_name: str) -> None:
    """Validate the complete geometry + TCP sample contract."""
    missing = sorted(REQUIRED_TRAINING_SAMPLE_KEYS.difference(sample))
    if missing:
        raise ValueError(
            f"Dataset source {source_name!r} returned an incomplete sample; "
            f"missing: {', '.join(missing)}"
        )

    tensors = {
        key: sample[key]
        for key in REQUIRED_TRAINING_SAMPLE_KEYS
        if key not in {"task", "episode"}
    }
    non_tensors = sorted(key for key, value in tensors.items() if not torch.is_tensor(value))
    if non_tensors:
        raise TypeError(
            f"Dataset source {source_name!r} returned non-tensor fields: "
            f"{', '.join(non_tensors)}"
        )

    images = sample["images"]
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError(
            f"Dataset source {source_name!r} images must have shape [S,3,H,W]"
        )
    sequence, _, height, width = images.shape
    if height % 14 or width % 14:
        raise ValueError(
            f"Dataset source {source_name!r} image size {height}x{width} is not "
            "divisible by patch size 14"
        )
    expected_shapes = {
        "depth": (sequence, height, width),
        "valid_mask": (sequence, height, width),
        "original_mask": (sequence, height, width),
        "intrinsics": (sequence, 3, 3),
        "frame_times": (sequence,),
        "tcp_state": (sequence, 2, 7),
        "tcp_query_points": (2, 2),
        "tcp_query_valid": (2,),
        "source_size": (2,),
        "padding": (4,),
    }
    for key, expected in expected_shapes.items():
        if tuple(sample[key].shape) != expected:
            raise ValueError(
                f"Dataset source {source_name!r} {key} must have shape "
                f"{list(expected)}, got {list(sample[key].shape)}"
            )
    extrinsics_shape = tuple(sample["extrinsics"].shape)
    if extrinsics_shape not in {(sequence, 3, 4), (sequence, 4, 4)}:
        raise ValueError(
            f"Dataset source {source_name!r} extrinsics must have shape "
            f"[S,3,4] or [S,4,4], got {list(extrinsics_shape)}"
        )


class WeightedDatasetMixture(Dataset[dict[str, Any]]):
    """Route requests to named adapters and expose weighted TCP statistics."""

    def __init__(self, sources: Sequence[DatasetSource]) -> None:
        if not sources:
            raise ValueError("At least one dataset source is required")
        self.sources = tuple(sources)
        names = [source.name for source in self.sources]
        if any(not name.strip() for name in names):
            raise ValueError("Dataset source names cannot be empty")
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"Duplicate dataset source names: {', '.join(duplicates)}"
            )

        weights = np.asarray([source.weight for source in self.sources], dtype=np.float64)
        if not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("Dataset source weights must be finite and non-negative")
        if not np.any(weights > 0):
            raise ValueError("At least one dataset source weight must be positive")
        self.weights = weights / weights.sum()

        for source in self.sources:
            _validate_adapter(source)

        positive_sources = np.flatnonzero(self.weights > 0)
        if len(positive_sources) == 1:
            source_index = int(positive_sources[0])
            self.tcp_position_mean = (
                self.sources[source_index].dataset.tcp_position_mean.clone()
            )
            self.tcp_position_std = (
                self.sources[source_index].dataset.tcp_position_std.clone()
            )
        else:
            means = torch.stack(
                [source.dataset.tcp_position_mean.double() for source in self.sources]
            )
            variances = torch.stack(
                [source.dataset.tcp_position_std.double().square() for source in self.sources]
            )
            tensor_weights = torch.from_numpy(self.weights).to(means)[:, None, None]
            mixed_mean = (tensor_weights * means).sum(dim=0)
            mixed_second_moment = (
                tensor_weights * (variances + means.square())
            ).sum(dim=0)
            mixed_variance = (
                mixed_second_moment - mixed_mean.square()
            ).clamp_min(1e-6)
            self.tcp_position_mean = mixed_mean.float()
            self.tcp_position_std = mixed_variance.sqrt().float()
        self.invalid_transition_count = sum(
            int(getattr(source.dataset, "invalid_transition_count", 0))
            for source in self.sources
        )
        self.segmented_episode_count = sum(
            int(getattr(source.dataset, "segmented_episode_count", 0))
            for source in self.sources
        )

    def __len__(self) -> int:
        return sum(len(source.dataset) for source in self.sources)

    @property
    def source_names(self) -> tuple[str, ...]:
        return tuple(source.name for source in self.sources)

    @property
    def source_summaries(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "name": source.name,
                "type": source.type_name,
                "episodes": len(source.dataset),
                "weight": float(weight),
            }
            for source, weight in zip(self.sources, self.weights)
        )

    def set_epoch(self, epoch: int) -> None:
        for source in self.sources:
            source.dataset.set_epoch(epoch)

    def __getitem__(
        self, request: MixtureSampleRequest | tuple[int, int, int, int]
    ) -> dict[str, Any]:
        if isinstance(request, tuple):
            if len(request) != 4:
                raise ValueError(
                    "Mixture tuple indices must be "
                    "(source, sample, num_views, sample_seed)"
                )
            request = MixtureSampleRequest(*(int(value) for value in request))
        if not isinstance(request, MixtureSampleRequest):
            raise TypeError("WeightedDatasetMixture requires a MixtureSampleRequest")
        if not 0 <= request.source_index < len(self.sources):
            raise IndexError(f"Dataset source index out of range: {request.source_index}")

        source = self.sources[request.source_index]
        sample = dict(
            source.dataset.get_sample(
                request.sample_index,
                request.num_views,
                request.sample_seed,
            )
        )
        validate_training_sample(sample, source.name)
        sample["dataset_source"] = source.name
        sample["dataset_type"] = source.type_name
        return sample


class WeightedMultiSourceBatchSampler(
    Sampler[list[MixtureSampleRequest]]
):
    """Build fixed-image batches drawn from one weighted source at a time."""

    def __init__(
        self,
        dataset: WeightedDatasetMixture,
        *,
        images_per_batch: int = 18,
        scene_counts: tuple[int, ...] | list[int] = (1, 2, 3, 6, 9),
        batches_per_epoch: int | None = None,
        recent_buffer_size: int = 10_000,
        seed: int = 42,
    ) -> None:
        if images_per_batch < 2:
            raise ValueError("images_per_batch must be at least 2")
        if not scene_counts:
            raise ValueError("scene_counts cannot be empty")
        if recent_buffer_size < 1:
            raise ValueError("recent_buffer_size must be positive")

        self.dataset = dataset
        self.images_per_batch = int(images_per_batch)
        self.scene_counts = tuple(int(value) for value in scene_counts)
        self.batches_per_epoch = (
            len(dataset) if batches_per_epoch is None else int(batches_per_epoch)
        )
        if self.batches_per_epoch < 1:
            raise ValueError("batches_per_epoch must be positive")
        self.recent_buffer_size = int(recent_buffer_size)
        self.seed = int(seed)
        self.epoch = 0

        for scene_count in self.scene_counts:
            if scene_count < 1 or self.images_per_batch % scene_count:
                raise ValueError(
                    f"scene_count={scene_count} must divide "
                    f"images_per_batch={self.images_per_batch}"
                )

        combinations_by_source: dict[
            int, tuple[tuple[int, int, np.ndarray], ...]
        ] = {}
        active_combinations: list[tuple[str, int, int]] = []
        for source_index, source in enumerate(dataset.sources):
            combinations: list[tuple[int, int, np.ndarray]] = []
            adapter = source.dataset
            for scene_count in self.scene_counts:
                views_per_scene = self.images_per_batch // scene_count
                if not adapter.min_views <= views_per_scene <= adapter.max_views:
                    continue
                eligible = np.asarray(
                    adapter.eligible_indices(views_per_scene), dtype=np.int64
                )
                if eligible.ndim != 1:
                    raise ValueError(
                        f"Dataset source {source.name!r} eligible_indices must "
                        "return a one-dimensional array"
                    )
                if len(np.unique(eligible)) != len(eligible):
                    raise ValueError(
                        f"Dataset source {source.name!r} returned duplicate "
                        "eligible indices"
                    )
                if len(eligible) and (
                    eligible.min() < 0 or eligible.max() >= len(adapter)
                ):
                    raise ValueError(
                        f"Dataset source {source.name!r} returned an out-of-range "
                        "eligible index"
                    )
                if len(eligible) >= scene_count:
                    combinations.append((scene_count, views_per_scene, eligible))
                    active_combinations.append(
                        (source.name, scene_count, views_per_scene)
                    )
            if combinations:
                combinations_by_source[source_index] = tuple(combinations)
            elif dataset.weights[source_index] > 0:
                raise ValueError(
                    f"Dataset source {source.name!r} has no feasible "
                    "scene×view combination"
                )

        active_sources = np.asarray(
            [
                index
                for index in combinations_by_source
                if dataset.weights[index] > 0
            ],
            dtype=np.int64,
        )
        if not len(active_sources):
            raise ValueError("No positive-weight dataset source can form a batch")
        active_weights = dataset.weights[active_sources]
        self._active_sources = active_sources
        self._active_weights = active_weights / active_weights.sum()
        self._combinations_by_source = combinations_by_source
        self.active_combinations = tuple(active_combinations)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        recent_queues = [deque() for _ in self.dataset.sources]
        recent_counts = [
            np.zeros(len(source.dataset), dtype=np.int64)
            for source in self.dataset.sources
        ]
        recent_capacities = [
            min(self.recent_buffer_size, len(source.dataset))
            for source in self.dataset.sources
        ]
        max_seed = np.iinfo(np.int64).max

        for _ in range(self.batches_per_epoch):
            if len(self._active_sources) == 1:
                source_index = int(self._active_sources[0])
            else:
                source_index = int(
                    rng.choice(self._active_sources, p=self._active_weights)
                )
            combinations = self._combinations_by_source[source_index]
            combination_index = int(rng.integers(len(combinations)))
            scene_count, views_per_scene, eligible = combinations[combination_index]

            counts = recent_counts[source_index]
            fresh = eligible[counts[eligible] == 0]
            candidates = fresh if len(fresh) >= scene_count else eligible
            selected = rng.choice(candidates, size=scene_count, replace=False)
            draw_seeds = rng.integers(
                0, max_seed, size=scene_count, dtype=np.int64
            )

            queue = recent_queues[source_index]
            capacity = recent_capacities[source_index]
            for sample_index in selected.tolist():
                while len(queue) >= capacity:
                    expired = queue.popleft()
                    counts[expired] -= 1
                queue.append(sample_index)
                counts[sample_index] += 1

            yield [
                MixtureSampleRequest(
                    source_index=source_index,
                    sample_index=int(sample_index),
                    num_views=views_per_scene,
                    sample_seed=int(draw_seed),
                )
                for sample_index, draw_seed in zip(selected, draw_seeds)
            ]


def collate_training_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate one source-homogeneous batch under the training contract."""
    if not samples:
        raise ValueError("Cannot collate an empty training batch")
    sources = {sample.get("dataset_source") for sample in samples}
    if len(sources) != 1:
        raise ValueError(
            "A training batch must contain samples from exactly one dataset source"
        )
    source_name = str(next(iter(sources)))
    for sample in samples:
        validate_training_sample(sample, source_name)
    sequence_lengths = {sample["images"].shape[0] for sample in samples}
    spatial_shapes = {tuple(sample["images"].shape[-2:]) for sample in samples}
    if len(sequence_lengths) != 1 or len(spatial_shapes) != 1:
        raise ValueError(
            f"Dataset source {source_name!r} returned incompatible samples in "
            "one batch"
        )
    return default_collate(samples)


def views_from_batch(batch: Mapping[str, Any]) -> list[dict[str, torch.Tensor]]:
    """Convert a contract-compliant batch to Arc's view-major input."""
    images = batch["images"]
    return [{"img": images[:, view_index]} for view_index in range(images.shape[1])]


_ROBOTWIN_CONFIG_KEYS = (
    "view",
    "min_views",
    "max_views",
    "min_interval",
    "max_interval",
    "reverse_probability",
    "frame_rate",
    "max_depth",
    "max_tcp_linear_speed",
    "max_tcp_angular_speed",
    "seed",
    "augment",
    "max_episodes",
)


def _build_robotwin_adapter(
    source_spec: Mapping[str, Any], config: Mapping[str, Any]
) -> SequenceDatasetAdapter:
    options = {
        key: config[key] for key in _ROBOTWIN_CONFIG_KEYS if key in config
    }
    configured_options = source_spec.get("options", {})
    if not isinstance(configured_options, Mapping):
        raise TypeError(
            f"Dataset source {source_spec.get('name')!r} options must be a mapping"
        )
    options.update(configured_options)
    root = options.pop("root", None)
    if root is None:
        raise ValueError(
            f"Dataset source {source_spec.get('name')!r} requires options.root"
        )
    if config.get("max_episodes") is not None:
        options["max_episodes"] = config["max_episodes"]
    return RoboTwin4RC(root, **options)


def build_training_dataset(config: Mapping[str, Any]) -> WeightedDatasetMixture:
    """Build a weighted adapter mixture from new or legacy configuration."""
    configured_sources = config.get("data_sources")
    if configured_sources is None:
        if not config.get("data_root"):
            raise ValueError("Configuration requires data_root or data_sources")
        configured_sources = (
            {
                "name": "robotwin",
                "type": "robotwin",
                "weight": 1.0,
                "options": {"root": config["data_root"]},
            },
        )
    if not isinstance(configured_sources, Sequence) or isinstance(
        configured_sources, (str, bytes)
    ):
        raise TypeError("data_sources must be a sequence of mappings")

    allowed_keys = {"name", "type", "weight", "options"}
    sources: list[DatasetSource] = []
    for source_index, spec in enumerate(configured_sources):
        if not isinstance(spec, Mapping):
            raise TypeError(f"data_sources[{source_index}] must be a mapping")
        unknown = sorted(set(spec).difference(allowed_keys))
        if unknown:
            raise ValueError(
                f"data_sources[{source_index}] has unknown keys: "
                f"{', '.join(unknown)}"
            )
        name = str(spec.get("name", "")).strip()
        type_name = str(spec.get("type", "")).strip()
        if not name or not type_name:
            raise ValueError(
                f"data_sources[{source_index}] requires non-empty name and type"
            )
        if type_name not in _DATASET_ADAPTER_REGISTRY:
            available = ", ".join(sorted(_DATASET_ADAPTER_REGISTRY)) or "none"
            raise ValueError(
                f"Unknown dataset adapter type {type_name!r}; available: {available}"
            )
        raw_weight = spec.get("weight", 1.0)
        if not isinstance(raw_weight, (int, float)) or not math.isfinite(
            float(raw_weight)
        ) or float(raw_weight) < 0:
            raise ValueError(
                f"Dataset source {name!r} weight must be finite and non-negative"
            )
        dataset = _DATASET_ADAPTER_REGISTRY[type_name](spec, config)
        sources.append(
            DatasetSource(
                name=name,
                type_name=type_name,
                dataset=dataset,
                weight=float(raw_weight),
            )
        )
    return WeightedDatasetMixture(sources)


register_dataset_adapter("robotwin", _build_robotwin_adapter)
