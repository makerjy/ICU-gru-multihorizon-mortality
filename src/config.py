from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    data_dir: Path = Path(os.environ.get("DATA_DIR", "./data")).expanduser().resolve()
    train_file: str = "train_processed.parquet"
    valid_file: str = "valid_processed.parquet"
    test_file: str = "test_processed.parquet"
    id_col: str = "stay_id"
    time_col: str = "t"
    event_col: str = "event"
    label_col: str = "_future_label"
    label_observable_col: str = "_label_observable"
    horizon_hours: int = 24
    drop_after_event: bool = True
    apply_cutoff_to_train: bool = False
    cutoff_hours: int = 24
    agg_mode: str = "max"
    target_recall: float = 0.80
    use_precomputed_labels: bool = True
    recompute_labels: bool = False
    feature_cols: list[str] = field(
        default_factory=lambda: [
            "HeartRate_std_6h",
            "GCS_Verbal",
            "SpO2_measured",
            "RespRate_std_6h",
            "SysBP",
            "GCS_Motor",
            "GCS_Total_mean_6h",
            "Temp_std_6h",
            "pH",
            "DiasBP_mean_6h",
            "MeanBP",
            "FiO2",
        ]
    )


@dataclass
class SequenceConfig:
    max_len: int | None = 120
    pad_value: float = -999.0


@dataclass
class ModelConfig:
    hidden: int = 64
    dropout: float = 0.2
    n_layers: int = 1


@dataclass
class TrainConfig:
    max_epochs: int = 50
    patience: int = 5
    min_delta: float = 1e-4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_train_cuda: int = 128
    batch_train_cpu: int = 64
    batch_eval_cuda: int = 256
    batch_eval_cpu: int = 128
    grad_clip: float = 5.0


@dataclass
class OutputConfig:
    artifacts_dir: Path = Path("./artifacts")
    output_dir: Path = Path("./output")


@dataclass
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
