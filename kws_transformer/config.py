from dataclasses import dataclass
from typing import Literal

@dataclass
class Model:
    name: str
    layers: int
    heads: int
    mlp_dim: int
    num_output_classes: int
    embedding_dim: int
    norm_type: Literal["postnorm", "prenorm"]
    dropout: float

@dataclass
class Data:
    path_to_data: str
    sample_rate: int
    time_window: int
    frequency: int
    patch_size_time: int
    patch_size_freq: int
    n_mels: int
    hop_length: int
    dataset_size: int

@dataclass
class Training:
    batch: int
    lr: float
    epochs: int
    model_path: str
    wandb_path: str
    project_name: str
    train_name: str
    save_best_of: int
    checkpoint_monitor: Literal["val_loss", "val_acc"]
    early_stopping_patience: int

@dataclass
class Scheduler_train:
    type_of_scheduler: Literal["ReduceOnPlateau", "OneCycleLR"]
    patience_reduce: int
    factor_reduce: float
    lr_coef_cycle: int

@dataclass
class Params:
    model: Model
    data: Data
    training: Training
    scheduler_train: Scheduler_train