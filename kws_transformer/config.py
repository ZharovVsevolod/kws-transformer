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

@dataclass
class Data:
    time_window: int
    frequency: int
    patch_size_t: int
    patch_size_f: int

@dataclass
class Training:
    batch: int
    lr: float

@dataclass
class Params:
    model: Model
    data: Data
    training: Training