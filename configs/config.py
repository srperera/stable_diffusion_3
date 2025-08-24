from types import NoneType
import torch.nn as nn
from datetime import datetime
from typing import Any, List, Callable
from dataclasses import dataclass, field


################################################################################################
################################################################################################
@dataclass
class ExperimentConfig:
    experiment_name: str = "fashionmnist_sd3"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb_tracking: dict[str, Any] = field(
        default_factory=lambda: {
            "group": "sd3_test",
            "name": f"sd3",
            "mode": "offline",
            "resume": False,
            "tags": [None],
            "notes": "None",
        }
    )
    save_dir: str = f"runs/{experiment_name}/{timestamp}"
    resume_training: bool = False


################################################################################################
################################################################################################
@dataclass
class MetricsConfig:
    # TODO: Not Used Yet
    metric_name: str = "accuracy"
    calculate_metrics: bool = False


################################################################################################
################################################################################################
@dataclass
class ModelConfig:
    # small and large vision model names
    clip_s_name: str = "facebook/metaclip-b16-400m"
    clip_l_name: str = "facebook/metaclip-l14-400m"
    compile_clip: bool = False
    device: str = "cuda:0"

    # language model name
    llm_name: str = "google/gemma-2-2b"
    compile_llm: bool = False

    # vae model names
    vae_name: str = "black-forest-labs/FLUX.1-schnell"
    compile_vae: bool = False

    # mmdit block parameters
    num_blocks: int = 8
    image_dim: int = 2304  # same as max(llm_dim, clip_s_dim, clip_l_dim)
    text_dim: int = 2304  # same as max(llm_dim, clip_s_dim, clip_l_dim)
    condition_dim: int = 1280  # 512 + 768
    act_fn: str = "silu"
    p_drop: float = 0.1
    num_heads: int = 8
    bias: bool = True
    qkv_bias: bool = True
    qk_norm: bool = False
    norm_layer: str = "rmsnorm"
    hidden_scale: float = 2.0  # scale for hidden dimension in MLPs

    # image params
    vae_input_size: int = 256
    vae_output_size: int = 32
    vae_output_dim: int = 16
    patch_size: int = 2
    patch_num_heads: int = 1
    patch_spatial_dims: int = 2
    patch_dropout_rate: float = 0.0
    patch_pos_embed_type: str = "sincos"  # or "learnable"


################################################################################################
################################################################################################
@dataclass
class TrainingConfig:
    num_epochs: int = 110
    batch_size: int = 16  # per gpu 48
    num_workers: int = 0
    drop_last: bool = True
    pin_memory: bool = False
    device: str = "cuda:0"
    len_dataset: int = 20000
    shuffle: bool = True
    accelerate_init_kwargs: dict[str, Any] = field(
        default_factory=lambda: {
            "gradient_accumulation_steps": 1,
            "mixed_precision": "no",  # choices: "no", "fp16", "bf16", "fp8",
            # "fsdp_plugin": None,  # can make this a dict later too for fsdp params
        }
    )

    loss_type: str = "mse"
    loss_args: dict[str, Any] = field(
        default_factory=lambda: {
            "enable_label_smoothing": False,
            "label_smoothing": 0.1,
        }
    )

    optimizer_type: str = "adamw"
    optimizer_args: dict[str, Any] = field(
        default_factory=lambda: {
            "lr": 1e-4,
            "weight_decay": 0.01,
            "epsilon": 1e-8,  # research shows this is better than 1e-5 AllenInstitue
        }
    )

    scheduler_type: str = "warmup_cosine"
    scheduler_args: dict[str, Any] = field(
        default_factory=lambda: {
            "num_warmup_steps": 1000,
            "num_tranining_steps": 10000,
            "num_cycles": 0.5,
        }
    )

    resume_training: bool = False
    resume_training_args: dict[str, Any] = field(
        default_factory=lambda: {
            "monitor_checkpoint_path": None,
        }
    )


################################################################################################
################################################################################################
@dataclass
class DatasetConfig:
    dataset_type: str = "cifar"
    # data_path: str = None
    dataset_names: List[str] = field(
        default_factory=lambda: [
            "tqa",
            "vsr",
            "ai2d",
            "scienceqa",
            # "aokvqa",
            # "visual7w",
        ]
    )
    train_percent: float = 0.90
    shuffle: bool = True


################################################################################################
################################################################################################
@dataclass
class DiffusionConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
