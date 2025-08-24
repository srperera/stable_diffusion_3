from typing import Dict
import torch.optim as optim
from transformers import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from torch.optim.lr_scheduler import LRScheduler


#############################################################################################
#############################################################################################
def build_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    scheduler_args: Dict,
) -> LRScheduler:
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_args["t_0_epochs"],
            T_mult=scheduler_args["t_mult"],
            eta_min=scheduler_args["min_lr"],
            last_epoch=-1,
            verbose=False,
        )
        return scheduler

    elif scheduler_type == "warmup_cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=scheduler_args["num_warmup_steps"],
            num_training_steps=scheduler_args["num_tranining_steps"],
            num_cycles=scheduler_args["num_cycles"],
        )
        return scheduler

    elif scheduler_type == "warmup_cosine_with_restarts":
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=scheduler_args["num_warmup_steps"],
            num_training_steps=scheduler_args["num_tranining_steps"],
            num_cycles=scheduler_args["num_cycles"],
        )
        return scheduler

    elif scheduler_type == "warmup_poly":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=scheduler_args["num_warmup_steps"],
            num_training_steps=scheduler_args["num_tranining_steps"],
            lr_end=scheduler_args["lr_end"],
            power=scheduler_args["power"],
        )
        return scheduler

    else:
        raise NotImplementedError("scheduler not implemented")
