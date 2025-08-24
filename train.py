import os
import json
import torch
import hydra
import logging
import argparse
from torch.nn import Module
from typing import Set, Type
from termcolor import colored
from criterion import criterion
from omegaconf import OmegaConf
import torch.distributed as dist
from accelerate import Accelerator
from trainer import RFDiffusionTrainer
from utils.utils import count_parameters
from configs.config import DiffusionConfig
from optimizers import optimizers, schedulers
from hydra.core.config_store import ConfigStore
from architectures.build_architecture import build_architecture
from augmentations.vision_augmentation import get_image_processor
from dataloaders.build_dataloader import build_dataset, build_dataloader
from architectures.mmdit_blocks import MMDiTBlock
from accelerate.utils import FullyShardedDataParallelPlugin

cs = ConfigStore.instance()
cs.store(name="rf_diff_config", node=DiffusionConfig)

logging.getLogger("accelerate.checkpointing").setLevel(logging.WARNING)
logging.getLogger("accelerate.accelerator").setLevel(logging.WARNING)


# TODO: Exdend this class such that we can easily switch betwen single gpu, single node, multi node
###################################################################################################
###################################################################################################
def make_custom_policy(
    transformer_classes: Set[Type[Module]],
    min_num_params: int = int(1e4),
):
    def policy_fn(module: Module, recurse: bool, nonwrapped_numel: int) -> bool:
        if not recurse:
            raise ValueError("HERE")
            return False
        if isinstance(module, tuple(transformer_classes)):
            param_count = sum(p.numel() for p in module.parameters())
            raise ValueError("HERE")
            if param_count >= min_num_params:
                print(
                    f"[FSDP] Wrapping {module.__class__.__name__} with {param_count} params"
                )
                # raise ValueError("HERE")
                return True
        return False

    return policy_fn


###################################################################################################
###################################################################################################
def init_dist(args):
    node_list = os.environ["SLURM_NODELIST"]
    num_gpus = torch.cuda.device_count()
    args.global_rank = int(os.environ["SLURM_PROCID"])
    args.local_rank = args.global_rank % num_gpus
    # args.local_rank = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ["SLURM_NTASKS"])
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["RANK"] = str(args.global_rank)
    addr = os.environ["MASTER_ADDR"]
    port = os.environ["MASTER_PORT"]
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)
    print(
        f"proc_id: {args.global_rank}; local_rank: {args.local_rank}; ntasks: {args.world_size};\n"
        f"node_list: {node_list}; num_gpus: {num_gpus}; addr: {addr}; port: {port}"
    )
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    return args


###################################################################################################
###################################################################################################
def display_info(config, accelerator, trainset, valset, model):
    # print experiment info
    accelerator.print(f"-------------------------------------------------------")
    accelerator.print(f"[info]: Experiment Info")
    accelerator.print(
        f"[info] ----- Project: {colored(config.experiment.experiment_name, color='red')}"
    )
    accelerator.print(
        f"[info] ----- Group: {colored(config.experiment.wandb_tracking["group"], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Name: {colored(config.experiment.wandb_tracking["name"], color='red')}"
    )
    accelerator.print(
        f"[info] ----- Batch Size: {colored(config.training.batch_size, color='red')}"
    )
    accelerator.print(
        f"[info] ----- Num Epochs: {colored(config.training.num_epochs, color='red')}"
    )
    accelerator.print(
        f"[info] ----- Loss: {colored(config.training.loss_type, color='red')}"
    )
    accelerator.print(
        f"[info] ----- Optimizer: {colored(config.training.optimizer_type, color='red')}"
    )
    accelerator.print(
        f"[info] ----- Train Dataset Size: {colored(len(trainset), color='red')}"
    )
    accelerator.print(
        f"[info] ----- Test Dataset Size: {colored(len(valset), color='red')}"
    )

    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(
        f"[info] ----- Distributed Training: {colored('True' if torch.cuda.device_count() > 1 else 'False', color='red')}"
    )
    accelerator.print(
        f"[info] ----- Params: {colored(count_parameters(model), color='red')}"
    )
    accelerator.print(f"-------------------------------------------------------")

    # log to experiment.log file
    log_filepath = os.path.join(config.experiment.save_dir, "experiment.log")
    with open(log_filepath, "a") as log_file:
        if config.experiment.resume_training:
            log_file.write("\nResuming Experiment\n")
        else:
            log_file.write("\nStarting Experiment\n")
        log_file.write("\nExperiment Config\n")
        log_file.write(
            json.dumps(OmegaConf.to_container(config, resolve=True), indent=4) + "\n"
        )
        log_file.write(
            f"\n\n-----------------------------------------------------------------"
        )
        log_file.write(f"\n[info]: Experiment Overview")
        log_file.write(f"\n[info] ----- Project: {config.experiment.experiment_name}")
        log_file.write(
            f"\n[info] ----- Group: {config.experiment.wandb_tracking["group"]}"
        )
        log_file.write(
            f"\n[info] ----- Name: {config.experiment.wandb_tracking["name"]}"
        )
        log_file.write(f"\n[info] ----- Batch Size: {config.training.batch_size}")
        log_file.write(f"\n[info] ----- Num Epochs: {config.training.num_epochs}")
        log_file.write(f"\n[info] ----- Loss: {config.training.loss_type}")
        log_file.write(f"\n[info] ----- Optimizer: {config.training.optimizer_type}")
        log_file.write(f"\n[info] ----- Train Dataset Size: {len(trainset)}")
        log_file.write(f"\n[info] ----- Test Dataset Size: {len(valset)}")
        log_file.write(f"\n[info] ----- Params: {count_parameters(model)}")
        log_file.write(
            f"\n[info] ----- Distributed Training: {'True' if torch.cuda.device_count() > 1 else 'False'}"
        )
        log_file.write(
            f"\n-----------------------------------------------------------------\n"
        )


###################################################################################################
###################################################################################################
def build_directories(config) -> None:
    # create necessary directories
    resume_training = config.experiment.resume_training
    root_exp_dir = config.experiment.save_dir
    models_dir = os.path.join(root_exp_dir, "models")

    # prevent experiment directory over ride
    if os.path.isdir(root_exp_dir) and resume_training == False:
        raise ValueError("checkpoint exits -- preventing file override -- rename file")
    elif os.path.isdir(root_exp_dir) and resume_training == True:
        pass
    else:
        os.makedirs(models_dir)


###################################################################################################
###################################################################################################
@hydra.main(config_name="rf_diff_config", version_base="1.4")
def main(config):

    # fsdp parameters
    wrap_policy = make_custom_policy(transformer_classes={MMDiTBlock}, min_num_params=0)
    fsdp_plugin = FullyShardedDataParallelPlugin(
        reshard_after_forward="FULL_SHARD",  # or "SHARD_GRAD_OP"
        backward_prefetch="BACKWARD_PRE",  # or "BACKWARD_POST"
        # mixed_precision_policy="NO",  # or "FP16", "BF16"
        auto_wrap_policy=wrap_policy,  # or a custom function
        forward_prefetch=False,
        use_orig_params=True,
        cpu_offload=True,
        cpu_ram_efficient_loading=True,
        sync_module_states=True,
        state_dict_type="SHARDED_STATE_DICT",  # "SHARDED_STATE_DICT",
    )

    # set up accelerator
    accelerator = Accelerator(
        fsdp_plugin=fsdp_plugin,
        **config.training.accelerate_init_kwargs,
    )

    # TODO: move this init trackers into the config file
    accelerator.init_trackers(project_name="test", config=config)
    accelerator.print(f"CONFIG -- {colored(config, color='yellow')}")

    # build model
    accelerator.print(f"INFO -- {colored('building model', color='green')}")
    model = build_architecture(config.model)

    # build dataset
    accelerator.print(f"INFO -- {colored('configuring datasets', color='green')}")
    train_dataset = build_dataset(
        config, True, get_image_processor(config.model.vae_input_size)
    )
    val_dataset = build_dataset(
        config, False, get_image_processor(config.model.vae_input_size)
    )

    # build dataloaders
    accelerator.print(f"INFO -- {colored('building dataloaders', color='green')}")
    train_dataloader = build_dataloader(config.training, train_dataset)
    val_dataloader = build_dataloader(config.training, val_dataset)

    # build loss function, optimizer and lr scheduler
    accelerator.print(
        f"INFO -- {colored('building loss function, optimizer and scheduler', color='green')}"
    )

    loss = criterion.build_loss_fn(config.training.loss_type, config.training.loss_args)

    optimizer = optimizers.build_optimizer(
        model["model"], config.training.optimizer_type, config.training.optimizer_args
    )

    steps_per_epoch = len(train_dataset) / (
        config.training.batch_size * torch.cuda.device_count()
    )
    scheduler_args = {
        "num_warmup_steps": steps_per_epoch * 5,
        "num_tranining_steps": steps_per_epoch * 100,
        "num_cycles": 0.5,
    }
    scheduler = schedulers.build_scheduler(
        optimizer, config.training.scheduler_type, scheduler_args
    )

    # prepare accelerator objects
    accelerator.print(f"INFO -- {colored('preparing accelerator', color='green')}")
    diffusion_model = accelerator.prepare_model(model=model["model"])
    caption_encoder = accelerator.prepare_model(model=model["caption_encoder"])
    vae = accelerator.prepare_model(model=model["vae"])
    optimizer = accelerator.prepare_optimizer(optimizer=optimizer)
    train_dataloader = accelerator.prepare_data_loader(data_loader=train_dataloader)
    val_dataloader = accelerator.prepare_data_loader(data_loader=val_dataloader)
    training_scheduler = accelerator.prepare_scheduler(scheduler=scheduler)

    accelerator.print(f"INFO -- {colored('building directories', color='green')}")
    if accelerator.is_main_process:
        build_directories(config)
        display_info(
            config,
            accelerator,
            train_dataset,
            val_dataset,
            accelerator.unwrap_model(diffusion_model),
        )

    # run trainings
    accelerator.print(f"INFO -- {colored('starting train/val', color='green')}")
    trainer = RFDiffusionTrainer(
        config=config,
        diffusion_model=diffusion_model,
        caption_encoder=caption_encoder,
        vae=vae,
        time_sampler=model["time_sampler"],
        optimizer=optimizer,
        criterion=loss,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        training_scheduler=training_scheduler,
        accelerator=accelerator,
        metrics=None,
    )
    trainer.train()

    # done training
    accelerator.end_training()


###################################################################################################
###################################################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    args = parser.parse_args()
    args = init_dist(args)
    main()

###################################################################################################
###################################################################################################
