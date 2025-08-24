import os
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
from termcolor import colored
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from utils.utils import combine_tensors
from typing import Dict, Callable, Tuple
from torch.optim.lr_scheduler import LRScheduler


#############################################################################################
# Class: Rectified Flow Diffusion Transformer Trainer
# # TODO: Untested # Move File To Folder?, Also Fix _save_checkpoint function to work with
# single gpu, single node, multinode
#############################################################################################
class RFDiffusionTrainer:
    def __init__(
        self,
        config: Dict,
        diffusion_model: torch.nn.Module,
        caption_encoder: torch.nn.Module,
        vae: torch.nn.Module,
        time_sampler: Callable,
        optimizer: Optimizer,
        criterion: torch.nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        training_scheduler: LRScheduler,
        metrics=None,
        accelerator=None,
    ) -> None:
        """
        Initialize Segmentation Trainer Class.

        Args:
            config (Dict): experiment config dictionary
            model (torch.nn.Module): pytorch model
            optimizer (Optimizer): optimizer
            criterion (torch.nn.Module): loss function
            train_dataloader (DataLoader): train data loader
            val_dataloader (DataLoader): val data loader
            training_scheduler (LRScheduler): scheduler
            metrics (_type_, optional): list of callable meterics. Defaults to None.
            accelerator (_type_, optional): huggingface accelerator object. Defaults to None.
        """

        # set up trainer
        self.config = config
        self.accelerator = accelerator

        # trainer components
        self.diffusion_model = diffusion_model
        self.caption_encoder = caption_encoder  # .to(self.diffusion_model.device)
        self.vae = vae  # .to(self.diffusion_model.device)
        self.time_sampler = time_sampler
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.scheduler = training_scheduler
        self.tracker = self.accelerator.get_tracker("wandb")

        # metrics
        if metrics:
            self.metrics = metrics

        # parameters to monitor
        self.monitor = {
            "current_epoch": 0,
            "lr": [],
            # train losses
            "train_loss": [],
            "best_train_loss": np.inf,
            # val losses
            "val_loss": [],
            "best_val_loss": np.inf,
            # metrics [only one is used to track performance]
            # "val_metrics": {metric: [] for metric in self.metrics},
            "best_val_metric": 0.0,
        }

        # configure trainer
        self._configure_trainer()

    def _configure_trainer(self) -> None:
        """
        Sets up the SegmentationTrainer Class
        Creates class variables for commonly used variables
        Raises:
            ValueError: _description_
        """
        self.accelerator.print("configuring trainer")
        # TODO: loading checkpoint to resume.. not verified
        # update self.monitor from a previously saved run
        if self.config.training.resume_training:
            monitor_ckpt_path = self.config.training.resume_training_args[
                "monitor_checkpoint_path"
            ]
            if os.path.isfile(monitor_ckpt_path):
                self.monitor = torch.load(monitor_ckpt_path)
            else:
                raise ValueError("invalid monitor checkpoint path")
            self.current_epoch = self.monitor["current_epoch"]
            self._load_checkpoint()
        else:
            self.current_epoch = 0

        self.num_epochs = self.config.training.num_epochs
        self.exp_save_dir = self.config.experiment.save_dir

    def add_noise(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adds noise to a rectified flow diffusion model.
        Args:
            x (torch.Tensor): images. [B, C, H, W]
            t (torch.Tensor): time steps b/w (0, 1). [B]
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        # expand time steps so we can broadcast same time step to each batch
        t = t[:, None, None, None]

        # sample gaussian noise
        epsilon = torch.randn_like(x).to(x.device)

        # modify noise based on time step
        x_t = (1 - t) * x + (t * epsilon)

        return x_t, epsilon

    def mask_input_text(
        self,
        c,
        c_pooled,
        mask_prob: float = 0.1,
        c_pooled_prob: float = 0.1,
        clip_prob: float = 0.2,
        llm_prob: float = 0.2,
    ):

        # get tensors from encoded text
        c_masked = c.clone()  # [B, 154, Dimc]
        c_pooled_masked = c_pooled.clone()  # [B, 154, Dpool]

        # Batch Size
        B, S, _ = c_masked.shape
        device = c_masked.device

        # figure out which batches wont get the text information
        c_pooled_mask = torch.randn(B)
        clip_prob_mask = torch.randn(B)
        llm_prob_mask = torch.randn(B)

        # get batch level mask for each.
        # ie: masked out batches will not see this information.
        c_pooled_mask = (c_pooled_mask < c_pooled_prob).to(torch.bool).to(device)
        clip_prob_mask = (clip_prob_mask < clip_prob).to(torch.bool).to(device)
        llm_prob_mask = (llm_prob_mask < llm_prob).to(torch.bool).to(device)

        # mask out these tensors
        c_pooled_masked[c_pooled_mask] *= 0  # final pooled into the model **
        c_masked[clip_prob_mask, :77] *= 0
        c_masked[llm_prob_mask, 77:] *= 0

        # above operations control which items the model does not see
        # which teaches the model to not over rely on one set of guidance features
        # next we need to mask out tokens that the model does see, so we can
        # use that as a guidance sigal for training.
        c_targets = c.clone()

        # generate Random mask per token
        seq_mask_probs = torch.rand((B, S), device=device)

        # get the mask for the active sequences ie for the ones the model sees
        # less than 0.25 True .. greater than 0.25 False
        txt_loss_mask = seq_mask_probs < mask_prob  # [B, S]

        # with & we zero out (ie: set to False) the batches where theres no text
        # because we want the text loss mask to only be for batches with text.
        # here False (0) = no text loss is computed, True (0) is text loss is computed.
        txt_loss_mask[:, :77] = txt_loss_mask[:, :77] & clip_prob_mask[:, None]
        txt_loss_mask[:, 77:] = txt_loss_mask[:, 77:] & llm_prob_mask[:, None]

        # final input c vector into the model **
        # we combine the batch masked (c_masked) with the text loss mask
        # to get the final masked input
        # here what we are doing is we initially random remove some batches
        # next for the remaining batches we randomly mask out some tokens
        # this is c_final.... BUT now .. what is the target?
        # we use ~ because we want to zero out the locations where its True
        c_final = c_masked * ~txt_loss_mask[:, :, None]

        # because we only want to calc loss on tokens that present
        # text loss mask is 0 for no text loss is computed, and 1 for text loss is computed
        # by multiplying original c with this .. we zero out the locations
        # we want to ignore.
        c_targets = c_targets * txt_loss_mask[:, :, None]

        out = {
            "c_masked": c_final,  # [B, 154, Dim]
            "c_targets": c_targets,  # [B, 154, Dim]
            "c_pooled_masked": c_pooled_masked,  # [B, DimPool]
        }

        return out

    def _train_step(self):
        """
        Runs a Single Training Epoch
        """
        # Initialize the training loss for the current epoch
        epoch_avg_loss = 0.0
        img_avg_loss = 0.0
        text_avg_loss = 0.0

        # set model to train
        self.diffusion_model.train()
        self.vae.eval()
        self.caption_encoder.eval()

        # run training
        progress_bar = tqdm(
            desc=f"Epoch: {self.monitor['current_epoch']} -- Training",
            total=len(self.train_loader),
            leave=True,
            bar_format="{l_bar}{bar}{r_bar}",
            ncols=120,
            colour="MAGENTA",
            disable=not self.accelerator.is_main_process,
        )
        for index, batch in enumerate(self.train_loader):
            with self.accelerator.accumulate(self.diffusion_model):
                images = batch["image_data"]
                text = batch["text_data"]

                self.optimizer.zero_grad()

                # sample time
                t = self.time_sampler(images.shape[0]).to(images.device)

                # encode text
                with torch.no_grad():
                    encoded_text = self.caption_encoder.forward(text, images.device)
                c, c_pooled = combine_tensors(encoded_text)

                # mask text
                masked = self.mask_input_text(c, c_pooled)
                c_masked = masked["c_masked"]
                c_pooled_masked = masked["c_pooled_masked"]
                c_target = masked["c_targets"]

                # encode images
                with torch.no_grad():
                    x = self.vae.forward(images).to(images.device)

                # add noise to latent image
                x_noised, epsilon = self.add_noise(x.detach(), t.detach())

                # pass info to mmdit
                # pred_vel = self.diffusion_model.forward(x_noised, c, c_pooled, t)
                pred_vel, text_out = self.diffusion_model.forward(
                    x_noised, c_masked, c_pooled_masked, t
                )

                # get targets by calculating vector field from noise to original image
                targets = epsilon - x

                # calculate loss
                image_loss = self.criterion(pred_vel, targets)
                text_loss = self.criterion(text_out, c_target)
                loss = image_loss + (text_loss * 1.0)

                # backward pass
                self.accelerator.backward(loss)

                # clip gradients
                self.accelerator.clip_grad_norm_(
                    self.diffusion_model.parameters(), max_norm=1.0
                )

                # update gradients & step scheduler
                # after backwards, since it accumulates gradients
                # from all nodes and devices.
                # if self.accelerator.is_main_process:
                self.optimizer.step()
                self.scheduler.step()

                # update loss
                epoch_avg_loss += loss.item()
                img_avg_loss += image_loss.item()
                text_avg_loss += text_loss.item()

                # update progress bar
                if self.accelerator.is_main_process:
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        tr_loss=f"{(epoch_avg_loss / (index + 1)):.5f}",
                        img=f"{(img_avg_loss / (index + 1)):.5f}",
                        txt=f"{(text_avg_loss / (index + 1)):.5f}",
                        lr=f"{self.scheduler.get_last_lr()[0]:.5f}",
                    )
                    progress_bar.refresh()

        # calculate and update epoch average loss accross all nodes
        global_avg_loss = self.accelerator.reduce(
            torch.tensor(epoch_avg_loss, device=self.accelerator.device),
            reduction="mean",
        ).item()
        self.monitor["train_loss"].append(global_avg_loss / (index + 1))

        # save learning rate
        self.monitor["lr"].append(self.scheduler.get_last_lr()[0])

    def _val_step(self):
        """
        Runs a Single Validation Epoch
        """
        # Initialize the training loss for the current Epoch
        epoch_avg_loss = 0.0

        # set model to eval mode
        self.diffusion_model.eval()
        self.vae.eval()
        self.caption_encoder.eval()

        # run validation
        progress_bar = tqdm(
            desc=f"Epoch: {self.monitor['current_epoch']} -- Validation",
            total=len(self.val_loader),
            leave=True,
            bar_format="{l_bar}{bar}{r_bar}",
            ncols=100,
            colour="YELLOW",
            disable=not self.accelerator.is_main_process,
        )
        with torch.no_grad():
            for index, batch in enumerate(self.val_loader):
                images = batch["image_data"]
                text = batch["text_data"]

                # sample time
                t = self.time_sampler(images.shape[0]).to(images.device)

                # encode text
                encoded_text = self.caption_encoder.forward(text, images.device)
                c, c_pooled = combine_tensors(encoded_text)

                # encode images
                x = self.vae.forward(images)

                # add noise to latent image
                x_noised, epsilon = self.add_noise(x.detach(), t)

                # pass info to mmdit
                pred_vel, _ = self.diffusion_model.forward(x_noised, c, c_pooled, t)

                # get targets by calculating vector field from noise to original image
                targets = epsilon - x

                # calculate loss
                loss = self.criterion(pred_vel, targets)

                # if self.calculate_metrics:
                #     raise ValueError("Calulate Metrics Not Working For Now")

                # update loss for the current batch
                epoch_avg_loss += loss.item()

                # update progress bars
                if self.accelerator.is_main_process:
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        val_loss=f"{(epoch_avg_loss / (index + 1)):.5f}",
                    )
                    progress_bar.refresh()

        # calculate and update epoch average loss
        global_avg_loss = self.accelerator.reduce(
            torch.tensor(epoch_avg_loss, device=self.accelerator.device),
            reduction="mean",
        ).item()
        self.monitor["val_loss"].append(global_avg_loss / (index + 1))

        # update metrics
        # if self.calculate_metrics:
        #     for name, metric in self.metrics.items():
        #         self.monitor["val_metrics"][name].append(metric.aggregate()["average"])
        #         metric.reset()

    def _update_metrics(self) -> None:
        """
        Updates Training Loss, Validation Loss and
        the single metric we are measuring performance by.
        denoted by: self.tracked_metric

        Note: Metrics are maximied, if there is a metric
        that needs to be minimized then modifications are needed.
        """

        # update training loss
        if self.monitor["train_loss"][-1] < self.monitor["best_train_loss"]:
            self.monitor["best_train_loss"] = self.monitor["train_loss"][-1]

        # update validation loss
        if self.monitor["val_loss"][-1] < self.monitor["best_val_loss"]:
            self.monitor["best_val_loss"] = self.monitor["val_loss"][-1]
            self.improved_val_loss = True

        # update metric we are monitoring performance by
        # this can only be a single metric.
        # ex: we can calculate iou and dice but will only track performance using iou
        # TODO: untested
        # if self.calculate_metrics:
        # raise ValueError("Calculate Metrics Untested")
        # value = self.monitor["val_metrics"][self.tracked_metric][-1]
        # if value > self.monitor["best_val_metric"]:
        #     self.monitor["best_val_metric"] = value
        #     self.improved_val_metric = True

    def _load_checkpoint(self) -> None:
        # TODO: Untested
        self.accelerator.load_state(
            self.config["training_parameters"]["resume_training"]["checkpoint_path"]
        )

    def _save_checkpoints(self) -> None:
        """
        Saves Model Checkpoints for:
            1. Best Validation Loss Model
            2. Best Metric Model (if we are calculating metrics)
            3. Model for the latest run

        Additionally saves the self.monitor dict as a .pth file.
        This file contains all the run progress and can help if we
        want resume runs and other downstream plotting.
        """
        get_file_path = lambda fname: os.path.join(self.exp_save_dir, "models", fname)
        # options = StateDictOptions(
        #     full_state_dict=True,
        #     cpu_offload=False,
        # )

        # save if validation loss improved
        if self.improved_val_loss:
            # fsdp_state_dict, fsdp_optim_state_dict = get_state_dict(
            #     self.diffusion_model, self.optimizer, options=options
            # )
            # unwrapped_model = self.accelerator.unwrap_model(self.diffusion_model)
            self.accelerator.save_model(  # this functino work
                self.diffusion_model,
                get_file_path("best_loss"),
                safe_serialization=False,
            )
            # self.accelerator.save_state(
            #     get_file_path("best_metric"),
            #     safe_serialization=False,
            # )
            # torch.save(
            #     self.accelerator.get_state_dict(self.diffusion_model),
            #     get_file_path("model.pth"),
            # )
            # torch.save(self.optimizer.state_dict(), get_file_path("optimizer.pth"))
            # writer = FileSystemWriter(get_file_path("best_loss"))
            # torch.distributed.checkpoint.save(fsdp_state_dict, writer)

        # update metric we are monitoring performance by
        # this can only be a single metric.
        # ex: we can calculate iou, dice but will only track performance using iou
        # if self.calculate_metrics:
        #     if self.improved_val_metric:
        #         self.accelerator.save_state(
        #             get_file_path("best_metric"),
        #             safe_serialization=False(),
        #         )

        # save latest model
        # fsdp_state_dict, fsdp_optim_state_dict = get_state_dict(
        #     self.diffusion_model, self.optimizer, options=options
        # )
        # torch.save(fsdp_state_dict, get_file_path("latest/model.pth"))
        # torch.save(fsdp_optim_state_dict, get_file_path("latest/optimizer.pth"))

        # save tracked metrics ie: monitor
        # if self.accelerator.is_main_process:
        # torch.save(self.monitor, get_file_path("monitor_ckpt.pth"))
        self.accelerator.print(f"Saved Checkpoint -- {colored(True, color='red')}")

    def _log_metrics(self, log_best: bool = False) -> None:
        """
        Logs Loss and Metric Information to Console and experiment.log File.
        If log_best = True, will only log the best loss and metrics.
        This should only be set to True at the final epoch so a neat
        results are printed at the end of the log file.

        Args:
            log_best (bool, optional): _description_. Defaults to False.
        """

        if not log_best:
            epoch = str(self.monitor["current_epoch"]).zfill(len(str(self.num_epochs)))
            train_loss = np.around(self.monitor["train_loss"][-1], 5)
            val_loss = np.around(self.monitor["val_loss"][-1], 5)

            # log to file
            log_filepath = os.path.join(self.exp_save_dir, "experiment.log")
            with open(log_filepath, "a") as log_file:
                try:
                    log_file.write("\n\n")
                    log_file.write("#" * 80)
                    log_file.write(
                        f"\nDate/Time -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    log_file.write(f"\nEpoch -- {epoch} / {self.num_epochs}")
                    log_file.write(f"\nTrain Loss -- {train_loss} || ")
                    log_file.write(f"Val Loss -- {val_loss} || ")
                    log_file.write(f"LR -- {self.monitor['lr'][-1]} ")

                    # for metric_name, values in self.monitor["val_metrics"].items():
                    #     log_file.write(f" \n{metric_name} -- {values[-1]:.8f}")

                    log_file.write(
                        f"\nImproved Val Loss -- {self.improved_val_loss} || "
                    )
                    # if self.calculate_metrics:
                    #     log_file.write(
                    #         f"Improved Val Metric -- {self.improved_val_metric}"
                    #     )

                    log_file.flush()
                except:
                    self.accelerator.print(f"Logging Error at Epoch -- {epoch}")

            # print to console
            # metric = np.around(self.monitor["val_metrics"][self.tracked_metric][-1], 5)
            self.accelerator.print(
                f"Epoch -- {colored(epoch, color='green')} || "
                f"Train Loss -- {colored(train_loss, color='green')} || "
                f"Val Loss-- {colored(val_loss, color='green')} || "
                f"Improved Val Loss -- {colored(str(self.improved_val_loss), color='magenta')} ||"
                f"LR -- {colored(self.monitor['lr'][-1], color='green')} || "
            )
            # if self.calculate_metrics:
            #     self.accelerator.print(
            #         f"Improved Val Loss -- {colored(str(self.improved_val_loss), color='magenta')} || "
            #         # f"Improved Val Metric -- {colored(str(self.improved_val_metric), color='magenta')}\n"
            #     )
            # else:
            # self.accelerator.print(
            #     f"Improved Val Loss -- {colored(str(self.improved_val_loss), color='magenta')} || "
            # )

        # prints only the best results
        # should be requested at the end of the run
        if log_best:
            train_loss = np.around(self.monitor["best_train_loss"], 5)
            val_loss = np.around(self.monitor["best_val_loss"], 5)
            log_filepath = os.path.join(self.exp_save_dir, "experiment.log")
            with open(log_filepath, "a") as log_file:
                try:
                    log_file.write("\n\n")
                    log_file.write("#" * 80)
                    log_file.write("\nFinal Results")
                    log_file.write(
                        f"\nDate/Time -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    log_file.write(f"Best Train Loss -- {train_loss} || ")
                    log_file.write(f"Val Loss -- {val_loss} || ")

                    for metric_name, values in self.monitor["val_metrics"].items():
                        values = np.around(values, 8)
                        values = np.max(values)
                        log_file.write(f"\n{metric_name} -- {values}")

                    log_file.write("#" * 80)
                    log_file.flush()
                except:
                    self.accelerator.print(f"Logging Error at Epoch -- {epoch}")

    def _run_train_val(self):
        """
        Run Full Training and Validation.
        """
        # Tell wandb to watch the model and optimizer values
        # if self.accelerator.is_main_process:
        # self.tracker.run.watch(
        #     self.model,
        #     self.criterion,
        #     log="all",
        #     log_freq=10,
        #     log_graph=True,
        # )

        # Run Complete Training and Validation
        for epoch in range(self.current_epoch, self.num_epochs):

            # set to false at each epoch
            self.improved_val_loss = False
            # if self.calculate_metrics:
            #     self.improved_val_metric = False

            self.monitor["current_epoch"] = epoch  # update epoch
            self._train_step()  # run a single training step
            self._val_step()  # run a single validation step
            if self.accelerator.is_main_process:
                self._update_metrics()
                self._save_checkpoints()
                self._log_metrics()

        # log final metrics (ie: best metrics)
        # self._log_metrics(log_best=True)

    def train(self):
        self._run_train_val()
