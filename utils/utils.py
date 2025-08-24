import os
import torch
import random
import numpy as np
from typing import Dict, Tuple


################################################################################################
################################################################################################
def combine_tensors(encoded_text: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Combines Caption Tensors Similar to Figure 2a in paper.

    Args:
        vision_features_small (torch.Tensor): vision features from small vlm: [B, S, F1]
        vision_features_large (torch.Tensor): vision features from large vlm: [B, S, F2]
        llm_features (torch.Tensor): text features from llm: [B, S, F3]

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: c = [B, S + S, F3], c_pooled=[B, F1+F2]
    """
    clip_s_hidden = encoded_text["clip_s_hidden"]
    clip_l_hidden = encoded_text["clip_l_hidden"]
    llm_hidden = encoded_text["llm_hidden"]
    clip_s_pooled = encoded_text["clip_s_pooled"]
    clip_l_pooled = encoded_text["clip_l_pooled"]

    # combine vision features from vlm ex: clip
    combined_vision = torch.concatenate([clip_s_hidden, clip_l_hidden], dim=2)

    # figure out padding
    # if size_diff < 0 pad llm, if size_diff > 0 pad combined vision
    # if size_diff == 0, no padding neededd
    size_diff = llm_hidden.shape[-1] - combined_vision.shape[-1]
    if size_diff < 0:
        B, S, _ = llm_hidden.shape
        padding_tensor = torch.zeros(
            (B, S, abs(size_diff)),
            dtype=clip_s_hidden.dtype,
            device=clip_s_hidden.device,
        )
        llm_hidden = torch.concatenate([llm_hidden, padding_tensor], dim=-1)
    elif size_diff > 0:
        B, S, _ = combined_vision.shape
        padding_tensor = torch.zeros(
            (B, S, abs(size_diff)),
            dtype=clip_s_hidden.dtype,
            device=clip_s_hidden.device,
        )
        combined_vision = torch.concatenate([combined_vision, padding_tensor], dim=-1)
    else:
        # print(f"no padding needed")
        pass

    # crete final rectangular matrix
    c = torch.concatenate([combined_vision, llm_hidden], dim=1)

    # create stacked pooled matrix
    c_pooled = torch.concatenate((clip_s_pooled, clip_l_pooled), dim=-1)

    return c, c_pooled


################################################################################################
################################################################################################
def add_noise(x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


#############################################################################################
# Function: Count Parameters
#############################################################################################
def count_parameters(model, print_result: bool = False):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    if print_result:
        print(f"Total parameters:     {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        print(f"Frozen parameters:    {frozen:,}")
    return {"total": total, "trainable": trainable, "frozen": frozen}


#############################################################################################
# Function: Seed Everything
#############################################################################################
def seed_everything(config) -> None:
    seed = config["training_parameters"]["seed"]
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
