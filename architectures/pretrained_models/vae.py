import torch
import torch.nn as nn
from diffusers import AutoencoderKL


################################################################################################
################################################################################################
class VAE(nn.Module):
    def __init__(self, compile: bool = False, device: str = None):
        super().__init__()
        self.model_name = "black-forest-labs/FLUX.1-schnell"
        self.vae = (
            AutoencoderKL.from_pretrained(
                self.model_name,
                subfolder="vae",
                cache_dir="../pretrained_models/models",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            ).eval()
            # .to(dtype=torch.bfloat16)
        )
        # del self.vae.decoder
        # not sure if set False makes a big difference in run time.
        for param in self.vae.parameters():
            param.requires_grad = False
        self.scaling_factor = self.vae.config.scaling_factor
        self.shift_factor = self.vae.config.shift_factor
        if compile:
            self.vae = torch.compile(self.vae)

        if device:
            self.device = device
            self.device = self.vae.to(self.device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded_input = (
            self.vae.encode(x.to(self.vae.dtype)).latent_dist.sample()
            * self.scaling_factor
            + self.shift_factor
        )
        return encoded_input.to(torch.float32)

    @torch.no_grad()
    @torch.inference_mode()
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        decoded_image = self.vae.decode(x).sample
        return decoded_image
