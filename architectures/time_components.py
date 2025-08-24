import torch
import torch.nn as nn
from typing import Tuple
from architectures.mmdit_blocks import MLP


##########################################################################################
##########################################################################################
class TimePositionalEncoding(nn.Module):
    def __init__(self, dim, device: str = "cpu"):
        super().__init__()
        self.dim = dim

        # Calculate the denominator of the position encodings
        # as this value is constant
        self.denom = (
            torch.tensor(10000.0) ** ((2 * torch.arange(self.dim)) / self.dim)
        ).to(dtype=torch.float)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to Time steps
        Args:
            t (torch.Tensor): scaled time steps. shape [B]
        Returns:
            torch.Tensor: (B, dim)
        """
        # Compute the current timestep embeddings
        embeddings = t[:, None] / self.denom[None, :].to(t.device)

        # Sin/Cos transformation for even, odd indices
        embeddings = torch.cat(
            (embeddings[:, ::2].sin(), embeddings[:, 1::2].cos()),
            dim=1,
        )
        return embeddings


##########################################################################################
##########################################################################################
class TimeSampler:
    def __init__(self, weighted=True, m=0.0, s=1.0):
        self.weighted = weighted
        self.m = m
        self.s = s

    def __call__(self, batch_size: int) -> torch.Tensor:
        return self.sample(batch_size)

    def sample(self, n):
        if self.weighted:
            # Sample n time points from a normal distribution
            u = torch.randn(n) * self.s + self.m

            # Map the samples to the range [0, 1] using the logistic function
            return torch.sigmoid(u)
        else:
            return torch.rand(n)


##########################################################################################
##########################################################################################
# TODO: update parameters like nn.SiLU to init function
class EncodeTime(nn.Module):
    def __init__(self, dim, hidden_scale=1.0):
        # dim = dim of the pooled + stacked vector.
        super().__init__()
        # scale time steps from (0,1) to (0, 1000)
        self.scale = nn.Parameter(
            torch.tensor([1000.0], dtype=torch.float), requires_grad=True
        )
        self.pe = TimePositionalEncoding(dim)
        self.mlp = MLP(dim, nn.SiLU, 0.0, hidden_scale=hidden_scale, bias=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        scales, adds positional encodings and embeds time information
        Args:
            t (torch.Tensor): time steps between (0, 1). shape [B]
        Returns:
            torch.Tensor: _description_
        """

        # scale time
        t = t.to(self.scale.dtype) * self.scale
        # add positional encodings
        t = self.pe(t)
        # mlp
        t = self.mlp(t)
        return t


##########################################################################################
##########################################################################################
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
    epsilon = torch.randn_like(x)

    # modify noise based on time step
    x_t = (1 - t) * x + (t * epsilon)

    return x_t, epsilon


##########################################################################################
##########################################################################################
