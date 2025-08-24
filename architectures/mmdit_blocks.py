import torch
import torch.nn as nn
from torch.nn import functional as F
from torchtune.modules import RotaryPositionalEmbeddings
from typing import Final, Optional, Type, Callable, Tuple


######################################################################################
######################################################################################
class MLP(nn.Module):
    """
    Implements a simple/straightforward MLP
    """

    def __init__(
        self,
        dim: int,
        act_fn: Callable,
        p_drop: float = 0.0,
        hidden_scale: float = 4.0,
        bias: bool = True,
    ) -> None:
        """
        Implements a simple/straightforward MLP

        Args:
            dim (int): dim of the input tensor
            act_fn (Callable): any torch activation function
            p_drop (float, optional): dropout probability. Defaults to 0.0
            bias (bool, optional): Linear layer bias. Defaults to True.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * int(hidden_scale), bias=bias),
            # act_fn(),
            nn.SiLU(),
            nn.LayerNorm(dim * int(hidden_scale)),
            nn.Dropout(p_drop),
            nn.Linear(dim * int(hidden_scale), dim, bias=bias),
            # act_fn(dim),
            # nn.Dropout(p_drop),
        )
        self.apply(self.initialize_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


######################################################################################
######################################################################################
class ModulationHelper(nn.Module):
    """
    We use Adaptive Layer Norm Modulation from DiT Paper/Repo (META)
    Helper Function takes in the conditioning tensors y (from the paper)
    and returns back x number of tensors to perform the scale and shift
    operations required for modulation.

    Example:
    in the MMDiT block in Figure 2 paper, the y tensor is used to generate
    6 scale, shift and gate operations. so num return tensors is 6 to obtain
    these 6 tensors in a single forward pass.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        num_return_tensors: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        """
        Args:
            dim (int): feature dimension of the input data 
            c_dim (int): feature dimension of the conditioning data
            bias (bool): wether to use bias in linear projection or not
        """
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, num_return_tensors * dim, bias=bias),
        )
        self.num_return_tensors = num_return_tensors
        self.apply(self.initialize_weights)

    def forward(self, condition: torch.Tensor) -> Tuple:
        """
        Get the modulation tensors

        Args:
            condition (torch.Tensor): [B, S, F_condition]

        Returns:
            torch.Tensor: [B, S, F_data]
        """
        # expand condition to have the same shape [B, F] -> [B, 1, F]
        condition = condition[:, None, :]

        # get modulation tensors
        modulation_tensors = self.adaLN_modulation(condition).chunk(
            self.num_return_tensors, dim=-1
        )

        return modulation_tensors

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


######################################################################################
######################################################################################
class PreAttentionModulationBlock(nn.Module):
    """
    In Figure 2 of the paper this block covers the ..
    ..Input (x or c) -> LayerNorm -> Modulation -> Linear components.
    This version of the implementation allows us to scale mmdit into x number..
    ..of modalities.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim (int): the dimension of the input tensor (x or c) in paper.
        """
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, dim, bias=True)
        self.apply(self.initialize_weights)

    def forward(
        self,
        data: torch.Tensor,
        scale_mlp: torch.Tensor,
        shift_mlp: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward PreAttention Blocks.
        Input (x or c) -> LayerNorm -> Modulation -> Linear components.

        Args:
            data (torch.Tensor): input tensor. [B, C, F]
            scale_mlp (torch.Tensor): [B, 1, F]
            shift_mlp (torch.Tensor): [B, 1, F]

        Returns:
            torch.Tensor: [B, C, F]
        """
        data = self.layer_norm(data)
        data = data * (1 + scale_mlp) + shift_mlp
        data = self.linear(data)
        return data

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


######################################################################################
######################################################################################
def maybe_add_mask(scores: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
    return scores if attn_mask is None else scores + attn_mask


class Attention(nn.Module):
    """Standard Multi-head Self Attention module with QKV projection.

    This module implements the standard multi-head attention mechanism used in transformers.
    It supports both the fused attention implementation (scaled_dot_product_attention) for
    efficiency when available, and a manual implementation otherwise. The module includes
    options for QK normalization, attention dropout, and projection dropout.
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        scale_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: Optional[Type[nn.Module]] = None,
        use_fused_attn: bool = True,
    ) -> None:
        """Initialize the Attention module.

        Args:
            dim: Input dimension of the token embeddings
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in the query, key, value projections
            qk_norm: Whether to apply normalization to query and key vectors
            proj_bias: Whether to use bias in the output projection
            attn_drop: Dropout rate applied to the attention weights
            proj_drop: Dropout rate applied after the output projection
            norm_layer: Normalization layer constructor for QK normalization if enabled
        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        if qk_norm or scale_norm:
            assert (
                norm_layer is not None
            ), "norm_layer must be provided if qk_norm or scale_norm is True"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=410)
        self.apply(self.initialize_weights)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        # q and k are [B, num_heads, seq len, head_dim]
        # print("pre: ", q.shape, k.shape)
        q = self.rope(q.moveaxis(1, 2)).moveaxis(1, 2)
        k = self.rope(k.moveaxis(1, 2)).moveaxis(1, 2)
        # print("post: ", q.shape, k.shape)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        # else:
        #     q = q * self.scale
        #     attn = q @ k.transpose(-2, -1)
        #     attn = maybe_add_mask(attn, attn_mask)
        #     attn = attn.softmax(dim=-1)
        #     attn = self.attn_drop(attn)
        #     x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


######################################################################################
######################################################################################
class PostAttentionModulationBlock(nn.Module):
    """
    Post Attention Section in Figure 2(b) in the paper.
    Linear -> Gate -> Residual -> LayerNorm -> Modulation -> MLP -> Gate -> Residual
    """

    def __init__(
        self,
        dim: int,
        act_fn: Callable,
        p_drop: float,
        hidden_scale: float,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=bias)
        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP(dim, act_fn, p_drop, hidden_scale=hidden_scale, bias=bias)
        self.apply(self.initialize_weights)

    def forward(
        self,
        data: torch.Tensor,
        residual_data: torch.Tensor,
        gate_msa: torch.Tensor,
        scale_mlp: torch.Tensor,
        shift_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
    ) -> torch.Tensor:
        data = self.linear(data)
        data = gate_msa * data
        data = residual_data + data
        data = self.layer_norm(data)
        data = data * (1 + scale_mlp) + shift_mlp
        data = self.mlp(data)
        data = gate_mlp * data
        data = residual_data + data
        return data

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


######################################################################################
######################################################################################
class MMDiTBlock(nn.Module):
    def __init__(
        self,
        c_dim: int,
        x_dim: int,
        y_dim: int,
        act_fn: Callable,
        p_drop: float,
        num_heads: int,
        hidden_scale: float,
        bias: bool = True,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        norm_layer: Type[nn.Module] = nn.RMSNorm,
    ) -> None:
        super().__init__()
        assert (
            x_dim == c_dim
        ), "x_dim must equal c_dim Model is Built With This Assumption."

        # preprocess condition vectors for each modality
        self.y_proj_m1 = nn.Sequential(nn.SiLU(), nn.Linear(y_dim, y_dim))
        self.y_proj_m2 = nn.Sequential(nn.SiLU(), nn.Linear(y_dim, y_dim))

        # get modulation vectors for each modality
        self.modulate_m1 = ModulationHelper(c_dim, y_dim, 6)
        self.modulate_m2 = ModulationHelper(x_dim, y_dim, 6)

        # pre-attention blocks for each modality
        self.preattention_m1 = PreAttentionModulationBlock(c_dim)
        self.preattention_m2 = PreAttentionModulationBlock(x_dim)

        # attention
        # if x_dim != c_dim .. dim below needs to be adjusted.
        self.attention = Attention(
            dim=x_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            qk_norm=qk_norm,
        )

        # post-attention blocks for each modality
        self.postattention_m1 = PostAttentionModulationBlock(
            c_dim, act_fn, p_drop, hidden_scale, bias
        )
        self.postattention_m2 = PostAttentionModulationBlock(
            x_dim, act_fn, p_drop, hidden_scale, bias
        )
        self.apply(self.initialize_weights)

    def forward(self, c: torch.Tensor, x: torch.Tensor, y: torch.tensor) -> Tuple:
        # x = image features [B, S_x, F_x]
        # c = caption features [B, S_c, F_c]
        # y = condition based on captions [B, S_y, F_y]

        # get modulation vectors for each modality
        # order of the modulation vectors from the order in the paper
        # [scale_msa, shift_msa, gate_msa, scale_mlp, shift_mlp, gate_mlp]
        modulation_c = self.modulate_m1(self.y_proj_m1(y))
        modulation_x = self.modulate_m2(self.y_proj_m2(y))

        # perform pre attention process for each modality
        c_pre = self.preattention_m1(
            c, scale_mlp=modulation_c[0], shift_mlp=modulation_c[1]
        )
        x_pre = self.preattention_m2(
            x, scale_mlp=modulation_x[0], shift_mlp=modulation_x[1]
        )

        # Attention
        ### Before attention we concat in seq dim or embedding dim
        ### After attention do we split back if we concat on seq dim?
        ### After attention do we split back if we concat on embedding dim?

        # here we will do seq wise concat
        attention_input = torch.concatenate((c_pre, x_pre), dim=1)
        attention_out = self.attention(attention_input)

        # split back to seperate tensors
        c_post_attention = attention_out[:, : c_pre.shape[1], :]
        x_post_attention = attention_out[:, c_pre.shape[1] :, :]

        # perform post attention process for each modality
        c_post = self.postattention_m1(c_post_attention, c, *modulation_c[2:])
        x_post = self.postattention_m2(x_post_attention, x, *modulation_x[2:])

        return c_post, x_post

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


######################################################################################
######################################################################################
class FinalLayer(nn.Module):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        patch_size: int,
        out_channels: int,
        image_size: int,
    ) -> None:
        """
        Args:
            x_dim (int): input dim. in paper this is the x features.
            y_dim (int): condition dim
            patch_size (int): patch size we used after VAE
            out_channels (int): number of channels out from VAE
            image_size (int): image size after VAE
        """
        super().__init__()
        self.modulate = ModulationHelper(x_dim, y_dim, 2)
        self.linear = nn.Linear(
            x_dim, patch_size * patch_size * out_channels, bias=True
        )
        self.patch_size = (patch_size, patch_size)
        self.out_channels = out_channels
        self.image_size = (image_size, image_size)
        self.apply(self.initialize_weights)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # modulate: scale and shift x * scale + shift
        modulate = self.modulate(y)
        x = x * (1 + modulate[0]) + modulate[1]
        x = self.linear(x)
        x = self.unpatchify(x, self.patch_size, self.image_size)
        return x

    def unpatchify(
        self,
        patches: torch.Tensor,
        patch_size: Tuple[int, int],
        original_shape: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Reconstruct the original images from flattened patches.

        Args:
            patches: Tensor of shape (N, num_patches, patch_size * patch_size * C), containing flattened patches.
            patch_size: Size of each patch (patch_height, patch_width).
            original_shape: Tuple of the original unpadded image shape (H_original, W_original).

        Returns:
            images: Tensor of shape (N, C, H_original, W_original), the reconstructed images.
        """
        N, num_patches, patch_dim = patches.shape
        patch_height, patch_width = patch_size
        H_original, W_original = original_shape

        # Compute the number of patches along height and width
        num_patches_h = (H_original + patch_height - 1) // patch_height
        num_patches_w = (W_original + patch_width - 1) // patch_width

        # Reshape patches back to (N, num_patches_h, num_patches_w, C, patch_height, patch_width)
        C = patch_dim // (patch_height * patch_width)
        patches = patches.view(
            N, num_patches_h, num_patches_w, C, patch_height, patch_width
        )

        # Permute and reshape to (N, C, H_padded, W_padded)
        images_padded = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        images_padded = images_padded.view(
            N, C, num_patches_h * patch_height, num_patches_w * patch_width
        )

        # Crop the images back to the original unpadded size
        images = images_padded[:, :, :H_original, :W_original]

        return images

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


######################################################################################
######################################################################################
class FinalLayerText(nn.Module):
    def __init__(
        self,
        c_dim: int,
        y_dim: int,
    ) -> None:
        """
        Args:
            c_dim (int): input dim. in paper this is the c features.
            y_dim (int): condition dim
            patch_size (int): patch size we used after VAE
            out_channels (int): number of channels out from VAE
            image_size (int): image size after VAE
        """
        super().__init__()
        self.modulate = ModulationHelper(c_dim, y_dim, 2)
        self.linear = nn.Linear(c_dim, c_dim, bias=True)
        self.apply(self.initialize_weights)

    def forward(self, c: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # modulate: scale and shift x * scale + shift
        modulate = self.modulate(y)
        c = c * (1 + modulate[0]) + modulate[1]
        c = self.linear(c)
        return c

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


######################################################################################
######################################################################################
