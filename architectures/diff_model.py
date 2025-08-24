import torch
import torch.nn as nn
from architectures.time_components import EncodeTime
from monai.networks.blocks import PatchEmbeddingBlock
from architectures.mmdit_blocks import MMDiTBlock, FinalLayer, MLP, FinalLayerText


################################################################################################
# Class: MultiModal Diffusion Transformer Model [From Scratch][Extendable]
################################################################################################
class MMDiT2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        # encode time
        self.encode_time = EncodeTime(config.condition_dim)

        # embed c_pooled ie: pooled captions
        self.embed_c_pooled = MLP(
            config.condition_dim,
            act_fn=get_act_fn(config.act_fn),
            p_drop=config.p_drop,
            bias=config.bias,
        )

        # embed c ie: combined captions tensor
        self.embed_c = nn.Linear(config.text_dim, config.text_dim, bias=True)

        # patch embedding + positional encoding
        # for images
        self.patch_encode = PatchEmbeddingBlock(
            in_channels=config.vae_output_dim,
            img_size=config.vae_output_size,
            patch_size=config.patch_size,
            hidden_size=config.text_dim,
            num_heads=config.patch_num_heads,
            proj_type="conv",
            pos_embed_type=config.patch_pos_embed_type,
            dropout_rate=config.patch_dropout_rate,
            spatial_dims=config.patch_spatial_dims,
        )

        # mmdit blocks
        self.mmdit_blocks = nn.ModuleList([])
        for _ in range(config.num_blocks):
            self.mmdit_blocks.append(
                MMDiTBlock(
                    x_dim=config.image_dim,
                    c_dim=config.text_dim,
                    y_dim=config.condition_dim,
                    act_fn=get_act_fn(config.act_fn),
                    p_drop=config.p_drop,
                    num_heads=config.num_heads,
                    hidden_scale=config.hidden_scale,
                    bias=config.bias,
                    qkv_bias=config.qkv_bias,
                    qk_norm=config.qk_norm,
                    norm_layer=get_norm_fn(config.norm_layer),
                )
            )

        # final layer (image)
        self.final_layer = FinalLayer(
            x_dim=config.image_dim,
            y_dim=config.condition_dim,
            patch_size=config.patch_size,
            out_channels=config.vae_output_dim,
            image_size=config.vae_output_size,
        )

        # final layer (text)
        self.final_layer_text = FinalLayerText(
            c_dim=config.text_dim,
            y_dim=config.condition_dim,
        )

        self.apply(self.initialize_weights)

    def forward(self, x, c, c_pooled, t):
        # x - [B, C, H, W] noisy latent
        # c - [B, Seq=154, Dim_c] combined captions
        # c_pooled - [B, Dim_cpooled] pooled caption
        # t - [B] time steps from sampler b/w (0, 1)
        # print(x.dtype, c.dtype, c_pooled.dtype, t.dtype)
        # exit()

        # process time steps
        t = self.encode_time(t)
        # print(f"encoded time: {t.shape}")

        # get y vector from paper
        y = self.embed_c_pooled(c_pooled) + t
        # print(f"y: {y.shape}")

        # embed c
        c = self.embed_c(c)
        # print(f"c: {c.shape}")

        # patch + pos encode x
        x = self.patch_encode(x)
        # print(f"x: {x.shape}")

        # mmdit blocks
        # TODO: we can save these intermediate steps
        for idx, block in enumerate(self.mmdit_blocks):
            c, x = block(c, x, y)
            # print(f"block {idx}: {c.shape, x.shape}")

        # final layer
        # TODO: add another final layer for text?
        # with torch.compile.disable():
        x = self.final_layer(x, y)  # [B, C, H, W]
        c = self.final_layer_text(c, y)  # [B, 154, Dim]
        # print(f"final out: {x.shape}")

        return x, c

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


################################################################################################
################################################################################################
def get_act_fn(fn_name):
    if fn_name == "silu":
        return nn.SiLU
    else:
        raise ValueError(
            "Activation Function Not Supported. \
            Add Functionality to architectures.diff_model.get_act_fn function"
        )


def get_norm_fn(fn_name):
    if fn_name == "rmsnorm":
        return nn.RMSNorm
    else:
        raise ValueError(
            "Normalization Function Not Supported. \
            Add Functionality to architectures.diff_model.get_norm_fn function"
        )
