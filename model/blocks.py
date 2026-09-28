"""Spatial-temporal transformer blocks used by PCSTracker."""

import torch
import torch.nn as nn
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True):
        super().__init__()
        if dim % num_heads:
            raise ValueError("attention dimension must be divisible by num_heads")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch, tokens, 3, self.num_heads, channels // self.num_heads
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (query * self.scale) @ key.transpose(-2, -1)
        attention = self.attn_drop(attention.softmax(dim=-1))
        output = (attention @ value).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(output))


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(0.0)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop2 = nn.Dropout(0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.drop1(self.act(self.fc1(inputs)))
        return self.drop2(self.fc2(inputs))


class AttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(hidden_size, int(hidden_size * mlp_ratio))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs + self.attn(self.norm1(inputs))
        return inputs + self.mlp(self.norm2(inputs))


class UpdateFormer(nn.Module):
    def __init__(
        self,
        space_depth: int,
        time_depth: int,
        input_dim: int,
        hidden_size: int,
        num_heads: int,
        output_dim: int,
        mlp_ratio: float,
        add_space_attn: bool,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.add_space_attn = add_space_attn
        self.input_transform = nn.Linear(input_dim, hidden_size, bias=True)
        self.flow_head = nn.Linear(hidden_size, output_dim, bias=True)
        self.time_blocks = nn.ModuleList(
            [
                AttentionBlock(hidden_size, num_heads, mlp_ratio)
                for _ in range(time_depth)
            ]
        )
        if add_space_attn:
            self.space_blocks = nn.ModuleList(
                [
                    AttentionBlock(hidden_size, num_heads, mlp_ratio)
                    for _ in range(space_depth)
                ]
            )
            if len(self.time_blocks) < len(self.space_blocks):
                raise ValueError("time_depth must be at least space_depth")
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def initialize(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(initialize)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        features = self.input_transform(input_tensor)
        space_index = 0
        for time_index, time_block in enumerate(self.time_blocks):
            batch, points, frames, _ = features.shape
            temporal = rearrange(
                features, "b n t c -> (b n) t c", b=batch, n=points, t=frames
            )
            temporal = time_block(temporal)
            features = rearrange(
                temporal, "(b n) t c -> b n t c", b=batch, n=points, t=frames
            )
            if self.add_space_attn and time_index % (
                len(self.time_blocks) // len(self.space_blocks)
            ) == 0:
                spatial = rearrange(
                    features, "b n t c -> (b t) n c", b=batch, n=points, t=frames
                )
                spatial = self.space_blocks[space_index](spatial)
                features = rearrange(
                    spatial, "(b t) n c -> b n t c", b=batch, n=points, t=frames
                )
                space_index += 1
        return self.flow_head(features)
