from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class FanInInitReLULayer(nn.Module):
    def __init__(
        self,
        inchan: int,
        outchan: int,
        layer_type: str = "conv",
        init_scale: float = 1.0,
        batch_norm: bool = False,
        group_norm_groups: int | None = None,
        layer_norm: bool = False,
        use_activation: bool = True,
        **layer_kwargs: object,
    ) -> None:
        super().__init__()
        self.norm: nn.Module | None = None
        if batch_norm:
            self.norm = nn.BatchNorm2d(inchan)
        elif group_norm_groups is not None:
            self.norm = nn.GroupNorm(group_norm_groups, inchan)
        elif layer_norm:
            self.norm = nn.LayerNorm(inchan)
        layer_cls = {"conv": nn.Conv2d, "linear": nn.Linear}[layer_type]
        self.layer = layer_cls(inchan, outchan, bias=self.norm is None, **layer_kwargs)
        self.use_activation = use_activation
        dim = tuple(range(1, self.layer.weight.data.ndim))
        self.layer.weight.data *= init_scale / self.layer.weight.norm(dim=dim, p=2, keepdim=True)
        if self.layer.bias is not None:
            self.layer.bias.data *= 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            x = self.norm(x)
        x = self.layer(x)
        if self.use_activation:
            x = F.relu(x, inplace=True)
        return x


class CnnBasicBlock(nn.Module):
    def __init__(
        self,
        inchan: int,
        init_scale: float = 1.0,
        group_norm_groups: int = 1,
    ) -> None:
        super().__init__()
        scale = math.sqrt(init_scale)
        kwargs = {"group_norm_groups": group_norm_groups}
        self.conv0 = FanInInitReLULayer(
            inchan, inchan, kernel_size=3, padding=1, init_scale=scale, **kwargs
        )
        self.conv1 = FanInInitReLULayer(
            inchan, inchan, kernel_size=3, padding=1, init_scale=scale, **kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv1(self.conv0(x))


class CnnDownStack(nn.Module):
    def __init__(
        self,
        inchan: int,
        nblock: int,
        outchan: int,
        init_scale: float = 1.0,
        pool: bool = True,
        post_pool_groups: int | None = 1,
        group_norm_groups: int = 1,
        first_conv_norm: bool = False,
    ) -> None:
        super().__init__()
        self.pool = pool
        first_kwargs: dict[str, object] = {"group_norm_groups": None, "batch_norm": False}
        if first_conv_norm:
            first_kwargs = {"group_norm_groups": group_norm_groups}
        self.firstconv = FanInInitReLULayer(
            inchan, outchan, kernel_size=3, padding=1, **first_kwargs
        )
        self.post_pool: nn.Module | None = None
        if post_pool_groups is not None:
            self.post_pool = nn.GroupNorm(post_pool_groups, outchan)
        block_scale = init_scale / math.sqrt(nblock)
        self.blocks = nn.ModuleList(
            [
                CnnBasicBlock(outchan, init_scale=block_scale, group_norm_groups=group_norm_groups)
                for _ in range(nblock)
            ]
        )
        self._outchan = outchan

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.firstconv(x)
        if self.pool:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        if self.post_pool is not None:
            x = self.post_pool(x)
        for block in self.blocks:
            x = block(x)
        return x

    def output_shape(self, shape: Sequence[int]) -> tuple[int, int, int]:
        channels, height, width = shape
        if self.pool:
            return (self._outchan, (height + 1) // 2, (width + 1) // 2)
        return (self._outchan, height, width)


class ImpalaEncoder(nn.Module):
    def __init__(
        self,
        inshape: Sequence[int],
        channels: Sequence[int],
        outsize: int,
        nblock: int,
    ) -> None:
        super().__init__()
        curshape = tuple(inshape)
        self.stacks = nn.ModuleList()
        stack_count = len(channels)
        for index, outchan in enumerate(channels):
            stack = CnnDownStack(
                curshape[0],
                nblock=nblock,
                outchan=outchan,
                init_scale=1.0 / math.sqrt(stack_count),
                first_conv_norm=index == 0,
            )
            self.stacks.append(stack)
            curshape = stack.output_shape(curshape)
        self.dense = FanInInitReLULayer(
            math.prod(curshape),
            outsize,
            layer_type="linear",
            init_scale=1.4,
            layer_norm=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stack in self.stacks:
            x = stack(x)
        return self.dense(x.reshape(x.size(0), -1))
