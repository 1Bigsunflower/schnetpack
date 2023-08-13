from typing import Callable, Union, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_

from torch.nn.init import zeros_


__all__ = ["Dense"]


class Dense(nn.Linear):
    # 带有激活函数的全连接线性层
    r"""Fully connected linear layer with activation function.

    .. math::
       y = activation(x W^T + b)
    """

    def __init__(
        self,
        in_features: int,  # 输入特征的数量
        out_features: int,  # 输出特征的数量
        bias: bool = True,  # 是否使用偏置参数
        activation: Union[Callable, nn.Module] = None,  # 激活函数
        weight_init: Callable = xavier_uniform_,  # 权重的初始化函数
        bias_init: Callable = zeros_,  # 偏置的初始化函数
    ):
        """
        Args:
            in_features: number of input feature :math:`x`.
            out_features: umber of output features :math:`y`.
            bias: If False, the layer will not adapt bias :math:`b`.
            activation: if None, no activation function is used.
            weight_init: weight initializer from current weight.
            bias_init: bias initializer from current bias.
        """
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(Dense, self).__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()  # 默认使用的激活函数

    def reset_parameters(self):  # 重新初始化权重和偏置参数
        self.weight_init(self.weight)
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        y = F.linear(input, self.weight, self.bias)  # 计算输入张量与权重和偏置的线性变换结果
        y = self.activation(y)  # 传入激活函数
        return y
