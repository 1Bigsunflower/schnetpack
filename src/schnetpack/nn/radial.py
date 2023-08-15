from math import pi

import torch
import torch.nn as nn

__all__ = ["gaussian_rbf", "GaussianRBF", "GaussianRBFCentered", "BesselRBF"]

from torch import nn as nn


# 高斯径向基函数
def gaussian_rbf(inputs: torch.Tensor, offsets: torch.Tensor, widths: torch.Tensor):
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    # 表示输入数据相对于每个中心偏移量的高斯函数值
    return y


class GaussianRBF(nn.Module):  # 计算高斯径向基函数
    r"""Gaussian radial basis functions."""

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 0.0, trainable: bool = False
    ):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: center of last Gaussian function, :math:`\mu_{N_g}`  最后一个高斯函数的中心值
            start: center of first Gaussian function, :math:`\mu_0`.  第一个高斯函数的中心值
            trainable: If True, widths and offset of Gaussian functions
                are adjusted during training process.  是否在训练过程中调整高斯函数的宽度和偏移量
        """
        super(GaussianRBF, self).__init__()
        self.n_rbf = n_rbf  # 高斯函数的数量

        # compute offset and width of Gaussian functions
        offset = torch.linspace(start, cutoff, n_rbf)  # 计算高斯函数的中心偏移量，使用 torch.linspace(start, cutoff, n_rbf) 生成均匀分布的点，作为高斯函数的中心
        widths = torch.FloatTensor(  # 计算高斯函数的宽度
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)  # 使用 torch.abs(offset[1] - offset[0]) * torch.ones_like(offset) 计算相邻中心之间的距离
        )

        if trainable:
            # 将 widths 和 offset 注册为可训练的模型参数
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            # 将它们注册为模块缓冲（buffer）
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        # 计算高斯径向基函数的结果
        return gaussian_rbf(inputs, self.offsets, self.widths)


class GaussianRBFCentered(nn.Module):
    # 高斯径向基函数来处理输入数据
    r"""Gaussian radial basis functions centered at the origin."""

    def __init__(
        self, n_rbf: int, cutoff: float, start: float = 1.0, trainable: bool = False
    ):
        """
        Args:
            n_rbf: total number of Gaussian functions, :math:`N_g`.
            cutoff: width of last Gaussian function, :math:`\mu_{N_g}`
            start: width of first Gaussian function, :math:`\mu_0`.
            trainable: If True, widths of Gaussian functions
                are adjusted during training process.
        """
        super(GaussianRBFCentered, self).__init__()
        self.n_rbf = n_rbf  # 高斯函数的总数

        # compute offset and width of Gaussian functions高斯函数的宽度和偏移量
        widths = torch.linspace(start, cutoff, n_rbf)
        offset = torch.zeros_like(widths)
        # 如果 trainable 为 True，将 widths 和 offset 声明为模型的参数，可以在训练过程中进行调整。
        # 否则，注册为模型的缓冲区（buffer）。
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: torch.Tensor):
        return gaussian_rbf(inputs, self.offsets, self.widths)


class BesselRBF(nn.Module):  # 使用 Bessel 函数作为径向基函数，并具有库仑衰减。用于处理输入数据
    """
    Sine for radial basis functions with coulomb decay (0th order bessel).

    References:

    .. [#dimenet] Klicpera, Groß, Günnemann:
       Directional message passing for molecular graphs.
       ICLR 2020
    """

    def __init__(self, n_rbf: int, cutoff: float):
        """
        Args:
            cutoff: radial cutoff
            n_rbf: number of basis functions.
        """
        super(BesselRBF, self).__init__()
        self.n_rbf = n_rbf  # 基函数的数量

        freqs = torch.arange(1, n_rbf + 1) * pi / cutoff  # 频率
        self.register_buffer("freqs", freqs)  # 注册为缓冲区

    def forward(self, inputs):
        ax = inputs[..., None] * self.freqs
        sinax = torch.sin(ax)  # 计算ax正弦值
        # 归一化因子，如果inputs中某个值为0，将其设为1，否则保持不变
        norm = torch.where(inputs == 0, torch.tensor(1.0, device=inputs.device), inputs)
        y = sinax / norm[..., None]
        return y
