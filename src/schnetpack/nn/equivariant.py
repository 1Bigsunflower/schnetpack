import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.nn as snn
from typing import Tuple

__all__ = ["GatedEquivariantBlock"]


class GatedEquivariantBlock(nn.Module):  # 用于在 PaiNN 中预测张量特性
    """
    Gated equivariant block as used for the prediction of tensorial properties by PaiNN.
    Transforms scalar and vector representation using gated nonlinearities.

    References:

    .. [#painn1] Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021 (to appear)

    """

    def __init__(
        self,
        n_sin: int,
        n_vin: int,
        n_sout: int,
        n_vout: int,
        n_hidden: int,
        activation=F.silu,
        sactivation=None,
    ):
        """
        Args:
            n_sin: number of input scalar features
            n_vin: number of input vector features
            n_sout: number of output scalar features
            n_vout: number of output vector features
            n_hidden: number of hidden units
            activation: interal activation function
            sactivation: activation function for scalar outputs
        """
        super().__init__()
        self.n_sin = n_sin  # 输入标量特征的数量
        self.n_vin = n_vin  # 输入向量特征的数量
        self.n_sout = n_sout  # 输出标量特征的数量
        self.n_vout = n_vout  # 输出向量特征的数量
        self.n_hidden = n_hidden  # 隐藏单元数量
        self.mix_vectors = snn.Dense(n_vin, 2 * n_vout, activation=None, bias=False)  # 创建线性层，将输入向量特征的数量减少到两倍输出向量特征的数量，且没有激活函数
        self.scalar_net = nn.Sequential(
            snn.Dense(n_sin + n_vout, n_hidden, activation=activation),  # 第一个线性层将输入标量特征和输出向量特征连起来,通过激活函数得到隐藏层特征
            snn.Dense(n_hidden, n_sout + n_vout, activation=None),  # 第二个线性层将隐藏层特征映射为输出标量特征和输出向量特征
        )
        self.sactivation = sactivation  # 标量输出的激活函数

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]):
        scalars, vectors = inputs
        vmix = self.mix_vectors(vectors)
        # 将 vmix 在最后一个维度上分割为两个张量 vectors_V 和 vectors_W，分别对应于输出向量特征中的 V 部分和 W 部分。
        vectors_V, vectors_W = torch.split(vmix, self.n_vout, dim=-1)
        # 计算 vectors_V 在倒数第二个维度上的范数
        vectors_Vn = torch.norm(vectors_V, dim=-2)

        ctx = torch.cat([scalars, vectors_Vn], dim=-1)
        x = self.scalar_net(ctx)
        s_out, x = torch.split(x, [self.n_sout, self.n_vout], dim=-1)
        v_out = x.unsqueeze(-2) * vectors_W

        if self.sactivation:
            # 对 s_out 应用该激活函数
            s_out = self.sactivation(s_out)

        return s_out, v_out
