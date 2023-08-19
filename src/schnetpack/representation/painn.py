from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import schnetpack.properties as properties
import schnetpack.nn as snn

__all__ = ["PaiNN", "PaiNNInteraction", "PaiNNMixing"]


class PaiNNInteraction(nn.Module):  # 用于建模原子系统的等变相互作用
    r"""PaiNN interaction block for modeling equivariant interactions of atomistic systems."""

    def __init__(self, n_atom_basis: int, activation: Callable):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            activation: if None, no activation function is used.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(PaiNNInteraction, self).__init__()
        self.n_atom_basis = n_atom_basis  # 描述原子环境的特征数量

        self.interatomic_context_net = nn.Sequential(  # 声明了一个包含两个全连接层的神经网络
            snn.Dense(n_atom_basis, n_atom_basis, activation=activation),
            snn.Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )

    def forward(
        self,
        q: torch.Tensor,  # 标量输入值
        mu: torch.Tensor,  # 矢量输入值
        Wij: torch.Tensor,  # 过滤器
        dir_ij: torch.Tensor,  # 原子对的方向向量
        idx_i: torch.Tensor,  # 中心原子 i 的索引
        idx_j: torch.Tensor,  # 邻居原子 j 的索引
        n_atoms: int,  # 原子数量
    ):
        """Compute interaction output.

        Args:
            q: scalar input values
            mu: vector input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        Returns:
            atom features after interaction
        """
        # inter-atomic
        x = self.interatomic_context_net(q)  # 对 q 进行一次全连接操作得到输出
        # 根据邻居原子的索引 idx_j 提取相应的特征 xj 和 muj
        xj = x[idx_j]
        muj = mu[idx_j]
        x = Wij * xj

        # dq 表示标量输入特征的改变量。
        # dmuR 表示矢量输入特征在原子对方向上的改变量。
        # dmumu 表示矢量输入特征在邻居原子特征方向上的改变量。
        dq, dmuR, dmumu = torch.split(x, self.n_atom_basis, dim=-1)

        # 将 dq 和 dmu 加到中心原子 i 的索引位置得到新的 q 和 mu
        dq = snn.scatter_add(dq, idx_i, dim_size=n_atoms)
        dmu = dmuR * dir_ij[..., None] + dmumu * muj
        dmu = snn.scatter_add(dmu, idx_i, dim_size=n_atoms)

        q = q + dq
        mu = mu + dmu

        return q, mu


class PaiNNMixing(nn.Module):  # 用于混合原子特征的 PaiNN 交互块
    r"""PaiNN interaction block for mixing on atom features."""

    def __init__(self, n_atom_basis: int, activation: Callable, epsilon: float = 1e-8):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            activation: if None, no activation function is used.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(PaiNNMixing, self).__init__()
        self.n_atom_basis = n_atom_basis  # 描述原子环境的特征数量

        self.intraatomic_context_net = nn.Sequential(
            # 包含两个全连接层的神经网络。
            # 第一个全连接层将输入特征从 2 * n_atom_basis 维度映射到 n_atom_basis 维度，并应用激活函数。
            # 第二个全连接层将输入特征从 n_atom_basis 维度映射到 3 * n_atom_basis 维度，不应用激活函数
            snn.Dense(2 * n_atom_basis, n_atom_basis, activation=activation),
            snn.Dense(n_atom_basis, 3 * n_atom_basis, activation=None),
        )
        self.mu_channel_mix = snn.Dense(
            n_atom_basis, 2 * n_atom_basis, activation=None, bias=False
        )
        self.epsilon = epsilon  # 加在范数中的稳定性常数，用于防止数值不稳定

    def forward(self, q: torch.Tensor, mu: torch.Tensor):
        """Compute intraatomic mixing.

        Args:
            q: scalar input values
            mu: vector input values

        Returns:
            atom features after interaction
        """
        # intra-atomic
        mu_mix = self.mu_channel_mix(mu)  # 通过 mu_channel_mix 对输入的 mu 进行一次全连接操作
        mu_V, mu_W = torch.split(mu_mix, self.n_atom_basis, dim=-1)
        # 计算 mu_V 的范数并添加稳定性常数 epsilon
        mu_Vn = torch.sqrt(torch.sum(mu_V**2, dim=-2, keepdim=True) + self.epsilon)

        # 将 q 和 mu_Vn 在最后一个维度上进行拼接
        ctx = torch.cat([q, mu_Vn], dim=-1)
        x = self.intraatomic_context_net(ctx)  # 进行一次全连接操作，得到输出 x

        dq_intra, dmu_intra, dqmu_intra = torch.split(x, self.n_atom_basis, dim=-1)
        dmu_intra = dmu_intra * mu_W

        dqmu_intra = dqmu_intra * torch.sum(mu_V * mu_W, dim=1, keepdim=True)

        q = q + dq_intra + dqmu_intra
        mu = mu + dmu_intra
        return q, mu


class PaiNN(nn.Module):  # 模拟分子之间的相互作用
    """PaiNN - polarizable interaction neural network

    References:

    .. [#painn1] Schütt, Unke, Gastegger:
       Equivariant message passing for the prediction of tensorial properties and molecular spectra.
       ICML 2021, http://proceedings.mlr.press/v139/schutt21a.html

    """

    def __init__(
        self,
        n_atom_basis: int,  # 描述原子环境的特征数量，确定了每个嵌入向量的大小
        n_interactions: int,  # 交互块的数量
        radial_basis: nn.Module,  # 将原子间距离按照基函数进行扩展的层
        cutoff_fn: Optional[Callable] = None,  # 截断函数
        activation: Optional[Callable] = F.silu,  # 激活函数
        max_z: int = 100,  # 原子序数的最大值
        shared_interactions: bool = False,  # 是否共享交互块的权重
        shared_filters: bool = False,  # 是否共享生成滤波器的网络的权重
        epsilon: float = 1e-8,  # 在范数中添加的稳定性常量
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            activation: activation function
            shared_interactions: if True, share the weights across
                interaction blocks.
            shared_interactions: if True, share the weights across
                filter-generating networks.
            epsilon: stability constant added in norm to prevent numerical instabilities
        """
        super(PaiNN, self).__init__()

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.radial_basis = radial_basis

        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)  # 用于将原子序数转化为对应的特征向量

        self.share_filters = shared_filters

        # 创建了一个全连接网络(snn.Dense)，用于生成滤波器。如果shared_filters为真，则所有交互块共享相同的权重
        if shared_filters:
            self.filter_net = snn.Dense(
                self.radial_basis.n_rbf, 3 * n_atom_basis, activation=None
            )
        else:
            self.filter_net = snn.Dense(
                self.radial_basis.n_rbf,
                self.n_interactions * n_atom_basis * 3,
                activation=None,
            )

        # 使用snn.replicate_module方法复制PaiNNInteraction和PaiNNMixing模块，得到多个交互块。如果shared_interactions参数为真，则所有交互块共享相同的权重。
        self.interactions = snn.replicate_module(
            lambda: PaiNNInteraction(
                n_atom_basis=self.n_atom_basis, activation=activation
            ),
            self.n_interactions,
            shared_interactions,
        )
        self.mixing = snn.replicate_module(
            lambda: PaiNNMixing(
                n_atom_basis=self.n_atom_basis, activation=activation, epsilon=epsilon
            ),
            self.n_interactions,
            shared_interactions,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        # get tensors from input dictionary
        # inputs，其中包含了SchNetPack输入张量的键-值对
        atomic_numbers = inputs[properties.Z]  # 原子序数
        r_ij = inputs[properties.Rij]  # 原子间距离
        idx_i = inputs[properties.idx_i]  # 原子索引
        idx_j = inputs[properties.idx_j]  # 原子索引
        n_atoms = atomic_numbers.shape[0]

        # compute atom and pair features
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)  # 原子间距离的范数
        dir_ij = r_ij / d_ij  # 指向向量
        phi_ij = self.radial_basis(d_ij)  # 径向基函数
        fcut = self.cutoff_fn(d_ij)

        filters = self.filter_net(phi_ij) * fcut[..., None]  # 滤波器
        # 如果shared_filters为真，则将滤波器复制为列表中的元素，长度为交互块的数量；
        # 否则，使用torch.split方法在最后一个维度上将滤波器分割成长度为3 * self.n_atom_basis的小块。
        if self.share_filters:
            filter_list = [filters] * self.n_interactions
        else:
            filter_list = torch.split(filters, 3 * self.n_atom_basis, dim=-1)

        # 利用嵌入层(embedding)将原子序数转化为原子特征向量(q)，并扩展为适应后续计算的形状(mu)。
        q = self.embedding(atomic_numbers)[:, None]
        qs = q.shape
        mu = torch.zeros((qs[0], 3, qs[2]), device=q.device)

        # 遍历交互块(interactions)和混合模块(mixing)，依次进行交互和混合操作。
        # 在每个交互块中，调用交互模块和混合模块的前向传播方法，并更新原子特征(q)和中间特征(mu)。
        for i, (interaction, mixing) in enumerate(zip(self.interactions, self.mixing)):
            q, mu = interaction(q, mu, filter_list[i], dir_ij, idx_i, idx_j, n_atoms)
            q, mu = mixing(q, mu)

        q = q.squeeze(1)  # 压缩原子特征的维度

        inputs["scalar_representation"] = q
        inputs["vector_representation"] = mu
        return inputs
