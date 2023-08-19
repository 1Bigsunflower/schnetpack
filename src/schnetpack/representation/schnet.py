from typing import Callable, Dict

import torch
from torch import nn

import schnetpack.properties as structure
from schnetpack.nn import Dense, scatter_add
from schnetpack.nn.activations import shifted_softplus

import schnetpack.nn as snn

__all__ = ["SchNet", "SchNetInteraction"]


# SchNet交互块
class SchNetInteraction(nn.Module):  # 用于建模原子系统之间的相互作用
    r"""SchNet interaction block for modeling interactions of atomistic systems."""

    def __init__(
        self,
        n_atom_basis: int,  # 原子特征维度(
        n_rbf: int,  # 径向基函数的数量
        n_filters: int,  # 连续滤波器卷积中的滤波器数量
        activation: Callable = shifted_softplus,  # 激活函数
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
            n_rbf (int): number of radial basis functions.
            n_filters: number of filters used in continuous-filter convolution.
            activation: if None, no activation function is used.
        """
        super(SchNetInteraction, self).__init__()
        self.in2f = Dense(n_atom_basis, n_filters, bias=False, activation=None)
        self.f2out = nn.Sequential(
            Dense(n_filters, n_atom_basis, activation=activation),
            Dense(n_atom_basis, n_atom_basis, activation=None),
        )
        self.filter_network = nn.Sequential(
            Dense(n_rbf, n_filters, activation=activation), Dense(n_filters, n_filters)
        )

    def forward(
        self,
        x: torch.Tensor,
        f_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        rcut_ij: torch.Tensor,
    ):
        """Compute interaction output.

        Args:
            x: input values
            Wij: filter
            idx_i: index of center atom i
            idx_j: index of neighbors j

        Returns:
            atom features after interaction
        """
        x = self.in2f(x)  # 将输入特征x通过线性变换层(in2f)转换为滤波器空间的特征
        Wij = self.filter_network(f_ij)  # 使用滤波器网络(filter_network)将径向基函数特征f_ij转换为滤波器Wij
        Wij = Wij * rcut_ij[:, None]  # 通过乘以截断距离矩阵rcut_ij进行截断操作

        # continuous-filter convolution  连续滤波器卷积操作
        x_j = x[idx_j]  # 根据邻居原子索引idx_j获取对应原子的特征x_j
        x_ij = x_j * Wij  # 原子间的相互作用
        x = scatter_add(x_ij, idx_i, dim_size=x.shape[0])  # 利用scatter_add函数将相互作用特征x_ij聚合到中心原子上，得到更新后的原子特征

        x = self.f2out(x)  # 通过两个线性变换层(f2out)对原子特征x进行进一步的变换和映射，并返回计算得到的更新后的原子特征。
        return x
        # 根据输入的原子特征和相互作用信息，计算并更新原子的表示。


class SchNet(nn.Module):  # 用于学习原子系统的表示，通过连续滤波器卷积来建模原子之间的相互作用
    """SchNet architecture for learning representations of atomistic systems

    References:

    .. [#schnet1] Schütt, Arbabzadah, Chmiela, Müller, Tkatchenko:
       Quantum-chemical insights from deep tensor neural networks.
       Nature Communications, 8, 13890. 2017.
    .. [#schnet_transfer] Schütt, Kindermans, Sauceda, Chmiela, Tkatchenko, Müller:
       SchNet: A continuous-filter convolutional neural network for modeling quantum
       interactions.
       In Advances in Neural Information Processing Systems, pp. 992-1002. 2017.
    .. [#schnet3] Schütt, Sauceda, Kindermans, Tkatchenko, Müller:
       SchNet - a deep learning architecture for molceules and materials.
       The Journal of Chemical Physics 148 (24), 241722. 2018.

    """

    def __init__(
        self,
        n_atom_basis: int,  # 原子特征维度
        n_interactions: int,  # 相互作用块的数量
        radial_basis: nn.Module,  # 径向基函数
        cutoff_fn: Callable,  # 截断函数
        n_filters: int = None,  # 连续滤波器数量
        shared_interactions: bool = False,  # 是否共享相互作用权重
        max_z: int = 100,  # 最大核电荷
        activation: Callable = shifted_softplus,  # 激活函数
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            n_filters: number of filters used in continuous-filter convolution
            shared_interactions: if True, share the weights across
                interaction blocks and filter-generating networks.
            max_z: maximal nuclear charge
            activation: activation function
        """
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.size = (self.n_atom_basis,)
        self.n_filters = n_filters or self.n_atom_basis
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff

        # layers
        # 嵌入层（Embedding Layer），用于将原子序数映射为原子特征向量。其中，max_z是最大核电荷值，self.n_atom_basis是原子特征维度（embedding维度），padding_idx=0表示使用零作为填充索引。
        self.embedding = nn.Embedding(max_z, self.n_atom_basis, padding_idx=0)

        # 多层相互作用块的集合。使用函数replicate_module重复生成n_interactions个相互作用块。每个相互作用块都是SchNetInteraction模型的实例
        self.interactions = snn.replicate_module(
            lambda: SchNetInteraction(
                n_atom_basis=self.n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                n_filters=self.n_filters,
                activation=activation,
            ),
            n_interactions,
            shared_interactions,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        # 从输入字典中获取
        atomic_numbers = inputs[structure.Z]  # 原子序数张量
        r_ij = inputs[structure.Rij]  # 原子之间的距离张量
        idx_i = inputs[structure.idx_i]  # 中心原子索引张量
        idx_j = inputs[structure.idx_j]  # 邻居原子索引张量

        # compute atom and pair features
        x = self.embedding(atomic_numbers)  # 通过嵌入层(self.embedding)将原子序数(atomic_numbers)转换为原子特征向量。
        d_ij = torch.norm(r_ij, dim=1)  # 计算原子之间的距离d_ij,通过计算torch.norm函数在第一个维度上求范数得到
        f_ij = self.radial_basis(d_ij)  # 径向基函数对象(self.radial_basis)将距离d_ij转化为径向基函数特征f_ij
        rcut_ij = self.cutoff_fn(d_ij)  # 截断矩阵特征rcut_ij

        # compute interaction block to update atomic embeddings
        for interaction in self.interactions:  # 循环遍历多个相互作用块(self.interactions)，依次更新原子特征x
            v = interaction(x, f_ij, idx_i, idx_j, rcut_ij)
            x = x + v

        inputs["scalar_representation"] = x
        return inputs
