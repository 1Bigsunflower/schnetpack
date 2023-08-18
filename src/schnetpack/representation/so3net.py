from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

import schnetpack.nn as snn
import schnetpack.nn.so3 as so3
import schnetpack.properties as properties

__all__ = ["SO3net"]


class SO3net(nn.Module):  # 用于生成原子环境的特征表示
    """
    A simple SO3-equivariant representation using spherical harmonics and
    Clebsch-Gordon tensor products.

    """

    def __init__(
        self,
        n_atom_basis: int,  # 描述原子环境的特征数量，决定了嵌入向量的大小
        n_interactions: int,  # 交互块的数量
        lmax: int,  # 球面谐波函数的最大角动量
        radial_basis: nn.Module,  # 用于在一组基函数中扩展原子间距离的层
        cutoff_fn: Optional[Callable] = None,  # 截断函数
        shared_interactions: bool = False,  # 是否共享交互块
        max_z: int = 100,  # 原子序数的最大值
        return_vector_representation: bool = False,  # 是否返回笛卡尔坐标系中的 l=1 特征
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            lmax: maximum angular momentum of spherical harmonics basis
            radial_basis: layer for expanding interatomic distances in a basis set
            cutoff_fn: cutoff function
            shared_interactions:
            max_z:
            conv_layer:
            return_vector_representation: return l=1 features in Cartesian XYZ order
                (e.g. for DipoleMoment output module)
        """
        super(SO3net, self).__init__()

        self.n_atom_basis = n_atom_basis
        self.n_interactions = n_interactions
        self.lmax = lmax
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
        self.radial_basis = radial_basis
        self.return_vector_representation = return_vector_representation

        self.embedding = nn.Embedding(max_z, n_atom_basis, padding_idx=0)
        self.sphharm = so3.RealSphericalHarmonics(lmax=lmax)

        self.so3convs = snn.replicate_module(
            lambda: so3.SO3Convolution(lmax, n_atom_basis, self.radial_basis.n_rbf),
            self.n_interactions,
            shared_interactions,
        )
        self.mixings1 = snn.replicate_module(
            lambda: nn.Linear(n_atom_basis, n_atom_basis, bias=False),
            self.n_interactions,
            shared_interactions,
        )
        self.mixings2 = snn.replicate_module(
            lambda: nn.Linear(n_atom_basis, n_atom_basis, bias=False),
            self.n_interactions,
            shared_interactions,
        )
        self.mixings3 = snn.replicate_module(
            lambda: nn.Linear(n_atom_basis, n_atom_basis, bias=False),
            self.n_interactions,
            shared_interactions,
        )
        self.gatings = snn.replicate_module(
            lambda: so3.SO3ParametricGatedNonlinearity(n_atom_basis, lmax),
            self.n_interactions,
            shared_interactions,
        )
        self.so3product = so3.SO3TensorProduct(lmax)

    def forward(self, inputs: Dict[str, torch.Tensor]):  # 用于计算原子表示或嵌入

        """
        Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        # get tensors from input dictionary  从输入字典中获取张量
        atomic_numbers = inputs[properties.Z]  # 原子序号
        r_ij = inputs[properties.Rij]  # 原子对之间的距离
        # 原子对的索引
        idx_i = inputs[properties.idx_i]
        idx_j = inputs[properties.idx_j]

        # compute atom and pair features  计算原子和原子对的特征
        d_ij = torch.norm(r_ij, dim=1, keepdim=True)  # 计算原子对距离的范数
        dir_ij = r_ij / d_ij  # 计算原子对方向向量

        Yij = self.sphharm(dir_ij)  # 使用球面谐波函数计算
        radial_ij = self.radial_basis(d_ij)  # 使用径向基函数计算
        cutoff_ij = self.cutoff_fn(d_ij)[..., None]  # 使用截断函数计算

        x0 = self.embedding(atomic_numbers)[:, None]  # 初始化原子的表示
        x = so3.scalar2rsh(x0, int(self.lmax))  # 球谐函数展开的表示 x

        for so3conv, mixing1, mixing2, gating, mixing3 in zip(
                self.so3convs, self.mixings1, self.mixings2, self.gatings, self.mixings3
        ):
            dx = so3conv(x, radial_ij, Yij, cutoff_ij, idx_i, idx_j)
            ddx = mixing1(dx)
            dx = dx + self.so3product(dx, ddx)
            dx = mixing2(dx)
            dx = gating(dx)
            dx = mixing3(dx)
            x = x + dx

        # 将计算得到的原子表示存储在输入字典中，包括标量表示和多极表示
        inputs["scalar_representation"] = x[:, 0]
        inputs["multipole_representation"] = x

        # extract cartesian vector from multipoles: [y, z, x] -> [x, y, z]
        if self.return_vector_representation:
            inputs["vector_representation"] = torch.roll(x[:, 1:4], 1, 1)

        return inputs
