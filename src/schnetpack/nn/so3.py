import math
import torch
import torch.nn as nn
import schnetpack.nn as snn
from .ops.so3 import generate_clebsch_gordan_rsh, sparsify_clebsch_gordon, sh_indices
from .ops.math import binom
from schnetpack.utils import as_dtype

__all__ = [
    "RealSphericalHarmonics",
    "SO3TensorProduct",
    "SO3Convolution",
    "SO3GatedNonlinearity",
    "SO3ParametricGatedNonlinearity",
]


class RealSphericalHarmonics(nn.Module):  # 用于生成一批向量的实球谐函数
    """
    Generates the real spherical harmonics for a batch of vectors.

    Note:
        The vectors passed to this layer are assumed to be normalized to unit length.

    Spherical harmonics are generated up to angular momentum `lmax` in dimension 1,
    according to the following order:
    - l=0, m=0
    - l=1, m=-1
    - l=1, m=0
    - l=1, m=1
    - l=2, m=-2
    - l=2, m=-1
    - etc.
    """

    def __init__(self, lmax: int, dtype_str: str = "float32"):
        """
        Args:
            lmax: maximum angular momentum
            dtype_str: dtype for spherical harmonics coefficients
        """
        super().__init__()
        self.lmax = lmax  # 最大角动量

        (
            powers,
            zpow,
            cAm,
            cBm,
            cPi,
        ) = self._generate_Ylm_coefficients(lmax)  # 计算球谐函数的系数

        # 注册缓冲区（buffer），在模型训练过程中被自动优化器所优化
        dtype = as_dtype(dtype_str)
        self.register_buffer("powers", powers.to(dtype=dtype), False)
        self.register_buffer("zpow", zpow.to(dtype=dtype), False)
        self.register_buffer("cAm", cAm.to(dtype=dtype), False)
        self.register_buffer("cBm", cBm.to(dtype=dtype), False)
        self.register_buffer("cPi", cPi.to(dtype=dtype), False)

        ls = torch.arange(0, lmax + 1)  # 生成一个从0到lmax的张量ls
        nls = 2 * ls + 1  # 新的张量nls，其中每个元素是对应ls中的元素乘以2再加1
        self.lidx = torch.repeat_interleave(ls, nls)  # ls对应的m值列表self.lidx
        self.midx = torch.cat([torch.arange(-l, l + 1) for l in ls])  # 角动量指标列表midx

        self.register_buffer("flidx", self.lidx.to(dtype=dtype), False)  # 将ls作为缓冲区flidx进行注册

    def _generate_Ylm_coefficients(self, lmax: int):  # 计算球谐函数的系数
        # see: https://en.wikipedia.org/wiki/Spherical_harmonics#Real_forms

        # calculate Am/Bm coefficients
        m = torch.arange(1, lmax + 1, dtype=torch.float64)[:, None]
        p = torch.arange(0, lmax + 1, dtype=torch.float64)[None, :]
        mask = p <= m
        mCp = binom(m, p)
        cAm = mCp * torch.cos(0.5 * math.pi * (m - p))
        cBm = mCp * torch.sin(0.5 * math.pi * (m - p))
        cAm *= mask
        cBm *= mask
        powers = torch.stack([torch.broadcast_to(p, cAm.shape), m - p], dim=-1)
        powers *= mask[:, :, None]

        # calculate Pi coefficients
        l = torch.arange(0, lmax + 1, dtype=torch.float64)[:, None, None]
        m = torch.arange(0, lmax + 1, dtype=torch.float64)[None, :, None]
        k = torch.arange(0, lmax // 2 + 1, dtype=torch.float64)[None, None, :]
        cPi = torch.sqrt(torch.exp(torch.lgamma(l - m + 1) - torch.lgamma(l + m + 1)))
        cPi = cPi * (-1) ** k * 2 ** (-l) * binom(l, k) * binom(2 * l - 2 * k, l)
        cPi *= torch.exp(torch.lgamma(l - 2 * k + 1) - torch.lgamma(l - 2 * k - m + 1))
        zpow = l - 2 * k - m

        # masking of invalid entries
        cPi = torch.nan_to_num(cPi, 100.0)
        mask1 = k <= torch.floor((l - m) / 2)
        mask2 = l >= m
        mask = mask1 * mask2
        cPi *= mask
        zpow *= mask

        return powers, zpow, cAm, cBm, cPi

    def forward(self, directions: torch.Tensor) -> torch.Tensor:  # 计算给定输入方向上的真实球谐函数值
        """
        Args:
            directions: batch of unit-length 3D vectors (Nx3)

        Returns:
            real spherical harmonics up ton angular momentum `lmax`
        """
        target_shape = [
            directions.shape[0],
            self.powers.shape[0],
            self.powers.shape[1],
            2,
        ]
        Rs = torch.broadcast_to(directions[:, None, None, :2], target_shape)
        pows = torch.broadcast_to(self.powers[None], target_shape)

        Rs = torch.where(pows == 0, torch.ones_like(Rs), Rs)

        temp = Rs**self.powers
        monomials_xy = torch.prod(temp, dim=-1)

        Am = torch.sum(monomials_xy * self.cAm[None], 2)
        Bm = torch.sum(monomials_xy * self.cBm[None], 2)
        ABm = torch.cat(
            [
                torch.flip(Bm, (1,)),
                math.sqrt(0.5) * torch.ones((Am.shape[0], 1), device=directions.device),
                Am,
            ],
            dim=1,
        )
        ABm = ABm[:, self.midx + self.lmax]

        target_shape = [
            directions.shape[0],
            self.zpow.shape[0],
            self.zpow.shape[1],
            self.zpow.shape[2],
        ]
        z = torch.broadcast_to(directions[:, 2, None, None, None], target_shape)
        zpows = torch.broadcast_to(self.zpow[None], target_shape)
        z = torch.where(zpows == 0, torch.ones_like(z), z)
        zk = z**zpows

        Pi = torch.sum(zk * self.cPi, dim=-1)  # batch x L x M
        Pi_lm = Pi[:, self.lidx, abs(self.midx)]
        sphharm = torch.sqrt((2 * self.flidx + 1) / (2 * math.pi)) * Pi_lm * ABm
        return sphharm


def scalar2rsh(x: torch.Tensor, lmax: int) -> torch.Tensor:  # 用于将形状为 [N, *] 的标量张量扩展成具有最大角动量 lmax 的球谐函数形状的张量
    """
    Expand scalar tensor to spherical harmonics shape with angular momentum up to `lmax`

    Args:
        x: tensor of shape [N, *]
        lmax: maximum angular momentum

    Returns:
        zero-padded tensor to shape [N, (lmax+1)^2, *]
    """
    # 创建零张量，这个零张量表示了在球谐函数形状中需要填充的部分。
    y = torch.cat(
        [
            x,
            torch.zeros(
                (x.shape[0], int((lmax + 1) ** 2 - 1), x.shape[2]),
                device=x.device,
                dtype=x.dtype,
            ),
        ],
        dim=1,  # 在维度1上填充了零的张量 y
    )
    return y  # 经过零填充后的球谐函数形状的张量


class SO3TensorProduct(nn.Module):  # 计算 SO3 等变的 Clebsch-Gordan 张量积
    """
    SO3-equivariant Clebsch-Gordon tensor product.

    With combined indexing s=(l,m), this can be written as:

    .. math::

        y_{s,f} = \sum_{s_1,s_2} x_{2,s_2,f} x_{1,s_2,f}  C_{s_1,s_2}^{s}

    """

    def __init__(self, lmax: int):
        super().__init__()
        self.lmax = lmax

        cg = generate_clebsch_gordan_rsh(lmax).to(torch.float32)
        cg, idx_in_1, idx_in_2, idx_out = sparsify_clebsch_gordon(cg)
        self.register_buffer("idx_in_1", idx_in_1, persistent=False)
        self.register_buffer("idx_in_2", idx_in_2, persistent=False)
        self.register_buffer("idx_out", idx_out, persistent=False)
        self.register_buffer("clebsch_gordan", cg, persistent=False)

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x1: atom-wise SO3 features, shape: [n_atoms, (l_max+1)^2, n_features]
            x2: atom-wise SO3 features, shape: [n_atoms, (l_max+1)^2, n_features]

        Returns:
            y: product of SO3 features

        """
        x1 = x1[:, self.idx_in_1, :]
        x2 = x2[:, self.idx_in_2, :]
        y = x1 * x2 * self.clebsch_gordan[None, :, None]
        y = snn.scatter_add(y, self.idx_out, dim_size=int((self.lmax + 1) ** 2), dim=1)
        return y


class SO3Convolution(nn.Module):  # 对 SO3 特征进行等变卷积的模型，通过在原子间进行旋转不变的卷积运算，能够保留原子特征的等变性质
    """
    SO3-equivariant convolution using Clebsch-Gordon tensor product.

    With combined indexing s=(l,m), this can be written as:

    .. math::

        y_{i,s,f} = \sum_{j,s_1,s_2} x_{j,s_2,f} W_{s_1,f}(r_{ij}) Y_{s_1}(r_{ij}) C_{s_1,s_2}^{s}

    """

    def __init__(self, lmax: int, n_atom_basis: int, n_radial: int):
        super().__init__()
        self.lmax = lmax
        self.n_atom_basis = n_atom_basis
        self.n_radial = n_radial

        cg = generate_clebsch_gordan_rsh(lmax).to(torch.float32)  # 生成 Clebsch-Gordan 系数 cg
        cg, idx_in_1, idx_in_2, idx_out = sparsify_clebsch_gordon(cg)  # 进行稀疏化处理
        # 将索引列表和稀疏化后的 Clebsch-Gordan 系数注册为模型的缓冲区，其中 persistent=False 表示这些缓冲区不会被序列化保存
        self.register_buffer("idx_in_1", idx_in_1, persistent=False)
        self.register_buffer("idx_in_2", idx_in_2, persistent=False)
        self.register_buffer("idx_out", idx_out, persistent=False)
        self.register_buffer("clebsch_gordan", cg, persistent=False)

        self.filternet = snn.Dense(  # 名为 "filternet" 的 Dense 层
            n_radial, n_atom_basis * (self.lmax + 1), activation=None
        )

        lidx, _ = sh_indices(lmax)  # 生成球谐函数的索引 lidx
        self.register_buffer("Widx", lidx[self.idx_in_1])  # 按照索引idx_in_1保存到widx缓冲区

    def _compute_radial_filter(  # 计算径向（具有旋转不变性）滤波器
        self, radial_ij: torch.Tensor, cutoff_ij: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute radial (rotationally-invariant) filter

        Args:
            radial_ij: radial basis functions with shape [n_neighbors, n_radial_basis]
            cutoff_ij: cutoff function with shape [n_neighbors, 1]

        Returns:
            Wij: radial filters with shape [n_neighbors, n_clebsch_gordon, n_features]
        """
        Wij = self.filternet(radial_ij) * cutoff_ij
        Wij = torch.reshape(Wij, (-1, self.lmax + 1, self.n_atom_basis))
        Wij = Wij[:, self.Widx]
        return Wij  # 径向滤波器张量 Wij

    def forward(  # 实现SO3特征卷积操作，在卷积过程中，首先根据原子 j 的索引选择相应的 SO3 特征子集，然后根据径向滤波器、方向向量和 Clebsch-Gordan 系数对子集进行多项式的SO3卷积运算，并将结果累加到对应的原子 i 的特征上，最终得到输出的 SO3 特征
        self,
        x: torch.Tensor,
        radial_ij: torch.Tensor,
        dir_ij: torch.Tensor,
        cutoff_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: atom-wise SO3 features, shape: [n_atoms, (l_max+1)^2, n_atom_basis]
            radial_ij: radial basis functions with shape [n_neighbors, n_radial_basis]
            dir_ij: direction from atom i to atom j, scaled to unit length
                [n_neighbors, 3]
            cutoff_ij: cutoff function with shape [n_neighbors, 1]
            idx_i: indices for atom i
            idx_j: indices for atom j

        Returns:
            y: convolved SO3 features

        """
        xj = x[idx_j[:, None], self.idx_in_2[None, :], :]
        Wij = self._compute_radial_filter(radial_ij, cutoff_ij)

        v = (
            Wij
            * dir_ij[:, self.idx_in_1, None]
            * self.clebsch_gordan[None, :, None]
            * xj
        )
        yij = snn.scatter_add(v, self.idx_out, dim_size=int((self.lmax + 1) ** 2), dim=1)
        y = snn.scatter_add(yij, idx_i, dim_size=x.shape[0])
        return y


class SO3ParametricGatedNonlinearity(nn.Module):  # 实现 SO3-equivariant 参数化门控非线性操作，可以提取和操作输入特征中不同角动量的信息，并通过门控机制对特征进行调节
    """
    SO3-equivariant parametric gated nonlinearity.

    With combined indexing s=(l,m), this can be written as:

    .. math::

        y_{i,s,f} = x_{j,s,f} * \sigma(f(x_{j,0,\cdot}))

    """

    def __init__(self, n_in: int, lmax: int):
        super().__init__()
        self.lmax = lmax  # 最大角动量
        self.n_in = n_in  # 输入特征维度
        self.lidx, _ = sh_indices(lmax)
        self.scaling = nn.Linear(n_in, n_in * (lmax + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = x[:, 0, :]
        h = self.scaling(s0).reshape(-1, self.lmax + 1, self.n_in)
        h = h[:, self.lidx]
        y = x * torch.sigmoid(h)
        return y  # 输出特征张量


class SO3GatedNonlinearity(nn.Module):  # 用于实现 SO3-equivariant 的门控非线性操作，SO3-equivariant 操作是指在三维空间中进行旋转操作时能够保持特征表示不变的操作
    """
    SO3-equivariant gated nonlinearity.

    With combined indexing s=(l,m), this can be written as:

    .. math::

        y_{i,s,f} = x_{j,s,f} * \sigma(x_{j,0,\cdot})

    """

    def __init__(self, lmax: int):
        super().__init__()
        self.lmax = lmax  # 最大的角动量量子数
        self.lidx, _ = sh_indices(lmax)  # 初始化为与角动量 l 相关的索引数组

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 实现了门控非线性操作，通过门控机制控制输入特征的流动
        s0 = x[:, 0, :]  # 表示 l=0、m=0 的角动量分量的特征
        y = x * torch.sigmoid(s0[:, None, :])  # 对 s0 进行激活，将激活后的张量 s0 和输入特征张量 x 逐元素相乘，得到形状相同的张量 y
        return y
