from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.nn.init import zeros_

import schnetpack.properties as structure
from schnetpack.nn import Dense, scatter_add
from schnetpack.nn.activations import shifted_softplus
from schnetpack.representation.schnet import SchNetInteraction
from schnetpack.utils import required_fields_from_properties

from schnetpack import properties
import schnetpack.nn as snn

__all__ = ["FieldSchNet", "NuclearMagneticMomentEmbedding"]


class FieldSchNetFieldInteraction(nn.Module):  # 用于对偶极特征与外部场进行交互，计算标量特征的整体更新
    """
    Model interaction of dipole features with external fields (see Ref. [#field2]_).
    Computes the overall update to the scalar features.

    Args:
        external_fields (list(str)): List of external fields
        n_atom_basis (int): Number of atomic features
        activation (Callable): Activation function for internal transformations.

    References:
    .. [#field2] Gastegger, Schütt, Müller:
       Machine learning of solvent effects on molecular spectra and reactions.
       Chemical Science, 12(34), 11473-11483. 2021.
    """

    def __init__(
        self,
        external_fields: List[str],  # 外部场的列表
        n_atom_basis: int,  # 原子特征的数量
        activation: Callable = shifted_softplus,  # 内部变换的激活函数
    ):
        super(FieldSchNetFieldInteraction, self).__init__()
        # 该字典的键为 external_fields 列表中的每个元素，值为一个 Dense 模块，接收 n_atom_basis 个输入特征，生成 n_atom_basis 个输出特征，并使用激活函数 activation。
        self.f2out = nn.ModuleDict(
            {
                field: Dense(n_atom_basis, n_atom_basis, activation=activation)
                for field in external_fields
            }
        )
        self.external_fields = external_fields

    def forward(
        self, mu: Dict[str, torch.Tensor], external_fields: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the update based on the fields.

        Args:
            mu (dict(str, torch.Tensor): Model dipole features.
            external_fields (dict(str, torch.Tensor): External fields

        Returns:
            torch.Tensor: Field update of scalar features.
        """
        dq = 0.0

        for field in self.external_fields:  # 外部场列表中的每个场
            v = torch.sum(mu[field] * external_fields[field], dim=1, keepdim=True)
            v = self.f2out[field](v)
            dq = dq + v
        # 对标量特征进行更新
        return dq


class DipoleUpdate(nn.Module):  # 基于邻居原子上的标量表示来更新偶极矩特征
    """
    Update the dipole moment features based on the scalar representations on the neighbor atoms.

    Args:
        external_fields list(str): List of external fields.
        n_atom_basis (int): Number of atomic features.  原子特征的数量
    """

    def __init__(self, external_fields: List[str], n_atom_basis: int):
        super(DipoleUpdate, self).__init__()
        self.external_fields = external_fields  # 外部场的列表

        # zero init is important here, otherwise updates grow uncontrollably
        self.transform = nn.ModuleDict(  # 字典模块,该字典的键为 external_fields 列表中的每个元素，值为一个 Dense 模块，接收 n_atom_basis 个输入特征和没有激活函数和偏置的输出特征
            {
                field: Dense(
                    n_atom_basis,
                    n_atom_basis,
                    activation=None,
                    bias=False,
                )
                for field in external_fields
            }
        )

    def forward(
        self,
        q: torch.Tensor,  # 标量特征
        mu: Dict[str, torch.Tensor],  # 标量特征
        v_ij: torch.Tensor,  # 未归一化的方向向量
        idx_i: torch.Tensor,  # 邻居对的索引
        idx_j: torch.Tensor,  # 邻居对的索引
        rcut_ij: torch.Tensor,  # 距离截断
    ) -> Dict[str, torch.Tensor]:
        """
        Perform dipole feature update.

        Args:
            q (torch.Tensor): Scalar features
            mu (dict(str, torch.Tensor): Dipole features.
            v_ij (torch.Tensor): Unnormalized direction vectors.
            idx_i (torch.Tensor): Indices of neighbor pairs atom i.
            idx_j (torch.Tensor): Indices of neighbor pairs atom j.
            rcut_ij (torch.Tensor): Cutoff for distances between i and j.

        Returns:
            dict(str, torch.Tensor): Updated dipole features for all fields.
        """
        for field in self.external_fields:
            # 对于外部场列表中的每个场 field，首先将输入的标量特征 q 通过 transform[field] 进行线性变换，得到变换后的特征向量 qi。
            qi = self.transform[field](q)
            # 根据邻居对的索引 idx_j，从 qi 中选择相应的元素，并乘以距离截断 rcut_ij 和方向向量 v_ij，计算出更新量 dmu_ij。
            dmu_ij = qi[idx_j] * rcut_ij[:, None, None] * v_ij[:, :, None]
            # 将 dmu_ij 中的值按照索引 idx_i 进行累加，并将结果存储在 dmu_i 中
            dmu_i = scatter_add(dmu_ij, idx_i, dim_size=q.shape[0])
            # 将 dmu_i 添加到原始的偶极特征 mu[field] 上，以完成偶极特征的更新
            mu[field] = mu[field] + dmu_i

        return mu


class DipoleInteraction(nn.Module):  # 用于计算偶极特征之间的相互作用，并更新标量特征
    def __init__(
        self,
        external_fields: List[str],
        n_atom_basis: int,
        n_rbf: int,
        activation: Callable = shifted_softplus,
    ):
        """
        Compute the update to the scalar features based on the interactions between the dipole features.
        This uses the classical dipole-dipole interaction Tensor weighted by a radial basis function, as introduced in
        [#field3]_

        Args:
            external_fields (list(str)): List of external fields.
            n_atom_basis (int): Number of atomic features.
            n_rbf (int): Number of radial basis functions used in distance expansion.
            activation (Callable): Activation function.

        References:
        .. [#field3] Gastegger, Schütt, Müller:
           Machine learning of solvent effects on molecular spectra and reactions.
           Chemical Science, 12(34), 11473-11483. 2021.
        """
        super(DipoleInteraction, self).__init__()
        self.external_fields = external_fields

        self.transform = nn.ModuleDict(  # 用于对外部场进行线性变换
            {
                field: Dense(n_atom_basis, n_atom_basis, activation=activation)
                for field in external_fields
            }
        )
        self.filter_network = nn.ModuleDict(  # 用于滤波操作
            {
                field: nn.Sequential(
                    Dense(n_rbf, n_atom_basis, activation=activation),
                    Dense(
                        n_atom_basis, n_atom_basis, activation=None, weight_init=zeros_
                    ),
                )
                for field in external_fields
            }
        )

    def forward(
        self,
        q: torch.Tensor,
        mu: Dict[str, torch.Tensor],
        f_ij: torch.Tensor,
        d_ij: torch.Tensor,
        v_ij: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        rcut_ij: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the update to the scalar features based on the dipole-dipole interactions.

        Args:
            q (torch.Tensor): Scalar features
            mu (dict(str, torch.Tensor): Dipole features.
            f_ij (torch.Tensor): Distance expansion of interatomic distances.
            d_ij (torch.Tensor): Interatomic distances.
            v_ij (torch.Tensor): Unnormalized direction vectors.
            idx_i (torch.Tensor): Indices of neighbor pairs atom i.
            idx_j (torch.Tensor): Indices of neighbor pairs atom j.
            rcut_ij (torch.Tensor): Cutoff for distances between i and j

        Returns:
            torch.Tensor: Scalar update.
        """
        dq = 0.0  # 用于存储标量特征的更新量

        for field in self.external_fields:
            # 对距离展开项 f_ij 进行线性变换，得到权重张量 Wij
            Wij = self.filter_network[field](f_ij) * rcut_ij[..., None]
            Wij = Wij.unsqueeze(1)

            mu_ij = mu[field][idx_j]
            # Dipole - dipole interaction tensor
            # 将 Wij 和偶极特征 mu[field] 根据邻居索引 idx_j 进行选择和乘法操作，得到偶极-偶极相互作用张量 tensor_ij。
            tensor_ij = mu_ij * d_ij[:, None, None] ** 2 - 3.0 * v_ij[
                :, :, None
            ] * torch.sum(v_ij[:, :, None] * mu_ij, dim=1, keepdim=True)
            tensor_ij = tensor_ij * Wij / d_ij[:, None, None] ** 5
            # 对 tensor_ij 进行累加操作，得到相应的更新量 tensor_i。
            tensor_i = scatter_add(tensor_ij, idx_i, dim_size=q.shape[0])
            # 将偶极特征 mu[field] 与 tensor_i 进行元素级乘法，并求和得到更新量 dq_i
            dq_i = torch.sum(mu[field] * tensor_i, dim=1, keepdim=True)
            # 将 dq_i 经过 transform[field] 进行非线性变换
            dq_i = self.transform[field](dq_i)

            # 将每个外部场的更新量累加
            dq = dq + dq_i

        return dq


class NuclearMagneticMomentEmbedding(nn.Module):  # 用于对核磁矩进行特殊的嵌入处理
    """
    Special embedding for nuclear magnetic moments, since they can scale differently based on an atoms gyromagnetic
    ratio.

    Args:
        n_atom_basis (int): number of features to describe atomic environments.
        max_z (int): Maximum number of atom types used in embedding.
    """

    def __init__(self, n_atom_basis: int, max_z: int):
        super(NuclearMagneticMomentEmbedding, self).__init__()
        self.gyromagnetic_ratio = nn.Embedding(max_z, 1, padding_idx=0)  # 一个 Embedding 层，用于存储原子的陀螺磁比
        self.vector_mapping = snn.Dense(1, n_atom_basis, activation=None, bias=False)  # 一个 Dense 层，用于将核磁矩映射到原子特征空间

    def forward(self, Z: torch.Tensor, nuclear_magnetic_moments: torch.Tensor):
        gamma = self.gyromagnetic_ratio(Z).unsqueeze(-1)  # 根据原子序数 Z 获取对应的陀螺磁比 gamma
        delta_nmm = self.vector_mapping(nuclear_magnetic_moments.unsqueeze(-1))  # 使用 vector_mapping 对核磁矩 nuclear_magnetic_moments 进行线性映射

        # for linear f f(a*x) = a * f(x)
        dmu = gamma * delta_nmm  # 核磁矩更新量

        return dmu


class FieldSchNet(nn.Module):  # 进行分子相互作用
    """FieldSchNet architecture for modeling interactions with external fields and response properties as described in
    [#field4]_.

    References:
    .. [#field4] Gastegger, Schütt, Müller:
       Machine learning of solvent effects on molecular spectra and reactions.
       Chemical Science, 12(34), 11473-11483. 2021.
    """

    def __init__(
        self,
        n_atom_basis: int,  # 描述原子环境的特征数量，决定了嵌入向量的大小
        n_interactions: int,  # 相互作用块的数量
        radial_basis: nn.Module,  # 扩展原子间距离的基函数
        external_fields: List[str] = [],  # 外部场列表，指定了所需的外部场
        response_properties: Optional[List[str]] = None,  # 所需的响应属性列表。如果不为None，则用于确定所需的外部场。
        cutoff_fn: Optional[Callable] = None,  # 截断函数
        activation: Optional[Callable] = shifted_softplus,  # 激活函数
        n_filters: int = None,  # 连续滤波卷积中使用的滤波器数量
        shared_interactions: bool = False,  # 如果为True，在相互作用块和生成滤波器网络之间共享权重
        max_z: int = 100,  # 嵌入中使用的最大原子类型数
        electric_field_modifier: Optional[nn.Module] = None,  # 用于修饰电场的模块
    ):
        """
        Args:
            n_atom_basis: number of features to describe atomic environments.
                This determines the size of each embedding vector; i.e. embeddings_dim.
            n_interactions: number of interaction blocks.
            radial_basis: layer for expanding interatomic distances in a basis set
            external_fields (list(str)): List of required external fields. Either this or the requested response
                                         properties needs to be specified.
            response_properties (list(str)): List of requested response properties. If this is not None, it is used to
                                             determine the required external fields.
            cutoff_fn: cutoff function
            activation (callable): activation function for nonlinearities.
            n_filters: number of filters used in continuous-filter convolution
            shared_interactions: if True, share the weights across
                interaction blocks and filter-generating networks.
            max_z (int): Maximum number of atom types used in embedding.
            electric_field_modifier (torch.nn.Module): If provided, use this module to modify the electric field. E.g.
                                                       for solvent models or fields from point charges in QM/MM.
        """
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.size = (self.n_atom_basis,)
        self.n_filters = n_filters or self.n_atom_basis
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn

        if response_properties is not None:
            external_fields = required_fields_from_properties(response_properties)

        self.external_fields = external_fields
        self.electric_field_modifier = electric_field_modifier

        # layers  嵌入层
        self.embedding = nn.Embedding(max_z, self.n_atom_basis, padding_idx=0)

        if properties.magnetic_field in self.external_fields:
            self.nmm_embedding = NuclearMagneticMomentEmbedding(
                n_atom_basis=n_atom_basis, max_z=max_z
            )
        else:
            self.nmm_embedding = None

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

        # External field interactions  相互作用块
        self.field_interaction = snn.replicate_module(
            lambda: FieldSchNetFieldInteraction(
                external_fields=self.external_fields,
                n_atom_basis=n_atom_basis,
                activation=activation,
            ),
            n_interactions,
            shared_interactions,
        )

        # Dipole interaction  外部场相互作用
        self.dipole_interaction = snn.replicate_module(
            lambda: DipoleInteraction(
                external_fields=self.external_fields,
                n_atom_basis=n_atom_basis,
                n_rbf=self.radial_basis.n_rbf,
                activation=activation,
            ),
            n_interactions,
            shared_interactions,
        )

        # Dipole updates  偶极子相互作用
        self.initial_dipole_update = DipoleUpdate(
            external_fields=self.external_fields, n_atom_basis=n_atom_basis
        )
        self.dipole_update = snn.replicate_module(
            lambda: DipoleUpdate(
                external_fields=self.external_fields, n_atom_basis=n_atom_basis
            ),
            n_interactions,
            shared_interactions,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:  # 计算原子的表示或嵌入
        """
        Compute atomic representations/embeddings.

        Args:
            inputs (dict of torch.Tensor): SchNetPack dictionary of input tensors.包含了原子的信息和原子之间的距离等

        Returns:
            torch.Tensor: atom-wise representation.
            list of torch.Tensor: intermediate atom-wise representations, if
            return_intermediate=True was used.
        """
        # 从输入字典中获取原子编号(atomic_numbers)、原子之间距离(r_ij)等信息
        atomic_numbers = inputs[structure.Z]
        r_ij = inputs[structure.Rij]
        idx_i = inputs[structure.idx_i]
        idx_j = inputs[structure.idx_j]
        idx_m = inputs[properties.idx_m]

        # Bring fields to final shape for model
        external_fields = {
            field: inputs[field][idx_m].unsqueeze(-1) for field in self.external_fields
        }

        # Apply field modifier
        if self.electric_field_modifier is not None:
            external_fields[properties.electric_field] = external_fields[
                properties.electric_field
            ] + self.electric_field_modifier(inputs)

        # compute atom and pair features
        d_ij = torch.norm(r_ij, dim=1)
        f_ij = self.radial_basis(d_ij)
        rcut_ij = self.cutoff_fn(d_ij)

        # 使用嵌入层(embedding)将原子编号转换为嵌入向量(q)。
        q = self.embedding(atomic_numbers)[:, None]
        qs = q.shape
        mu = {
            field: torch.zeros((qs[0], 3, qs[2]), device=q.device)
            for field in self.external_fields
        }

        # First dipole update based on embeddings
        mu = self.initial_dipole_update(q, mu, r_ij, idx_i, idx_j, rcut_ij)

        # 如果存在磁场嵌入层(nmm_embedding)，则将核磁矩(nuclear_magnetic_moments)转换为磁场嵌入，并将其与偶极子向量相加。
        if self.nmm_embedding is not None:
            mu[properties.magnetic_field] = mu[
                properties.magnetic_field
            ] + self.nmm_embedding(
                atomic_numbers, inputs[properties.nuclear_magnetic_moments]
            )

        # 通过多个相互作用块进行更新。对于每个相互作用块，依次进行以下步骤：
        # 使用基本的SchNet更新，计算原子的变化量(dq)。
        # 计算外部场的变化量(dq_field)。
        # 计算偶极子的变化量(dq_dipole)。
        # 将以上三个变化量相加，并将其应用于原子的表示(q)。
        # 更新偶极子向量(mu）。

        for (
            i,
            (interaction, field_interaction, dipole_interaction, dipole_update),
        ) in enumerate(
            zip(
                self.interactions,
                self.field_interaction,
                self.dipole_interaction,
                self.dipole_update,
            )
        ):
            # Basic SchNet update
            dq = interaction(q.squeeze(1), f_ij, idx_i, idx_j, rcut_ij).unsqueeze(1)

            # Field and dipole updates
            dq_field = field_interaction(mu, external_fields)
            dq_dipole = dipole_interaction(
                q, mu, f_ij, d_ij, r_ij, idx_i, idx_j, rcut_ij
            )

            dq = dq + dq_field + dq_dipole
            q = q + dq

            mu = dipole_update(dq, mu, r_ij, idx_i, idx_j, rcut_ij)

        inputs["scalar_representation"] = q.squeeze(1)
        return inputs
