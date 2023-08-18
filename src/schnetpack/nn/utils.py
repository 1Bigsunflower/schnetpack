from typing import Callable, Optional

import torch
from torch import nn as nn

__all__ = ["replicate_module", "derivative_from_atomic", "derivative_from_molecular"]

from torch.autograd import grad


def replicate_module(
        module_factory: Callable[[], nn.Module],  # 用于创建单个模块
        n: int,  # n：重复次数，表示要创建多少个相同的模块
        share_params: bool  # 用于指定是否共享模块的参数
):
    if share_params:
        # 创建一个包含 n 个相同模块的模块列表
        module_list = nn.ModuleList([module_factory()] * n)
    else:
        # 为每个重复的模块创建独立的实例
        module_list = nn.ModuleList([module_factory() for i in range(n)])
    return module_list


def derivative_from_molecular(  # 用于计算 fx 相对于 dx 的导数。这个函数主要用于计算分子属性（如能量、偶极矩等）
        fx: torch.Tensor,  # 要对其导数的张量
        dx: torch.Tensor,  # 导数
        create_graph: bool = False,  # 指定是否创建计算图
        retain_graph: bool = False,  # 是否保留计算图
):
    """
    Compute the derivative of `fx` with respect to `dx` if the leading dimension of `fx` is the number of molecules
    (e.g. energies, dipole moments, etc).

    Args:
        fx (torch.Tensor): Tensor for which the derivative is taken.
        dx (torch.Tensor): Derivative.
        create_graph (bool): Create computational graph.
        retain_graph (bool): Keep the computational graph.

    Returns:
        torch.Tensor: derivative of `fx` with respect to `dx`.
    """
    # 获取张量形状
    fx_shape = fx.shape
    dx_shape = dx.shape
    # Final shape takes into consideration whether derivative will yield atomic or molecular properties
    final_shape = (dx_shape[0], *fx_shape[1:], *dx_shape[1:])

    fx = fx.view(fx_shape[0], -1)

    dfdx = torch.stack(
        [
            grad(
                fx[..., i],
                dx,
                torch.ones_like(fx[..., i]),
                create_graph=create_graph,
                retain_graph=retain_graph,
            )[0]
            for i in range(fx.shape[1])
        ],
        dim=1,
    )
    dfdx = dfdx.view(final_shape)
    # 计算 fx 相对于 dx 的导数，并根据导数会产生原子或分子属性的情况确定最终结果的形状。
    return dfdx


def derivative_from_atomic(  # 计算张量 fx 相对于 dx 的导数的函数，适用于具有原子维度的 fx（例如位置、速度等），并可以用于计算 Hessian 矩阵和类似的响应属性（例如核自旋-自旋耦合）
        fx: torch.Tensor,  # 要对其求导数的张量
        dx: torch.Tensor,  # 导数
        n_atoms: torch.Tensor,  # 包含每个分子的原子数的张量
        create_graph: bool = False,  # 指定是否创建计算图
        retain_graph: bool = False,  # 指定是否创建计算图
):
    """
    Compute the derivative of a tensor with the leading dimension of (batch x atoms) with respect to another tensor of
    either dimension (batch * atoms) (e.g. R) or (batch * atom pairs) (e.g. Rij). This function is primarily used for
    computing Hessians and Hessian-like response properties (e.g. nuclear spin-spin couplings). The final tensor will
    have the shape ( batch * atoms * atoms x ....).

    This is quite inefficient, use with care.

    Args:
        fx (torch.Tensor): Tensor for which the derivative is taken.
        dx (torch.Tensor): Derivative.
        n_atoms (torch.Tensor): Tensor containing the number of atoms for each molecule.
        create_graph (bool): Create computational graph.
        retain_graph (bool): Keep the computational graph.

    Returns:
        torch.Tensor: derivative of `fx` with respect to `dx`.
    """
    # Split input tensor for easier bookkeeping
    fxm = fx.split(list(n_atoms))  # 按照 n_atoms 列表拆分成多个子张量

    dfdx = []  # 用于存储导数结果

    n_mol = 0
    # Compute all derivatives
    for idx in range(len(fxm)):  # 循环遍历每个分子的张量。对于每个分子的张量 fxm[idx]，将其重塑为一维张量
        fx = fxm[idx].view(-1)

        # Generate the individual derivatives
        dfdx_mol = []
        for i in range(fx.shape[0]):  # 遍历张量 fx 中的每个元素，并使用 grad() 函数计算其相对于 dx 的导数
            dfdx_i = grad(
                fx[i],
                dx,
                torch.ones_like(fx[i]),
                create_graph=create_graph,
                retain_graph=retain_graph,
            )[0]

            dfdx_mol.append(dfdx_i[n_mol: n_mol + n_atoms[idx], ...])

        # Build molecular matrix and reshape
        # 将所有原子的导数结果堆叠在一起
        dfdx_mol = torch.stack(dfdx_mol, dim=0)
        dfdx_mol = dfdx_mol.view(n_atoms[idx], 3, n_atoms[idx], 3)
        dfdx_mol = dfdx_mol.permute(0, 2, 1, 3)
        dfdx_mol = dfdx_mol.reshape(n_atoms[idx] ** 2, 3, 3)

        dfdx.append(dfdx_mol)

        n_mol += n_atoms[idx]

    # Accumulate everything
    dfdx = torch.cat(dfdx, dim=0)

    return dfdx
