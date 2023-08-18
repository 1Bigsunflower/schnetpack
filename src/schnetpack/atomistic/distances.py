from typing import Dict, Optional

import torch
import torch.nn as nn

import schnetpack.properties as properties


class PairwiseDistances(nn.Module):  # 根据给定的邻居列表索引计算分子间的距离
    """
    Compute pair-wise distances from indices provided by a neighbor list transform.
    """

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        R = inputs[properties.R]  # properties.R：表示原子的位置
        offsets = inputs[properties.offsets]  # properties.offsets：表示偏移量，用于处理周期性边界条件
        idx_i = inputs[properties.idx_i]  # properties.offsets：表示偏移量，用于处理周期性边界条件
        idx_j = inputs[properties.idx_j]  # properties.idx_j：表示分子对中的第二个分子的索引

        Rij = R[idx_j] - R[idx_i] + offsets
        inputs[properties.Rij] = Rij  # 计算出分子对之间的距离，并将结果存储在properties.Rij键下
        return inputs


class FilterShortRange(nn.Module):  # 用于将短程和长程距离分开
    """
    Separate short-range from all supplied distances.

    The short-range distances will be stored under the original keys (properties.Rij,
    properties.idx_i, properties.idx_j), while the original distances can be accessed for long-range terms via
    (properties.Rij_lr, properties.idx_i_lr, properties.idx_j_lr).
    """

    def __init__(self, short_range_cutoff: float):
        super().__init__()
        self.short_range_cutoff = short_range_cutoff  # 表示短程距离的截断值

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 将满足短程距离条件的距离和索引从输入字典中提取出来，并将剩余的距离和索引存储在新的键下

        idx_i = inputs[properties.idx_i]  # properties.idx_i：原子对的第一个原子的索引
        idx_j = inputs[properties.idx_j]  # properties.idx_j：原子对的第二个原子的索引
        Rij = inputs[properties.Rij]  # 原子对之间的距离

        rij = torch.norm(Rij, dim=-1)
        cidx = torch.nonzero(rij <= self.short_range_cutoff).squeeze(-1)  # 将满足短程距离条件(rij <= self.short_range_cutoff)的索引提取出来

        # 将原始的距离和索引存储在新的键下
        inputs[properties.Rij_lr] = Rij
        inputs[properties.idx_i_lr] = idx_i
        inputs[properties.idx_j_lr] = idx_j

        # 将满足短程距离条件的距离和索引更新为输入字典的新值
        inputs[properties.Rij] = Rij[cidx]
        inputs[properties.idx_i] = idx_i[cidx]
        inputs[properties.idx_j] = idx_j[cidx]
        return inputs
