import torch
from torchmetrics import Metric
from torchmetrics.functional.regression.mae import (
    _mean_absolute_error_compute,
    _mean_absolute_error_update,
)

from typing import Optional, Tuple

__all__ = ["TensorDiagonalMeanAbsoluteError"]


# 监测张量（例如极化率）对角线和非对角线上的平均绝对误差
class TensorDiagonalMeanAbsoluteError(Metric):
    """
    Custom torch metric for monitoring the mean absolute error on the diagonals and offdiagonals of tensors, e.g.
    polarizability.
    """

    is_differentiable = True  # 指示该指标是否可导
    higher_is_better = False  # 是否指标值更高更好
    sum_abs_error: torch.Tensor  # 存储绝对误差之和的张量
    total: torch.Tensor  # 存储样本总数

    def __init__(
            self,
            diagonal: Optional[bool] = True,  # 表示是否使用对角线的值，false表示非对角线
            diagonal_dims: Optional[Tuple[int, int]] = (-2, -1),  # 张量中表示方阵的轴的维度，（-2，-1）表示最后两个维度为方阵的维度
            dist_sync_on_step=False,  # 是否进行同步操作
    ) -> None:
        """

        Args:
            diagonal (bool): If true, diagonal values are used, if False off-diagonal.
            diagonal_dims (tuple(int,int)): axes of the square matrix for which the diagonals should be considered.
            dist_sync_on_step (bool): synchronize.
        """
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # 将多个进程中的值求和
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

        self.diagonal = diagonal
        self.diagonal_dims = diagonal_dims
        self._diagonal_mask = None

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:  # 更新指标
        """
        Update the metric.

        Args:
            preds (torch.Tensor): network predictions.
            target (torch.Tensor): reference values.
        """
        # update metric states
        # 转换格式
        preds = self._input_format(preds)  # 预测值
        target = self._input_format(target)  # 参考值

        sum_abs_error, n_obs = _mean_absolute_error_update(preds, target)  # 计算preds和target之间平均绝对误差和观测样本数量

        self.sum_abs_error += sum_abs_error
        self.total += n_obs

    def compute(self) -> torch.Tensor:  # 最终计算指标：对角线或非对角线元素的平均绝对误差
        """
        Compute the final metric.

        Returns:
            torch.Tensor: mean absolute error of diagonal or offdiagonal elements.
        """
        # compute final result
        return _mean_absolute_error_compute(self.sum_abs_error, self.total)

    def _input_format(self, x) -> torch.Tensor:  # 从输入张量中提取对角线/非对角线元素
        """
        Extract diagonal / offdiagonal elements from input tensor.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: extracted and flattened elements (diagonal / offdiagonal)
        """
        if self._diagonal_mask is None:
            self._diagonal_mask = self._init_diag_mask(x)
        return x.masked_select(self._diagonal_mask)

    # 根据给定的轴和输入张量的形状初始化用于提取对角线元素的掩码
    def _init_diag_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Initialize the mask for extracting the diagonal elements based on the given axes and the shape of the
        input tensor.

        Args:
            x (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: Boolean diagonal mask.
        """
        tensor_shape = x.shape  # 获取输入张量形状
        # 根据形状找到对应的维度，这两个维度决定方阵大小
        dim_0 = tensor_shape[self.diagonal_dims[0]]
        dim_1 = tensor_shape[self.diagonal_dims[1]]

        # 判断是否为方阵
        if not dim_0 == dim_1:
            raise AssertionError(
                "Found different size for diagonal dimensions, expected square sub matrix."
            )

        # 创建形状与张量相同的全为1的视图，并将该方阵两个维度更新为dim_0
        view = [1 for _ in tensor_shape]
        view[self.diagonal_dims[0]] = dim_0
        view[self.diagonal_dims[1]] = dim_1

        # 创建单位矩阵
        diag_mask = torch.eye(dim_0, device=x.device, dtype=torch.long).view(view)

        # 判断提取对角线还是非对角线元素，
        if self.diagonal:
            # 返回对角线元素的布尔型掩码
            return diag_mask == 1
        else:
            # 返回非对角线元素的布尔型掩码
            return diag_mask != 1
