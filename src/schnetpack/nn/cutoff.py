import math
import torch
from torch import nn

__all__ = [
    "CosineCutoff",
    "MollifierCutoff",
    "mollifier_cutoff",
    "cosine_cutoff",
    "SwitchFunction",
]


def cosine_cutoff(input: torch.Tensor, cutoff: torch.Tensor):  #  Behler 风格的余弦截断函数
    """ Behler-style cosine cutoff.

        .. math::
           f(r) = \begin{cases}
            0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
              & r < r_\text{cutoff} \\
            0 & r \geqslant r_\text{cutoff} \\
            \end{cases}

        Args:
            cutoff (float, optional): cutoff radius.

        """
    # 小于截断半径 cutoff 的部分按照余弦函数进行缩放，大于等于截断半径 cutoff 的部分被截断为0
    # Compute values of cutoff function
    input_cut = 0.5 * (torch.cos(input * math.pi / cutoff) + 1.0)
    # Remove contributions beyond the cutoff radius
    input_cut *= (input < cutoff).float()
    return input_cut


class CosineCutoff(nn.Module):  # 实现 Behler 风格的余弦截断函数
    r""" Behler-style cosine cutoff module.

    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    """

    def __init__(self, cutoff: float):
        """
        Args:
            cutoff (float, optional): cutoff radius.截断半径
        """
        super(CosineCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))  # 将其注册为模块的缓冲区

    def forward(self, input: torch.Tensor):
        return cosine_cutoff(input, self.cutoff)


def mollifier_cutoff(input: torch.Tensor, cutoff: torch.Tensor, eps: torch.Tensor):  # 分子截断函数，用于在分子模拟中对原子之间的距离进行截断操作
    r""" Mollifier cutoff scaled to have a value of 1 at :math:`r=0`.

    .. math::
       f(r) = \begin{cases}
        \exp\left(1 - \frac{1}{1 - \left(\frac{r}{r_\text{cutoff}}\right)^2}\right)
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff: Cutoff radius.
        eps: Offset added to distances for numerical stability.

    """
    mask = (input + eps < cutoff).float()
    exponent = 1.0 - 1.0 / (1.0 - torch.pow(input * mask / cutoff, 2))
    cutoffs = torch.exp(exponent)
    cutoffs = cutoffs * mask
    return cutoffs


class MollifierCutoff(nn.Module):  # 分子截断函数模块
    r""" Mollifier cutoff module scaled to have a value of 1 at :math:`r=0`.

    .. math::
       f(r) = \begin{cases}
        \exp\left(1 - \frac{1}{1 - \left(\frac{r}{r_\text{cutoff}}\right)^2}\right)
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}
    """

    def __init__(self, cutoff: float, eps: float = 1.0e-7):  # 初始化截断半径和偏移量
        """
        Args:
            cutoff: Cutoff radius.
            eps: Offset added to distances for numerical stability.
        """
        super(MollifierCutoff, self).__init__()
        self.register_buffer("cutoff", torch.FloatTensor([cutoff]))
        self.register_buffer("eps", torch.FloatTensor([eps]))

    def forward(self, input: torch.Tensor):  # 分子截断函数的值
        return mollifier_cutoff(input, self.cutoff, self.eps)


def _switch_component(
    x: torch.Tensor, ones: torch.Tensor, zeros: torch.Tensor
) -> torch.Tensor:
    """
    Basic component of switching functions.

    Args:
        x (torch.Tensor): Switch functions.
        ones (torch.Tensor): Tensor with ones.
        zeros (torch.Tensor): Zero tensor

    Returns:
        torch.Tensor: Output tensor.
    """
    x_ = torch.where(x <= 0, ones, x)
    # 在 x <= 0 的位置上为 0，在 x > 0 的位置上根据指数函数计算得到
    return torch.where(x <= 0, zeros, torch.exp(-ones / x_))


class SwitchFunction(nn.Module):  # 实现从 1 到 0 之间的衰减开关函数
    """
    Decays from 1 to 0 between `switch_on` and `switch_off`.
    """

    def __init__(self, switch_on: float, switch_off: float):
        """

        Args:
            switch_on (float): Onset of switch.
            switch_off (float): Value from which on switch is 0.
        """
        super(SwitchFunction, self).__init__()
        # 分别是开关函数的起始点和终止点
        self.register_buffer("switch_on", torch.Tensor([switch_on]))
        self.register_buffer("switch_off", torch.Tensor([switch_off]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): tensor to which switching function should be applied to.

        Returns:
            torch.Tensor: switch output
        """
        # 将输入 x 转化为开关函数的输入范围
        x = (x - self.switch_on) / (self.switch_off - self.switch_on)

        # 创建与输入 x 相同形状的全为 1 和全为 0 的张量
        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)
        # 正向开关函数的输出
        fp = _switch_component(x, ones, zeros)
        # 反向开关函数的输出
        fm = _switch_component(1 - x, ones, zeros)

        # 如果 x 小于等于 0，则输出为 1；如果 x 大于等于 1，则输出为 0；否则按照公式 (fm / (fp + fm)) 计算开关函数的输出。
        f_switch = torch.where(x <= 0, ones, torch.where(x >= 1, zeros, fm / (fp + fm)))
        return f_switch
