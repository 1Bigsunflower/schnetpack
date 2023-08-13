import torch

__all__ = ["ReduceLROnPlateau"]

# 对ReduceLROnPlateau学习率调度器进行扩展，添加监测指标的指数移动平均平滑处理。
class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """
    Extends PyTorch ReduceLROnPlateau by exponential smoothing of the monitored metric

    """

    def __init__(
        self,
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-8,
        verbose=False,
        smoothing_factor=0.0,  # 控制指数移动平均的平滑因子
    ):
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            mode (str): One of `min`, `max`. In `min` mode, lr will
                be reduced when the quantity monitored has stopped
                decreasing; in `max` mode it will be reduced when the
                quantity monitored has stopped increasing. Default: 'min'.
            factor (float): Factor by which the learning rate will be
                reduced. new_lr = lr * factor. Default: 0.1.
            patience (int): Number of epochs with no improvement after
                which learning rate will be reduced. For example, if
                `patience = 2`, then we will ignore the first 2 epochs
                with no improvement, and will only decrease the LR after the
                3rd epoch if the loss still hasn't improved then.
                Default: 10.
            threshold (float): Threshold for measuring the new optimum,
                to only focus on significant changes. Default: 1e-4.
            threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
                dynamic_threshold = best * ( 1 + threshold ) in 'max'
                mode or best * ( 1 - threshold ) in `min` mode.
                In `abs` mode, dynamic_threshold = best + threshold in
                `max` mode or best - threshold in `min` mode. Default: 'rel'.
            cooldown (int): Number of epochs to wait before resuming
                normal operation after lr has been reduced. Default: 0.
            min_lr (float or list): A scalar or a list of scalars. A
                lower bound on the learning rate of all param groups
                or each group respectively. Default: 0.
            eps (float): Minimal decay applied to lr. If the difference
                between new and old lr is smaller than eps, the update is
                ignored. Default: 1e-8.
            verbose (bool): If ``True``, prints a message to stdout for
                each update. Default: ``False``.
            smoothing_factor: smoothing_factor of exponential moving average
        """
        super().__init__(
            optimizer=optimizer,  # 优化器
            mode=mode,  # min/max模式，根据监控指标减小或增大时，更改学习速率
            factor=factor,  # 学习率降低因子，新学习率=就学习率*因子
            patience=patience,  # 在没有改善的情况下经过多少迭代后，学习率降低
            threshold=threshold,  # 测量新最优解的阈值
            threshold_mode=threshold_mode,  # 取值为rel或abs之一。不同模式下的动态阈值计算方法不同
            cooldown=cooldown,  # 降低学习速率后，恢复正常操作之前等待的迭代次数
            min_lr=min_lr,  # 学习率下限
            eps=eps,  # 应用于学习率的最小衰减，新旧学习率之间差异小于eps，则忽略更新
            verbose=verbose,  # 每次更新时是否将消息打印到stdout
        )
        self.smoothing_factor = smoothing_factor  # 指数移动平均的平滑因子
        self.ema_loss = None  # 存储指数移动平均损失值

    def step(self, metrics, epoch=None):  # 用于在每个训练步骤中执行指数移动平均的更新
        current = float(metrics)  # 当前度量指标值
        if self.ema_loss is None:  # 确保第一次更新时ema_loss有初值
            self.ema_loss = current
        else:
            # 计算指数移动平均的损失值
            self.ema_loss = (
                self.smoothing_factor * self.ema_loss
                + (1.0 - self.smoothing_factor) * current
            )
        super().step(current, epoch)  # 将当前的度量指标值和训练周期传递，进行相应的更新操作
