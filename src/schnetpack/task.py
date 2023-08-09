import warnings
from typing import Optional, Dict, List, Type, Any

import pytorch_lightning as pl
import torch
from torch import nn as nn
from torchmetrics import Metric

from schnetpack.model.base import AtomisticModel

__all__ = ["ModelOutput", "AtomisticTask"]


class ModelOutput(nn.Module):
    """
    Defines an output of a model, including mappings to a loss function and weight for training
    and metrics to be logged.
    """

    def __init__(
        self,
        name: str,  # 输出的名称
        loss_fn: Optional[nn.Module] = None,  # 损失函数
        loss_weight: float = 1.0,  # 损失中的损失权重
        metrics: Optional[Dict[str, Metric]] = None,  # 指标字典
        constraints: Optional[List[torch.nn.Module]] = None,
        target_property: Optional[str] = None,  # 用于约束损失函数和日志指标的约束类
    ):
        """
        Args:
            name: name of output in results dict
            target_property: Name of target in training batch. Only required for supervised training.
                If not given, the output name is assumed to also be the target name.
            loss_fn: function to compute the loss
            loss_weight: loss weight in the composite loss: $l = w_1 l_1 + \dots + w_n l_n$
            metrics: dictionary of metrics with names as keys
            constraints:
                constraint class for specifying the usage of model output in the loss function and logged metrics,
                while not changing the model output itself. Essentially, constraints represent postprocessing transforms
                that do not affect the model output but only change the loss value. For example, constraints can be used
                to neglect or weight some atomic forces in the loss function. This may be useful when training on
                systems, where only some forces are crucial for its dynamics.
        """
        super().__init__()
        self.name = name
        self.target_property = target_property or name
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight
        self.train_metrics = nn.ModuleDict(metrics)
        self.val_metrics = nn.ModuleDict({k: v.clone() for k, v in metrics.items()})
        self.test_metrics = nn.ModuleDict({k: v.clone() for k, v in metrics.items()})
        self.metrics = {
            "train": self.train_metrics,
            "val": self.val_metrics,
            "test": self.test_metrics,
        }
        self.constraints = constraints or []

    def calculate_loss(self, pred, target):
        # 计算loss并×损失权重
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0
        loss = self.loss_weight * self.loss_fn(
            pred[self.name], target[self.target_property]
        )
        return loss

    def update_metrics(self, pred, target, subset):
        # 更新指标值。使用给定的预测值和目标值更新训练、验证或测试集合中的指标值。
        for metric in self.metrics[subset].values():
            metric(pred[self.name], target[self.target_property])


class UnsupervisedModelOutput(ModelOutput):  # 不依赖于标签数据的无监督损失或正则化项
    """
    Defines an unsupervised output of a model, i.e. an unsupervised loss or a regularizer
    that do not depend on label data. It includes mappings to the loss function,
    a weight for training and metrics to be logged.
    """

    def calculate_loss(self, pred, target=None):
        # 接受预测值 pred，而不需要目标值 target。它使用给定的预测值计算损失，并乘以损失权重。
        if self.loss_weight == 0 or self.loss_fn is None:
            return 0.0
        loss = self.loss_weight * self.loss_fn(pred[self.name])  # loss(模型输出,模型输入)
        return loss

    def update_metrics(self, pred, target, subset):
        # 同理，接受预测试，不需要目标值。使用预测值更新训练、验证或测试集合中的指标值
        for metric in self.metrics[subset].values():
            metric(pred[self.name])


class AtomisticTask(pl.LightningModule):
    """
    The basic learning task in SchNetPack, which ties model, loss and optimizer together.

    """

    def __init__(
        self,
        model: AtomisticModel,
        outputs: List[ModelOutput],
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.Adam,  # 优化器类型
        optimizer_args: Optional[Dict[str, Any]] = None,  # 优化器的关键字参数字典
        scheduler_cls: Optional[Type] = None,  # 学习率调度器类型
        scheduler_args: Optional[Dict[str, Any]] = None,  # 调度器的关键字参数字典
        scheduler_monitor: Optional[str] = None,  # 用于ReduceLROnPlateau的观察指标的名称
        warmup_steps: int = 0,  # 在训练开始时，从零线性增加学习率到目标学习率所使用的步数
    ):
        """
        Args:
            model: the neural network model
            outputs: list of outputs an optional loss functions
            optimizer_cls: type of torch optimizer,e.g. torch.optim.Adam
            optimizer_args: dict of optimizer keyword arguments
            scheduler_cls: type of torch learning rate scheduler
            scheduler_args: dict of scheduler keyword arguments
            scheduler_monitor: name of metric to be observed for ReduceLROnPlateau
            warmup_steps: number of steps used to increase the learning rate from zero
              linearly to the target learning rate at the beginning of training
        """
        super().__init__()
        self.model = model
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_args
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_args
        self.schedule_monitor = scheduler_monitor
        self.outputs = nn.ModuleList(outputs)

        self.grad_enabled = len(self.model.required_derivatives) > 0
        self.lr = optimizer_args["lr"]
        self.warmup_steps = warmup_steps
        self.save_hyperparameters()  # 将所有超参数保存到日志中，可以方便在训练过程中跟踪和访问超参数的值

    def setup(self, stage=None):
        # 在训练阶段为"fit"时被调用，用于初始化或转换模型的数据
        if stage == "fit":
            # 通过数据模块来初始化或转换模型的数据
            self.model.initialize_transforms(self.trainer.datamodule)

    def forward(self, inputs: Dict[str, torch.Tensor]):
        results = self.model(inputs)
        return results

    def loss_fn(self, pred, batch):
        # 通过遍历self.outputs中的每一个元素，计算每个输出对应的损失，并将其加到总损失loss上。
        # 这样设计可以支持多个输出，并且每个输出可以有不同的损失函数。
        loss = 0.0
        for output in self.outputs:
            loss += output.calculate_loss(pred, batch)
        return loss

    def log_metrics(self, pred, targets, subset):
        # 用于记录模型在每个子集（subset）上的指标（metrics）情况
        for output in self.outputs:
            # 通过遍历self.outputs中的每个输出对象，调用其update_metrics方法来更新针对该输出对象的指标
            output.update_metrics(pred, targets, subset)
            for metric_name, metric in output.metrics[subset].items():
                self.log(
                    f"{subset}_{output.name}_{metric_name}",
                    metric,
                    on_step=(subset == "train"),  # 仅在训练步骤上记录
                    on_epoch=(subset != "train"),  # 在非训练过程的轮次上记录
                    prog_bar=False,  # 不在进度条中显示
                )

    def apply_constraints(self, pred, targets):
        # 应用约束（constraints）到模型的预测输出和真实标签上
        for output in self.outputs:
            for constraint in output.constraints:
                pred, targets = constraint(pred, targets, output)
        return pred, targets

    def training_step(self, batch, batch_idx):
        # 根据每个输出对象的target_property属性，从批次数据中提取出对应的真实标签（targets）。
        # 如果输出对象是UnsupervisedModelOutput类型，则不包含其真实标签。
        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        # 尝试从批次数据中获取"considered_atoms"（考虑的原子）的信息，并将其添加到targets中。如果获取失败（即该信息不在批次数据中），则跳过。
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass
        # predict_without_postprocessing使用在训练步骤中获取模型的预测输出，而不经过后处理，从而获得预测值
        pred = self.predict_without_postprocessing(batch)
        # 将预测输出和真实标签应用约束
        pred, targets = self.apply_constraints(pred, targets)
        # 计算约束后的预测和标签值计算loss
        loss = self.loss_fn(pred, targets)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log_metrics(pred, targets, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        # 判断是否启用梯度计算
        torch.set_grad_enabled(self.grad_enabled)
        # 根据每个输出对象的target_property属性，从批次数据中提取出对应的真实标签（targets）。
        # 如果输出对象是UnsupervisedModelOutput类型，则不包含其真实标签。
        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        # 尝试从批次数据中获取"considered_atoms"（考虑的原子）的信息，并将其添加到targets中。如果获取失败（即该信息不在批次数据中），则跳过。
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass
        # predict_without_postprocessing使用在训练步骤中获取模型的预测输出，而不经过后处理，从而获得预测值
        pred = self.predict_without_postprocessing(batch)
        # 将预测输出和真实标签应用约束
        pred, targets = self.apply_constraints(pred, targets)
        # 计算约束后的预测和标签值计算loss
        loss = self.loss_fn(pred, targets)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(pred, targets, "val")

        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        # 判断是否启用梯度计算
        torch.set_grad_enabled(self.grad_enabled)
        # 根据每个输出对象的target_property属性，从批次数据中提取出对应的真实标签（targets）。
        # 如果输出对象是UnsupervisedModelOutput类型，则不包含其真实标签。
        targets = {
            output.target_property: batch[output.target_property]
            for output in self.outputs
            if not isinstance(output, UnsupervisedModelOutput)
        }
        # 尝试从批次数据中获取"considered_atoms"（考虑的原子）的信息，并将其添加到targets中。如果获取失败（即该信息不在批次数据中），则跳过。
        try:
            targets["considered_atoms"] = batch["considered_atoms"]
        except:
            pass
        # predict_without_postprocessing使用在训练步骤中获取模型的预测输出，而不经过后处理，从而获得预测值
        pred = self.predict_without_postprocessing(batch)
        # 将预测输出和真实标签应用约束
        pred, targets = self.apply_constraints(pred, targets)
        # 计算约束后的预测和标签值计算loss
        loss = self.loss_fn(pred, targets)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_metrics(pred, targets, "test")
        return {"test_loss": loss}

    def predict_without_postprocessing(self, batch):
        # 将模型的do_postprocessing属性保存到变量pp中，以便后续恢复
        pp = self.model.do_postprocessing
        # 将模型的do_postprocessing属性设置为False，以禁用后处理步骤。
        self.model.do_postprocessing = False
        # 使用模型对输入批次进行预测，得到预测结果
        pred = self(batch)
        # 将模型的do_postprocessing属性恢复为之前保存的值pp
        self.model.do_postprocessing = pp
        return pred

    def configure_optimizers(self):
        # 创建一个优化器对象
        optimizer = self.optimizer_cls(
            params=self.parameters(), **self.optimizer_kwargs
        )
        # 检查是否定义了学习率调度器
        if self.scheduler_cls:
            # 创建一个空列表schedulers，用于存储学习率调度器配置
            schedulers = []
            # 创建一个调度器对象
            schedule = self.scheduler_cls(optimizer=optimizer, **self.scheduler_kwargs)
            # 构建一个字典optimconf，包含了调度器对象和名称"lr_schedule"
            optimconf = {"scheduler": schedule, "name": "lr_schedule"}
            # 如果定义了self.schedule_monitor，将其设置为监控指标
            if self.schedule_monitor:
                optimconf["monitor"] = self.schedule_monitor
            # incase model is validated before epoch end (not recommended use of val_check_interval)
            # 根据self.trainer.val_check_interval的值来确定调度器的触发间隔
            if self.trainer.val_check_interval < 1.0:
                # 如果val_check_interval小于1.0，表示在每个训练周期结束之后进行
                warnings.warn(
                    "Learning rate is scheduled after epoch end. To enable scheduling before epoch end, "
                    "please specify val_check_interval by the number of training epochs after which the "
                    "model is validated."
                )
            # incase model is validated before epoch end (recommended use of val_check_interval)
            if self.trainer.val_check_interval > 1.0:
                # 如果val_check_interval大于1.0，表示在每个训练batch过程中，以self.trainer.val_check_interval的频率触发调度器
                optimconf["interval"] = "step"
                optimconf["frequency"] = self.trainer.val_check_interval
            # 将optimconf添加到schedulers列表中
            schedulers.append(optimconf)
            # 返回优化器对象和学习率调度器配置
            return [optimizer], schedulers
        # 如果没定义学习率调度器，则只返回optimizer
        else:
            return optimizer

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer=None,
        optimizer_closure=None,
    ):
        # 检查当前全局步数是否小于预热步数.预热步数是一个在训练开始阶段进行学习率预热的步数。
        if self.global_step < self.warmup_steps:
            # 首先计算一个学习率缩放因子lr_scale，其值为当前全局步数加1除以预热步数的比值，取其最小值不超过1
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.warmup_steps)
            # 遍历每个参数组，将学习率设置为lr_scale乘预设的学习率
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.lr

        # update params执行参数更新操作
        optimizer.step(closure=optimizer_closure)

    def save_model(self, path: str, do_postprocessing: Optional[bool] = None):
        # 全局排名为0的进程保存模型
        if self.global_rank == 0:
            # 获取当前模型的后处理状态,后处理在保存模型之前可以进行一些额外的操作或修改
            pp_status = self.model.do_postprocessing
            # 如果给定了do_postprocessing参数,说明需要修改后处理状态
            if do_postprocessing is not None:
                self.model.do_postprocessing = do_postprocessing
            # 将模型保存到指定的路径
            torch.save(self.model, path)
            # 恢复模型的原始后处理状态
            self.model.do_postprocessing = pp_status


class ConsiderOnlySelectedAtoms(nn.Module):
    """
    Constraint that allows to neglect some atomic targets (e.g. forces of some specified atoms) for model optimization,
    while not affecting the actual model output. The indices of the atoms, which targets to consider in the loss
    function, must be provided in the dataset for each sample in form of a torch tensor of type boolean
    (True: considered, False: neglected).
    """
    # 用于在模型优化过程中忽略一些指定原子的目标（例如某些特定原子的力），而不影响实际模型输出。
    def __init__(self, selection_name):
        # 关联数据集中需要考虑的原子列表
        """
        Args:
            selection_name: string associated with the list of considered atoms in the dataset
        """
        super().__init__()
        self.selection_name = selection_name

    def forward(self, pred, targets, output_module):
        """
        A torch tensor is loaded from the dataset, which specifies the considered atoms. Only the
        predictions of those atoms are considered for training, validation, and testing.

        :param pred: python dictionary containing model outputs
        :param targets: python dictionary containing targets
        :param output_module: torch.nn.Module class of a particular property (e.g. forces)
        :return: model outputs and targets of considered atoms only
        """
        # 从数据集中加载一个张量，该张量指定了需要考虑的原子。只有这些原子的预测结果会被用于训练、验证和测试。
        # 获取到需要考虑的原子的索引
        considered_atoms = targets[self.selection_name].nonzero()[:, 0]

        # drop neglected atoms
        # 保留需要考虑的原子的预测结果
        pred[output_module.name] = pred[output_module.name][considered_atoms]
        # 保留需要考虑的原子的目标值
        targets[output_module.target_property] = targets[output_module.target_property][
            considered_atoms
        ]
        # 返回经过处理后的模型输出和目标值,只包含需要考虑的原子部分的结果
        return pred, targets
