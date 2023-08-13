from copy import copy
from typing import Dict

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint as BaseModelCheckpoint

from torch_ema import ExponentialMovingAverage as EMA

import torch
import os
from pytorch_lightning.callbacks import BasePredictionWriter
from typing import List, Any
from schnetpack.task import AtomisticTask
from schnetpack import properties
from collections import defaultdict

# "ModelCheckpoint"：该回调类用于保存训练过程中的最佳模型
# "PredictionWriter"：用于在训练过程中保存模型的预测结果
# "ExponentialMovingAverage"：用于实现指数移动平均，是一种平滑数据的方法
__all__ = ["ModelCheckpoint", "PredictionWriter", "ExponentialMovingAverage"]


class PredictionWriter(BasePredictionWriter):
    """
    Callback to store prediction results using ``torch.save``.
    """

    def __init__(
            self,
            output_dir: str,  # 输出目录
            write_interval: str,  # 写入间隔（batch、epoch等
            write_idx: bool = False,  # 是否写入分子id
    ):
        """
        Args:
            output_dir: output directory for prediction files
            write_interval: can be one of ["batch", "epoch", "batch_and_epoch"]
            write_idx: Write molecular ids for all atoms. This is needed for
                atomic properties like forces.
        """
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.write_idx = write_idx
        os.makedirs(output_dir, exist_ok=True)  # 如果不存在，则创建名为output_dir的目录

    def write_on_batch_end(
        self,
        trainer,  # 训练器
        pl_module: AtomisticTask,  # atomistictask模型
        prediction: Any,  # 预测结果
        batch_indices: List[int],  # 批次索引列表
        batch: Any,  # 批次数据
        batch_idx: int,  # 批次索引
        dataloader_idx: int,  # 数据加载索引
    ):
        bdir = os.path.join(self.output_dir, str(dataloader_idx))  # 根据dataloader_idx创建目录bdir
        os.makedirs(bdir, exist_ok=True)  # 确保目录存在
        torch.save(prediction, os.path.join(bdir, f"{batch_idx}.pt"))  # 将prediction保存为一个.pt文件，保存在目录bdir下

    def write_on_epoch_end(
        self,
        trainer,
        pl_module: AtomisticTask,
        predictions: List[Any],
        batch_indices: List[Any],
    ):
        # collect batches of predictions and restructure
        concatenated_predictions = defaultdict(list)  # 用于存储预测结果，defaultdict(list)可以方便地将数据按照属性名进行分组
        for batch_prediction in predictions[0]:  # 遍历预测列表中每个batch_prediction
            for property_name, data in batch_prediction.items():
                if not self.write_idx and property_name == properties.idx_m:
                    continue
                concatenated_predictions[property_name].append(data)  # 将数据添加到concatenated_predictions字典对应属性的列表中
        concatenated_predictions = {  # 对 concatenated_predictions 字典中的数据进行拼接
            property_name: torch.concat(data)
            for property_name, data in concatenated_predictions.items()
        }

        # save concatenated predictions，保存到output_dir路径下的predictions.pt文件中
        torch.save(
            concatenated_predictions,
            os.path.join(self.output_dir, "predictions.pt"),
        )


class ModelCheckpoint(BaseModelCheckpoint):
    """
    Like the PyTorch Lightning ModelCheckpoint callback,
    but also saves the best inference model with activated post-processing
    """

    def __init__(self, model_path: str, do_postprocessing=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = model_path  # 保存模型的路径
        self.do_postprocessing = do_postprocessing  # 是否进行后处理

    def on_validation_end(self, trainer, pl_module: AtomisticTask) -> None:  # 在验证结束后调用的方法，用于设置训练器和任务
        self.trainer = trainer
        self.task = pl_module
        super().on_validation_end(trainer, pl_module)  # super().on_validation_end(trainer, pl_module)

    def _update_best_and_save(  # 更新最佳模型并保存的方法
        self, current: torch.Tensor, trainer, monitor_candidates: Dict[str, Any]  # current:当前模型评分，monitor：监控指标的候选值
    ):
        # save model checkpoint
        super()._update_best_and_save(current, trainer, monitor_candidates)

        # save best inference model
        # 如果current的类型是torch.Tensor并且是NaN值，将其替换为正无穷大（如果模式为"min"）或负无穷大（如果模式为"max"）
        if isinstance(current, torch.Tensor) and torch.isnan(current):
            current = torch.tensor(float("inf" if self.mode == "min" else "-inf"))

        if current == self.best_model_score:  # 判断是否模型和最佳模型评分是否相等
            if self.trainer.strategy.local_rank == 0:  # 是否是本地模型
                # remove references to trainer and data loaders to avoid pickle error in ddp
                self.task.save_model(self.model_path, do_postprocessing=True)  # 保存模型到指定路径，并进行后处理操作


class ExponentialMovingAverage(Callback):  # 在训练过程中实现指数移动平均算法
    def __init__(self, decay, *args, **kwargs):
        self.decay = decay  # 指数移动平均的衰减因子
        self.ema = None  # 指数移动平均对象
        self._to_load = None  # 要加载的状态字典

    def on_fit_start(self, trainer, pl_module: AtomisticTask):  # 初始化工作
        if self.ema is None:  # 如果ema为空。则创建一个指数移动平均对象，并设置衰减因子
            self.ema = EMA(pl_module.model.parameters(), decay=self.decay)
        if self._to_load is not None:  # 如果状态字典不空，加载状态字典到ema中
            self.ema.load_state_dict(self._to_load)
            self._to_load = None

        # load average parameters, to have same starting point as after validation
        self.ema.store()  # 保存当前模型的参数状态，确保在验证后重新开始训练时，模型参数的起始点与验证后保持一致
        self.ema.copy_to()  # 将指数移动平均对象中的参数拷贝到模型中，在进行验证时模型使用的是平滑后的参数，而不是原始的参数值

    def on_train_epoch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.ema.restore()  # 训练开始时，将ema中参数恢复到模型中

    def on_train_batch_end(self, trainer, pl_module: AtomisticTask, *args, **kwargs):
        self.ema.update()  # 训练batch结束时，更新ema参数

    def on_validation_epoch_start(  # 每个验证epoch开始时，将模型参数保存到ema中，并将ema中的参数拷贝到模型中。
        self, trainer: "pl.Trainer", pl_module: AtomisticTask, *args, **kwargs
    ):
        self.ema.store()
        self.ema.copy_to()

    def load_state_dict(self, state_dict):  # 加载状态字典。如果状态字典中包含"ema"键，并且ema为空，将状态字典保存到_to_load中；否则，将状态字典加载到ema中。
        if "ema" in state_dict:
            if self.ema is None:
                self._to_load = state_dict["ema"]
            else:
                self.ema.load_state_dict(state_dict["ema"])

    def state_dict(self):  # 返回包含ema状态字典的字典
        return {"ema": self.ema.state_dict()}
