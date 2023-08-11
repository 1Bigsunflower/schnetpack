from typing import Union, Dict, Sequence

import pytorch_lightning as pl
import rich
import yaml
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from rich.syntax import Syntax
from rich.tree import Tree

# log_hyperparameters:将配置信息保存到Lightning loggers中，采用三个参数：
# config：一个 DictConfig 对象或字典配置。包含了整个实验的配置信息，比如学习速率、batch_size、layer数量等
# model：模型。
# trainer：用于训练和管理模型的训练器。
# 会将配置信息发送给所有的 loggers

# print_config用于使用 Rich 库和树状结构打印 DictConfig 的内容
__all__ = ["log_hyperparameters", "print_config"]


def empty(*args, **kwargs):
    pass


# 将配置对象转换为字典格式
def todict(config: Union[DictConfig, Dict]):
    # 将配置对象转换为 YAML 字符串，设置 resolve=True 参数来解析配置中的引用字段，并加载字典对象
    config_dict = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))
    return config_dict


@rank_zero_only
def log_hyperparameters(
        config: DictConfig,
        model: pl.LightningModule,
        trainer: pl.Trainer,
) -> None:
    """
    This saves Hydra config using Lightning loggers.
    """

    # send hparams to all loggers
    trainer.logger.log_hyperparams(config)  # 将配置信息发送给所有的 loggers，以记录超参数

    # disable logging any more hyperparameters for all loggers
    trainer.logger.log_hyperparams = empty  # 禁用后续的超参数记录


@rank_zero_only
def print_config(
        config: DictConfig,  # 包含了要打印的配置信息
        fields: Sequence[str] = (  # 确定要打印的字段，并决定顺序。
                "run",
                "globals",
                "data",
                "model",
                "task",
                "trainer",
                "callbacks",
                "logger",
                "seed",
        ),
        resolve: bool = True,  # 指示是否解析 DictConfig 的引用字段
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Config.
        fields (Sequence[str], optional): Determines which main fields from config will be printed
        and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"  # 节点和内容的样式
    tree = Tree(
        f":gear: Running with the following config:", style=style, guide_style=style
    )

    for field in fields:  # 遍历 fields 参数中的字段
        branch = tree.add(field, style=style, guide_style=style)  # 设置节点样式和引导线的样式

        config_section = config.get(field)  # 从配置中获取指定字段的部分
        branch_content = str(config_section)  # 转换为字符串格式
        # 检查 config_section 是否是 DictConfig 对象，并将其转换为 YAML 格式的字符串
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(Syntax(branch_content, "yaml"))  # 字符串branch_content添加到分支节点branch中

    rich.print(tree)
