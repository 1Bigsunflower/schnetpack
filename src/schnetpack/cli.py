import logging
import os
import uuid
import tempfile
import socket
from typing import List

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.logger import Logger

import schnetpack as spk
from schnetpack.utils import str2class
from schnetpack.utils.script import log_hyperparameters, print_config
from schnetpack.data import BaseAtomsData, AtomsLoader
from schnetpack.train import PredictionWriter
from schnetpack import properties

log = logging.getLogger(__name__)

# 生成一个 UUID并将其转换为字符串。
OmegaConf.register_new_resolver("uuid", lambda x: str(uuid.uuid1()))
# 函数创建一个临时目录，并返回该目录的路径。
OmegaConf.register_new_resolver("tmpdir", tempfile.mkdtemp, use_cache=True)

header = """
   _____      __    _   __     __  ____             __  
  / ___/_____/ /_  / | / /__  / /_/ __ \____ ______/ /__
  \__ \/ ___/ __ \/  |/ / _ \/ __/ /_/ / __ `/ ___/ //_/
 ___/ / /__/ / / / /|  /  __/ /_/ ____/ /_/ / /__/ ,<   
/____/\___/_/ /_/_/ |_/\___/\__/_/    \__,_/\___/_/|_|                                                          
"""


@hydra.main(config_path="configs", config_name="train", version_base="1.2")
def train(config: DictConfig):
    """
    General training routine for all models defined by the provided hydra configs.

    """
    print(header)
    # 输出当前程序运行的主机名
    log.info("Running on host: " + str(socket.gethostname()))
    # 检查配置文件中是否缺少"data_dir"字段
    if OmegaConf.is_missing(config, "run.data_dir"):
        log.error(
            f"Config incomplete! You need to specify the data directory `data_dir`."
        )
        return
    # 检查配置文件中是否缺少model和data字段
    if not ("model" in config and "data" in config):
        log.error(
            f"""
        Config incomplete! You have to specify at least `data` and `model`! 
        For an example, try one of our pre-defined experiments:
        > spktrain data_dir=/data/will/be/here +experiment=qm9
        """
        )
        return
    # 检查配置文件
    if os.path.exists("config.yaml"):
        log.info(
            f"Config already exists in given directory {os.path.abspath('.')}."
            + " Attempting to continue training."
        )

        # save old config
        old_config = OmegaConf.load("config.yaml")  # 加载原始配置文件
        count = 1
        while os.path.exists(f"config.old.{count}.yaml"):
            count += 1
        with open(f"config.old.{count}.yaml", "w") as f:
            OmegaConf.save(old_config, f, resolve=False)

        # resume from latest checkpoint
        if config.run.ckpt_path is None:
            if os.path.exists("checkpoints/last.ckpt"):
                config.run.ckpt_path = "checkpoints/last.ckpt"
        # 显示正在从哪个检查点进行恢复训练
        if config.run.ckpt_path is not None:
            log.info(
                f"Resuming from checkpoint {os.path.abspath(config.run.ckpt_path)}"
            )
    else:
        # 将当前配置保存到 "config.yaml" 文件中
        with open("config.yaml", "w") as f:
            OmegaConf.save(config, f, resolve=False)

    if config.get("print_config"):
        print_config(config, resolve=False)

    # Set seed for random number generators in pytorch, numpy and python.random
    # 进行随机数生成的初始化
    if "seed" in config:
        log.info(f"Seed with <{config.seed}>")
        seed_everything(config.seed, workers=True)
    else:
        log.info(f"Seed randomly...")
        seed_everything(workers=True)

    if not os.path.exists(config.run.data_dir):
        os.makedirs(config.run.data_dir)

    # Init Lightning datamodule
    log.info(f"Instantiating datamodule <{config.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.data)

    # Init model
    log.info(f"Instantiating model <{config.model._target_}>")
    model = hydra.utils.instantiate(config.model)

    # Init LightningModule
    log.info(f"Instantiating task <{config.task._target_}>")
    scheduler_cls = (
        str2class(config.task.scheduler_cls) if config.task.scheduler_cls else None
    )
    # 传入配置中的task参数、model参数、optimizer_cls参数、scheduler_cls参数
    task: spk.AtomisticTask = hydra.utils.instantiate(
        config.task,
        model=model,
        optimizer_cls=str2class(config.task.optimizer_cls),
        scheduler_cls=scheduler_cls,
    )

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[Logger] = []

    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                l = hydra.utils.instantiate(lg_conf)

                logger.append(l)

    # Init Lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=os.path.join(config.run.id),
        _convert_="partial",
    )

    log.info("Logging hyperparameters.")
    log_hyperparameters(config=config, model=task, trainer=trainer)

    # Train the model
    log.info("Starting training.")
    trainer.fit(model=task, datamodule=datamodule, ckpt_path=config.run.ckpt_path)

    # Evaluate model on test set after training
    log.info("Starting testing.")
    trainer.test(model=task, datamodule=datamodule, ckpt_path="best")

    # Store best model
    best_path = trainer.checkpoint_callback.best_model_path
    log.info(f"Best checkpoint path:\n{best_path}")

    log.info(f"Store best model")
    best_task = type(task).load_from_checkpoint(best_path)
    torch.save(best_task, config.globals.model_path + ".task")

    best_task.save_model(config.globals.model_path, do_postprocessing=True)
    log.info(f"Best model stored at {os.path.abspath(config.globals.model_path)}")


@hydra.main(config_path="configs", config_name="predict", version_base="1.2")
def predict(config: DictConfig):
    log.info(f"Load data from `{config.data.datapath}`")
    # 读入数据集
    dataset: BaseAtomsData = hydra.utils.instantiate(config.data)
    loader = AtomsLoader(dataset, batch_size=config.batch_size, num_workers=8)
    # 读入最佳模型
    model = torch.load("best_model")

    class WrapperLM(LightningModule):
        def __init__(self, model, enable_grad=config.enable_grad):
            super().__init__()
            self.model = model
            self.enable_grad = enable_grad  # 控制是否进行梯度计算

        def forward(self, x):
            return model(x)

        def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
            torch.set_grad_enabled(self.enable_grad)
            # 获取预测值
            results = self(batch)
            # 将指定索引的样本从 batch 中提取出来，并将其存储到 results 字典中特定的键对应的值中
            results[properties.idx_m] = batch[properties.idx][batch[properties.idx_m]]
            # 分离字典中的键和值，并将数据转移到 CPU 上，最后构造成一个新的字典并重新赋值给result。这样可以避免在后续使用 results 中的数据时产生梯度计算的额外开销和内存消耗
            results = {k: v.detach().cpu() for k, v in results.items()}
            return results

    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    # 用Trainer对象来管理模型的训练
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        # 将推理结果写入文件，output_dir：保存推理结果，write_interval：每隔多少个批次将结果写入文件一次，write_idx：指定写入的索引
        callbacks=[
            PredictionWriter(
                output_dir=config.outputdir, write_interval=config.write_interval, write_idx=config.write_idx_m
            )
        ],
        default_root_dir=".",
        # 用于将函数的默认参数设置为部分参数。目的是为了允许在配置文件中省略一些参数，而使用在代码中定义的默认值。使用部分参数有助于简化配置文件，并减少重复定义
        _convert_="partial",
    )
    # 预测结果
    trainer.predict(
        WrapperLM(model, config.enable_grad),
        dataloaders=loader,
        # 用于指定检查点文件的路径
        ckpt_path=config.ckpt_path,
    )
