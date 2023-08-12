from __future__ import annotations

from typing import Dict, Optional, List

from schnetpack.transform import Transform
import schnetpack.properties as properties
from schnetpack.utils import as_dtype

import torch
import torch.nn as nn

# __all__ 是一个特殊的变量，用于定义模块的公共接口。当使用 from module import * 导入模块时，只有在 __all__ 列表中列出的属性、函数、类等才会被导入。
__all__ = ["AtomisticModel", "NeuralNetworkPotential"]


class AtomisticModel(nn.Module):
    """
    Base class for all SchNetPack models.

    SchNetPack models should subclass `AtomisticModel` implement the forward method.
    To use the automatic collection of required derivatives, each submodule that
    requires gradients w.r.t to the input, should list them as strings in
    `submodule.required_derivatives = ["input_key"]`. The model needs to call
    `self.collect_derivatives()` at the end of its `__init__`.

    To make use of post-processing transform, the model should call
    `input = self.postprocess(input)` at the end of its `forward`. The post processors
    will only be applied if `do_postprocessing=True`.

    Example:
         class SimpleModel(AtomisticModel):
            def __init__(
                self,
                representation: nn.Module,
                output_module: nn.Module,
                postprocessors: Optional[List[Transform]] = None,
                input_dtype_str: str = "float32",
                do_postprocessing: bool = True,
            ):
                super().__init__(
                    input_dtype_str=input_dtype_str,
                    postprocessors=postprocessors,
                    do_postprocessing=do_postprocessing,
                )
                self.representation = representation
                self.output_modules = output_modules

                self.collect_derivatives()

            def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
                inputs = self.initialize_derivatives(inputs)

                inputs = self.representation(inputs)
                inputs = self.output_module(inputs)

                # apply postprocessing (if enabled)
                inputs = self.postprocess(inputs)
                return inputs

    """

    def __init__(
        self,
        postprocessors: Optional[List[Transform]] = None,
        input_dtype_str: str = "float32",
        do_postprocessing: bool = True,
    ):
        """
        Args:
            postprocessors: Post-processing transforms that may be
                initialized using the `datamodule`, but are not
                applied during training.一个可选的后处理转换器列表
            input_dtype: The dtype of real inputs as string.实数输入的数据类型
            do_postprocessing: If true, post-processing is activated.表示是否启用后处理
        """
        super().__init__()
        self.input_dtype_str = input_dtype_str
        self.do_postprocessing = do_postprocessing  # 后处理标志
        self.postprocessors = nn.ModuleList(postprocessors)  # 转化为模型列表
        self.required_derivatives: Optional[List[str]] = None  # 指定在反向传播时需要计算梯度的输入键
        self.model_outputs: Optional[List[str]] = None  # 指定模型的输出结果

    def collect_derivatives(self) -> List[str]:  # 收集需要在反向传播中计算梯度的输入键
        self.required_derivatives = None  # 清除之前可能已经收集到的梯度输入键
        required_derivatives = set()   # 存储要收集的梯度输入键
        for m in self.modules():  # 遍历模型中模块
            if (
                hasattr(m, "required_derivatives")
                and m.required_derivatives is not None
            ):  # 检查是否有 required_derivatives 属性，且不为None
                required_derivatives.update(m.required_derivatives)  # 将模块的 required_derivatives 添加到 required_derivatives 集合中
        required_derivatives: List[str] = list(required_derivatives)  # 转换为列表
        self.required_derivatives = required_derivatives

    def collect_outputs(self) -> List[str]:  # 收集模型的输出结果
        self.model_outputs = None  # 清除之前收集的模型输出结果
        model_outputs = set()
        for m in self.modules():  # 遍历模型模块（指的是各个层）
            if hasattr(m, "model_outputs") and m.model_outputs is not None:  # 检查是否有 model_outputs 属性，并且该属性不为 None
                model_outputs.update(m.model_outputs)
        model_outputs: List[str] = list(model_outputs)  # 添加模型模块
        self.model_outputs = model_outputs

    def initialize_derivatives(  # 初始化输入张量的梯度，将输入张量中对应的张量启用梯度计算
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        for p in self.required_derivatives:  # 遍历梯度的输出键列表
            if p in inputs.keys():  # 判断这些输入键是否需要计算梯度  键:只包含名字。输入：既包含键又包含值（可以多个键多个值）
                inputs[p].requires_grad_()  # 启用对该张量的梯度计算
        return inputs

    def initialize_transforms(self, datamodule):  # 为每个模块进行初始化操作
        for module in self.modules():
            if isinstance(module, Transform):  # 如果模块是一个转换类型的实例
                module.datamodule(datamodule)

    def postprocess(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:  # 后处理函数
        if self.do_postprocessing:  # 后标志处理为真
            # apply postprocessing
            for pp in self.postprocessors:  # 遍历后处理器列表
                inputs = pp(inputs)  # 对输入进行额外的处理
        return inputs

    def extract_outputs(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:  # 提取模型输出
        results = {k: inputs[k] for k in self.model_outputs}  # 遍历模型输出中的键，将输入字典中对应键的值复制到结果字典中
        return results  # 返回结果字典


class NeuralNetworkPotential(AtomisticModel):
    """
    A generic neural network potential class that sequentially applies a list of input
    modules, a representation module and a list of output modules.

    This can be flexibly configured for various, e.g. property prediction or potential
    energy sufaces with response properties.
    """

    def __init__(
        self,
        representation: nn.Module,
        input_modules: List[nn.Module] = None,
        output_modules: List[nn.Module] = None,
        postprocessors: Optional[List[Transform]] = None,  # 后处理转换列表
        input_dtype_str: str = "float32",  # 输入数据类型
        do_postprocessing: bool = True,  # 是否激活后处理
    ):
        """
        Args:
            representation: The module that builds representation from inputs.
            input_modules: Modules that are applied before representation, e.g. to
                modify input or add additional tensors for response properties.
            output_modules: Modules that predict output properties from the
                representation.
            postprocessors: Post-processing transforms that may be initialized using the
                `datamodule`, but are not applied during training.
            input_dtype_str: The dtype of real inputs.
            do_postprocessing: If true, post-processing is activated.
        """
        super().__init__(
            input_dtype_str=input_dtype_str,
            postprocessors=postprocessors,
            do_postprocessing=do_postprocessing,
        )
        self.representation = representation  # 从输入中构建表示的模块
        self.input_modules = nn.ModuleList(input_modules)  # 在表示之前应用的模块，例如修改输入或添加额外的张量以获取响应属性
        self.output_modules = nn.ModuleList(output_modules)  # 预测输出属性的模块

        self.collect_derivatives()  # 收集需要在反向传播中计算梯度的输入键
        self.collect_outputs()  # 收集输出

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 接受一个包含输入数据的字典作为参数，并返回一个包含输出结果的字典。
        # initialize derivatives for response properties
        inputs = self.initialize_derivatives(inputs)  # 初始化输入张量的梯度，将输入张量中对应的张量启用梯度计算

        for m in self.input_modules:  # 依次处理输入模块列表中的每个模块
            inputs = m(inputs)

        inputs = self.representation(inputs)  # 从输入中构建表示的模块

        for m in self.output_modules:  # 预测输出属性
            inputs = m(inputs)

        # apply postprocessing (if enabled)
        inputs = self.postprocess(inputs)  # 如果启用后处理功能，对inputs进行后处理
        results = self.extract_outputs(inputs)  # 从inputs中提取输出结果

        return results
