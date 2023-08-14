from typing import Union, Sequence, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import schnetpack.nn as snn

__all__ = ["build_mlp", "build_gated_equivariant_mlp"]


def build_mlp(
    n_in: int,  # 输入节点数
    n_out: int,  # 输出节点数
    n_hidden: Optional[Union[int, Sequence[int]]] = None,  # 隐藏层节点数。如果是整数，则隐藏层具有相同数量节点。如果为None，则每个隐藏层节点数依次减半
    n_layers: int = 2,  # 网络中层数
    activation: Callable = F.silu,  # 隐藏层使用相同激活函数（swish）
    last_bias: bool = True,  # 是否在最后一层添加偏置参数
    last_zero_init: bool = False,  # 是否在最后一层添加零初始化的权重
) -> nn.Module:
    """
    Build multiple layer fully connected perceptron neural network.

    Args:
        n_in: number of input nodes.
        n_out: number of output nodes.
        n_hidden: number hidden layer nodes.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers: number of layers.
        activation: activation function. All hidden layers would
            the same activation function except the output layer that does not apply
            any activation function.
    """
    # get list of number of nodes in input, hidden & output layers
    # 根据输入的参数确定每层的节点数，并存储在 n_neurons 列表中
    if n_hidden is None:
        # 根据层数和输入节点数计算每个隐藏层的节点数
        c_neurons = n_in
        n_neurons = []  # 用于存储每个隐藏层节点数
        for i in range(n_layers):
            n_neurons.append(c_neurons)  # 当前隐藏层节点数添加到n_neurons列表
            c_neurons = max(n_out, c_neurons // 2)  # 每次更新为n_out和当前层/2的较大值
        n_neurons.append(n_out)  # 存储每个隐藏层节点数
    else:
        # get list of number of nodes hidden layers
        # 该整数复制多次以得到每个隐藏层的节点数
        if type(n_hidden) is int:
            n_hidden = [n_hidden] * (n_layers - 1)
        else:
            n_hidden = list(n_hidden)
        # 最终的节点数列表 n_neurons 形成如下：首先是输入层节点数 n_in，接着是所有隐藏层的节点数，最后是输出层节点数 n_out
        n_neurons = [n_in] + n_hidden + [n_out]

    # assign a Dense layer (with activation function) to each hidden layer
    # 为每个隐藏层分配一个带有激活函数的 Dense 层
    layers = [
        snn.Dense(n_neurons[i], n_neurons[i + 1], activation=activation)
        for i in range(n_layers - 1)
    ]
    # assign a Dense layer (without activation function) to the output layer
    # 为输出层分配一个没有激活函数的 Dense 层
    if last_zero_init:
        layers.append(
            snn.Dense(
                n_neurons[-2],
                n_neurons[-1],
                activation=None,
                weight_init=torch.nn.init.zeros_,
                bias=last_bias,
            )
        )
    else:
        layers.append(
            snn.Dense(n_neurons[-2], n_neurons[-1], activation=None, bias=last_bias)
        )
    # put all layers together to make the network
    out_net = nn.Sequential(*layers)  # 将 layers 列表中的所有层组合起来
    return out_net


# 构建一个具有门控等变性的多层感知机
def build_gated_equivariant_mlp(
    n_in: int,
    n_out: int,
    n_hidden: Optional[Union[int, Sequence[int]]] = None,
    n_gating_hidden: Optional[Union[int, Sequence[int]]] = None,
    n_layers: int = 2,
    activation: Callable = F.silu,
    sactivation: Callable = F.silu,
):
    """
    Build neural network analog to MLP with `GatedEquivariantBlock`s instead of dense layers.

    Args:
        n_in: number of input nodes.
        n_out: number of output nodes.
        n_hidden: number hidden layer nodes.
            If an integer, same number of node is used for all hidden layers resulting
            in a rectangular network.
            If None, the number of neurons is divided by two after each layer starting
            n_in resulting in a pyramidal network.
        n_layers: number of layers.
        activation: Activation function for gating function.
        sactivation: Activation function for scalar outputs. All hidden layers would
            the same activation function except the output layer that does not apply
            any activation function.
    """
    # get list of number of nodes in input, hidden & output layers
    # 根据输入参数确定隐藏层和输出层的节点数。
    # 如果 n_hidden 为 None，则节点数会按照一定规律递减，形成一个金字塔形状的网络结构；
    if n_hidden is None:
        c_neurons = n_in
        n_neurons = []
        for i in range(n_layers):
            n_neurons.append(c_neurons)
            c_neurons = max(n_out, c_neurons // 2)
        n_neurons.append(n_out)
    else:
        # 如果 n_hidden 为一个整数，则隐藏层的节点数都相同，形成一个矩形的网络结构
        # get list of number of nodes hidden layers
        if type(n_hidden) is int:
            n_hidden = [n_hidden] * (n_layers - 1)
        else:
            n_hidden = list(n_hidden)
        n_neurons = [n_in] + n_hidden + [n_out]

    # 确定用于门控函数的隐藏层的节点数
    # 如果 n_gating_hidden 为 None，则使用与对应的神经元数量相同的节点数；
    # 如果 n_gating_hidden 是一个整数，则使用相同的节点数；
    # 否则，将 n_gating_hidden 视为列表，并使用列表中的值作为隐藏层的节点数。
    if n_gating_hidden is None:
        n_gating_hidden = n_neurons[:-1]
    elif type(n_gating_hidden) is int:
        n_gating_hidden = [n_gating_hidden] * n_layers
    else:
        n_gating_hidden = list(n_gating_hidden)

    # assign a GatedEquivariantBlock (with activation function) to each hidden layer
    layers = [
        snn.GatedEquivariantBlock(
            n_sin=n_neurons[i],
            n_vin=n_neurons[i],
            n_sout=n_neurons[i + 1],
            n_vout=n_neurons[i + 1],
            n_hidden=n_gating_hidden[i],
            activation=activation,
            sactivation=sactivation,
        )
        for i in range(n_layers - 1)
    ]
    # assign a GatedEquivariantBlock (without scalar activation function)
    # to the output layer
    layers.append(
        snn.GatedEquivariantBlock(
            n_sin=n_neurons[-2],
            n_vin=n_neurons[-2],
            n_sout=n_neurons[-1],
            n_vout=n_neurons[-1],
            n_hidden=n_gating_hidden[-1],
            activation=activation,
            sactivation=None,
        )
    )
    # put all layers together to make the network
    out_net = nn.Sequential(*layers)
    return out_net
