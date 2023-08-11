import ase
from ase import Atoms
from ase.io import write
import torch
import copy
import argparse

Z = [6, 6, 1, 1, 1, 1, 1, 8, 1]  # Z 是一个包含原子的原子序数（碳的原子序数是6，氢的原子序数是1，氧的原子序数是8）的列表
R = [  # 对于每个原子，使用一个三元组 (x, y, z) 表示其在三维空间中的位置。
    [-4.91832096, 1.53666755, -0.06624112],
    [-3.41563316, 1.44992331, -0.14155274],
    [-5.22246355, 2.28682457, 0.66707348],
    [-5.34200431, 0.57373599, 0.22595757],
    [-5.3338372, 1.81223882, -1.03759683],
    [-3.00608229, 1.18685677, 0.84392479],
    [-2.99789458, 2.42727565, -0.42151118],
    [-3.0796312, 0.46731581, -1.10256879],
    [-2.12365275, 0.40566152, -1.15678517],
]
# 根据Z和R变量，可以确定分子的组成和几何结构
from schnetpack.interfaces import SpkCalculator, AseInterface

if __name__ == "__main__":
    # 接收模型文件的路径
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="Path to model file.")
    args = parser.parse_args()

    # 根据Z和R创建一个分子系统
    atoms = Atoms(Z, R)

    # 加载模型
    model = torch.jit.load(args.model, map_location="cuda")

    # Initialize the calculator
    calc = SpkCalculator(model, 5.0)

    # Perform the calculation 计算分子的能量和力
    calc.calculate(atoms, properties=["energy", "forces"])

    print(calc.results)
    print(" I now INIT THE MODEL")

    # 将分子结构保存到文件 "test-inpu.xyz" 中，使用 X Y Z 格式保存
    write("test-inpu.xyz", atoms, format="xyz")
    # 传递分子结构文件、工作目录、加载的模型以及截断半径（5.0）作为参数。precision="float32" 指定使用单精度浮点数进行计算。
    aseinf = AseInterface("test-inpu.xyz", "workdir", model, 5.0, precision="float32")

    # 使用加载的模型在给定的分子结构上计算并返回该分子的能量值
    aseinf.calculate_single_point()
    # 优化分子结构。它将尝试调整分子中原子的位置，以使能量达到局部或全局最小值。优化过程会迭代多次，直到满足停止条件或达到最大迭代次数。
    aseinf.optimize()
    # 将优化后的分子结构保存到文件 "TEST_SAVE" 中
    aseinf.save_molecule("TEST_SAVE")
    # 计算分子中原子在不同振动频率下的位移和相应的振动频率
    aseinf.compute_normal_modes()
    # 初始化分子动力学模拟并运行100步的分子动力学模拟，通过解决牛顿方程来模拟原子之间的相互作用和运动
    aseinf.init_md("TEST_MD")
    aseinf.run_md(100)
