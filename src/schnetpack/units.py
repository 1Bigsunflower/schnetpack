import re
from typing import Union, Dict

from ase import units as aseunits
from ase.units import Units
import numpy as np

__all__ = ["convert_units"]

# Internal units (MD internal -> ASE internal)
__md_base_units__ = {
    "energy": "kJ / mol",  # 能量
    "length": "nm",  # 长度
    "mass": 1.0,  # 1 Dalton in ASE reference frame 质量
    "charge": 1.0,  # Electron charge 电荷
}


def setup_md_units(md_base_units: Dict[str, Union[str, float]]):
    # 用于设置分子动力学中使用的单位
    """
    Define the units used in molecular dynamics. This is done based on the base units for energy, length and mass
    from which all other quantities are derived.

    Args:
        md_base_units (dict): Dictionary defining the basic units for molecular dynamics simulations

    Returns:
        dict(str, float):
    """
    # Initialize basic unit system
    # 将md_base_units字典中的每个单位进行解析和初始化，并更新原字典
    md_base_units = {u: _parse_unit(md_base_units[u]) for u in md_base_units}

    # Set up unit dictionary
    # 用units存储单位信息
    units = Units(md_base_units)
    # 计算出其他相关量的派生单位，如时间、力、应力、压力等。
    # Derived units (MD internal -> ASE internal)
    units["time"] = units["length"] * np.sqrt(units["mass"] / units["energy"])
    units["force"] = units["energy"] / units["length"]
    units["stress"] = units["energy"] / units["length"] ** 3
    units["pressure"] = units["stress"]

    # Conversion of length units
    # 将输入的长度单位与ASE库中定义的长度单位进行比例缩放，从而将其转换为统一的单位表示
    units["A"] = aseunits.Angstrom / units["length"]
    units["Ang"] = units["A"]
    units["Angs"] = units["A"]
    units["Angstrom"] = units["A"]
    units["nm"] = aseunits.nm / units["length"]
    units["a0"] = aseunits.Bohr / units["length"]
    units["Bohr"] = units["a0"]

    # Conversion of energy units
    # 能量单位的转换
    units["kcal"] = aseunits.kcal / units["energy"]
    units["kJ"] = aseunits.kJ / units["energy"]
    units["eV"] = aseunits.eV / units["energy"]
    units["Hartree"] = aseunits.Hartree / units["energy"]
    units["Ha"] = units["Hartree"]

    # Time units
    units["fs"] = aseunits.fs / units["time"]
    units["s"] = aseunits.s / units["time"]
    units["aut"] = aseunits._aut * aseunits.s / units["time"]

    # Pressure units
    units["Pascal"] = aseunits.Pascal / units["pressure"]
    units["bar"] = 1e5 * units["Pascal"]

    # Mol
    units["mol"] = aseunits.mol

    # Mass
    units["Dalton"] = 1.0 / units["mass"]
    units["amu"] = aseunits._amu / units["mass"]

    # Charge distributions
    units["Debye"] = aseunits.Debye / (units["charge"] * units["length"])
    units["C"] = aseunits.C / units["charge"]

    # Constants (internal frame)
    units["kB"] = aseunits.kB / units["energy"]  # Always uses Kelvin
    units["hbar"] = (
            aseunits._hbar * (aseunits.J * aseunits.s) / (units["energy"] * units["time"])
    )  # hbar is given in J*s by ASE
    units["ke"] = (
            units["a0"] * units["Ha"] / units["charge"] ** 2
    )  # Coulomb constant is 1 in atomic units

    # For spectra
    units["hbar2icm"] = units["hbar"] * 100.0 * aseunits._c * aseunits._aut
    # 返回所有单位的转换结果
    return units


# Placeholders for expected unit entries
# 定义34个变量作为单位的占位符。这些占位符用于存储不同单位的转换结果。每个占位符都被初始化为0.0。
(
    energy,
    length,
    mass,
    charge,
    time,
    force,
    stress,
    pressure,
    kB,
    hbar,
    hbar2icm,
    A,
    Ang,
    Angs,
    Angstrom,
    nm,
    a0,
    Bohr,
    kcal,
    kJ,
    eV,
    Hartree,
    Ha,
    fs,
    s,
    aut,
    mol,
    Dalton,
    amu,
    Debye,
    C,
    ke,
    bar,
    Pascal,
) = [0.0] * 34


def _conversion_factor_ase(unit: str):
    # 根据给定的单位字符串获取ASE库中对应的单位
    """Get units by string and convert to ase unit system."""
    if unit == "A":
        raise Warning(
            "The unit string 'A' specifies Ampere. For Angstrom, please use 'Ang' or 'Angstrom'."
        )
    return getattr(aseunits, unit)


def _conversion_factor_internal(unit: str):
    # 根据给定的单位字符串获取内部单位系统中对应的单位
    """Get units by string and convert to internal unit system."""
    return globals()[unit]


def _parse_unit(unit, conversion_factor=_conversion_factor_ase):
    # 解析单位字符串并将其转换为相应的转换因子。unit表示待解析的单位字符串，conversion_factor表示转换因子函数
    if type(unit) == str:
        # If a string is given, split into parts.
        # 将单位字符串拆分为多个部分进行解析。按照非单词字符进行拆分，以保留单位中的运算符（如分数）。如2 km/h会被拆分为['2', ' ', 'km', '/', 'h']
        parts = re.split("(\W)", unit)

        conversion = 1.0
        divide = False  # 设定一个标记判断是否划分
        for part in parts:
            if part == "/":
                # 表示下一个拆分部分需要与之前的结果进行除法运算。
                divide = True
            elif part == "" or part == " ":
                # 检查当前部分是否为空字符串或空格。如果是，则跳过当前迭代
                pass
            else:
                # 如果当前部分既不是除号也不为空
                # 将当前部分作为参数传递给该函数，以获取相应的转换因子，并存入p中
                p = conversion_factor(part)
                if divide:
                    # 根据divide的值进行相应的乘除运算。如果divide为True，即前面遇到了除号，此时需要执行除法运算，将conversion除以转换因子p，并将divide设置为False。
                    conversion /= p
                    divide = False
                else:
                    # 如果没有遇到除号，直接将转换因子p乘以conversion。
                    conversion *= p
        # 返回计算得到的转换因子
        return conversion
    else:
        # If unit is given as number, return number
        return unit


def unit2internal(src_unit: Union[str, float]):
    # 将单位转换为上面定义的内部单位系统
    """
    Convert unit to internal unit system defined above.

    Args:
        src_unit (str, float): Name of unit

    Returns:
        float: conversion factor from external to internal unit system.
    """
    return _parse_unit(src_unit, conversion_factor=_conversion_factor_internal)


def convert_units(src_unit: Union[str, float], tgt_unit: Union[str, float]):
    """Return conversion factor for given units"""
    # 返回给定单位之间的转换因子
    return _parse_unit(src_unit) / _parse_unit(tgt_unit)


# 更新全局命名空间中的变量，将字典中的所有键值添加到全局命名空间中
globals().update(setup_md_units(__md_base_units__))
