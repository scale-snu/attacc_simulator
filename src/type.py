from enum import Enum


class DataType(Enum):
    W16A16 = 0
    W16A8 = 1
    W8A16 = 2
    W8A8 = 3


class LayerType(Enum):
    FC = 0
    MATMUL = 1
    ACT = 2
    SOFTMAX = 3
    NORM = 4
    G2G = 5
    X2G = 6


class DeviceType(Enum):
    NONE = 0
    GPU = 1
    CPU = 2
    PIM = 3


class PIMType(Enum):
    BA = 0
    BG = 1
    BUFFER = 2


class InterfaceType(Enum):
    NVLINK4 = 0
    NVLINK3 = 1
    PCIE4 = 2
    PCIE5 = 3


class GPUType(Enum):
    A100a = 0
    H100 = 1
