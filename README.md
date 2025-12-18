# QRLM-CUDA: Quantum-Resonance Language Model

这是一个基于 PyTorch 的实验性语言模型，针对 NVIDIA 4070 Ti (Ampere/Ada 架构) 进行了深度优化。

## 主要特性
- **Hamiltonian Layers**: 使用哈密顿矩阵构建层，保持能量守恒。
- **Dynamic Rank Growth**: 训练过程中根据 Loss 停滞情况自动增加秩（Rank）。
- **CUDA TF32 加速**: 针对 30系/40系 显卡开启了 TensorFloat-32 加速。

## 运行环境
- Python 3.8+
- PyTorch 2.0+ (需要 CUDA 支持)
- NVIDIA GPU (推荐 12GB+ 显存)

## 快速开始
1. 安装依赖: `pip install -r requirements.txt`
2. 运行训练: `python main.py`

## 注意事项
首次运行会自动下载 WikiText-2 数据集。

# QRLM-CUDA: Quantum-Resonance Language Model

**An experimental language model architecture featuring Hamiltonian Layers, Dynamic Rank Growth, and CUDA-optimized Wave Structure Banks.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Key Features

*   **Balanced Hamiltonian Layers**: Replaces standard linear layers with a custom `WaveStructureBank` that maintains orthogonality and preserves wave energy.
*   **Dynamic Rank Growth**: The model automatically detects loss stagnation and grows its internal rank (complexity) during training, similar to neural architecture search.
*   **Physics-Inspired Mixing**: Uses a structured Hamiltonian matrix construction for global token mixing.
*   **NVIDIA Ampere/Ada Optimization**:
    *   Full support for **TF32 (TensorFloat-32)** execution.
    *   Custom Mixed Precision (AMP) training loop.
    *   Gradient Accumulation optimized for 16GB VRAM GPUs (e.g., RTX 4070 Ti).

## 🛠️ Architecture

Unlike standard Transformers, QRLM uses:
1.  **WaveStructureBank**: A shared bank of orthogonal components stored on GPU.
2.  **Orthogonality Loss**: An auxiliary loss term (`axiom_lambda`) to enforce component independence.
3.  **Energy Monitoring**: Real-time tracking of wave energy norms throughout the network.

