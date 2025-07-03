# LLama4 MOE2Dense - MoE模型转Dense模型工具

[English](#english) | [中文](#chinese)

## English

### Overview

MOE2Dense is a powerful tool for converting Mixture of Experts (MoE) models to dense models. This tool provides various optimization techniques including pruning and expert merging methods to transform sparse MoE architectures into efficient dense neural networks.

### Features

- **Multiple Pruning Methods**: Structured and unstructured pruning with configurable ratios
- **Expert Merging Strategies**: 
  - Mean merge: Simple averaging of expert weights
  - Utilization-based merge: Weighted merging based on expert utilization
  - Weighted merge: Router-based weighted combination
  - Remove: Complete removal of MoE structure, keeping only shared experts
- **GPU Support**: Automatic device detection and GPU acceleration
- **Flexible Configuration**: Command-line interface with extensive parameter options

### Installation

```bash
pip install torch transformers modelscope
```

### Usage

#### Basic Usage

```bash
python llama4_optimizer.py --model_path /path/to/moe/model --output_path /path/to/output
```

#### Advanced Usage with Pruning

```bash
python llama4_optimizer.py \
    --model_path /path/to/moe/model \
    --output_path /path/to/output \
    --enable_prune \
    --prune_ratio 0.3 \
    --prune_type structured \
    --moe_method mean_merge
```

#### Parameters

- `--model_path`: Path to the original MoE model
- `--output_path`: Output path for the converted model
- `--prune_ratio`: Pruning ratio (0-1, default: 0.3)
- `--prune_type`: Pruning type (`structured` or `unstructured`, default: `structured`)
- `--enable_prune`: Enable pruning operation (default: disabled)
- `--moe_method`: MoE conversion method (`mean_merge`, `utilization_merge`, `weighted_merge`, `remove`, default: `remove`)
- `--dense_size`: Target dense layer size (None to maintain original expert capacity)

### Supported Models

- Llama4 models with MoE architecture
- Compatible with modelscope and transformers libraries

### Examples

#### Convert MoE to Dense with Mean Merge
```bash
python llama4_optimizer.py \
    --model_path ./llama4-moe-model \
    --output_path ./llama4-dense-model \
    --moe_method mean_merge
```

#### Prune and Convert
```bash
python llama4_optimizer.py \
    --model_path ./llama4-moe-model \
    --output_path ./llama4-optimized-model \
    --enable_prune \
    --prune_ratio 0.2 \
    --prune_type unstructured \
    --moe_method utilization_merge
```

### Performance Considerations

- **Memory Usage**: Converting large MoE models may require significant GPU memory
- **Speed**: Dense models are generally faster for inference but may have reduced capacity
- **Quality**: Expert merging may affect model performance; evaluate on your specific tasks

### Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### License

MIT License

---

## Chinese

### 概述

MOE2Dense 是一个强大的工具，用于将专家混合（MoE）模型转换为稠密模型。该工具提供各种优化技术，包括剪枝和专家合并方法，将稀疏的MoE架构转换为高效的稠密神经网络。

### 功能特性

- **多种剪枝方法**: 结构化和非结构化剪枝，可配置剪枝比例
- **专家合并策略**:
  - 均值合并: 专家权重的简单平均
  - 基于利用率的合并: 根据专家利用率进行加权合并
  - 加权合并: 基于路由器的加权组合
  - 移除: 完全移除MoE结构，仅保留共享专家
- **GPU支持**: 自动设备检测和GPU加速
- **灵活配置**: 命令行界面，提供丰富的参数选项

### 安装

```bash
pip install torch transformers modelscope
```

### 使用方法

#### 基本使用

```bash
python llama4_optimizer.py --model_path /path/to/moe/model --output_path /path/to/output
```

#### 高级使用（包含剪枝）

```bash
python llama4_optimizer.py \
    --model_path /path/to/moe/model \
    --output_path /path/to/output \
    --enable_prune \
    --prune_ratio 0.3 \
    --prune_type structured \
    --moe_method mean_merge
```

#### 参数说明

- `--model_path`: 原始MoE模型路径
- `--output_path`: 转换后模型的输出路径
- `--prune_ratio`: 剪枝比例 (0-1，默认: 0.3)
- `--prune_type`: 剪枝类型 (`structured` 或 `unstructured`，默认: `structured`)
- `--enable_prune`: 启用剪枝操作 (默认: 禁用)
- `--moe_method`: MoE转换方法 (`mean_merge`, `utilization_merge`, `weighted_merge`, `remove`，默认: `remove`)
- `--dense_size`: 目标稠密层大小 (None表示保持原始专家容量)

### 支持的模型

- 具有MoE架构的Llama4模型
- 兼容modelscope和transformers库

### 使用示例

#### 使用均值合并将MoE转换为稠密模型
```bash
python llama4_optimizer.py \
    --model_path ./llama4-moe-model \
    --output_path ./llama4-dense-model \
    --moe_method mean_merge
```

#### 剪枝并转换
```bash
python llama4_optimizer.py \
    --model_path ./llama4-moe-model \
    --output_path ./llama4-optimized-model \
    --enable_prune \
    --prune_ratio 0.2 \
    --prune_type unstructured \
    --moe_method utilization_merge
```

### 性能考虑

- **内存使用**: 转换大型MoE模型可能需要大量GPU内存
- **速度**: 稠密模型通常推理速度更快，但容量可能降低
- **质量**: 专家合并可能影响模型性能；请在特定任务上评估

### 贡献

欢迎贡献！请随时提交问题和拉取请求。

### 许可证

MIT许可证
