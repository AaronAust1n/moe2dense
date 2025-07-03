"""
Llama4模型优化脚本 - 剪枝与MOE转换 (PyTorch版本)
版本: 2.0
更新:
1. 添加GPU设备指定功能
2. 实现剪枝和MOE转换具体逻辑
3. 优化错误处理
"""

import torch
import torch.nn as nn
from modelscope import Llama4ForConditionalGeneration, AutoProcessor
import os
import argparse
import warnings
from typing import Optional, Literal
from transformers import AutoModelForCausalLM, AutoConfig

class ModelOptimizer:
    def __init__(self, model_path: str, device: str = None):
        """
        初始化优化器
        Args:
            model_path: 原始模型路径
            device: 运行设备 (cuda/cpu/cuda:0等)
        """
        # 自动检测设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # 加载模型
        print(f"⏳ 从 {model_path} 加载模型到 {self.device}...")
        try:
            self.model = Llama4ForConditionalGeneration.from_pretrained(
                model_path,
                attn_implementation="flash_attention_2",
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            self.config = self.model.config
            self._verify_model()
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            raise RuntimeError("模型加载失败，请检查路径和设备配置")

        
    def _verify_model(self):
        """验证模型完整性"""
        # 检查是否为MoE模型 - 修复配置访问方式
        if hasattr(self.config, 'text_config'):
            text_config = self.config.text_config
            if hasattr(text_config, 'num_local_experts'):
                print(f"🔍 检测到MoE模型: {text_config.num_local_experts}个专家")
            else:
                warnings.warn("⚠️ 非MOE模型，MOE转换可能无效")
        else:
            warnings.warn("⚠️ 无法检测模型配置，MOE转换可能无效")

    def prune(
        self,
        ratio: float = 0.3,
        prune_type: Literal["structured", "unstructured"] = "structured",
        importance_metric: str = "l1"
    ) -> nn.Module:
        """
        执行剪枝操作
        Args:
            ratio: 剪枝比例 (0-1)
            prune_type: 剪枝类型 (structured/unstructured)
            importance_metric: 重要性评估标准 (l1/l2)
        Returns:
            优化后的模型
        """
        if prune_type == "structured":
            return self._structured_prune(ratio, importance_metric)
        else:
            return self._unstructured_prune(ratio)
       
    def _structured_prune(self, ratio: float, metric: str) -> nn.Module:
        """结构化剪枝实现"""
        print(f"🔧 执行结构化剪枝 (ratio={ratio}, metric={metric})")
        
        # 遍历所有线性层进行剪枝
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # 计算通道重要性
                if metric == "l1":
                    importance = module.weight.data.abs().sum(dim=1)
                else:  # l2
                    importance = torch.norm(module.weight.data, p=2, dim=1)
                
                # 确定保留的通道数
                num_keep = int(module.out_features * (1 - ratio))
                if num_keep <= 0:
                    continue
                
                # 选择最重要的通道
                _, keep_indices = torch.topk(importance, num_keep, largest=True)
                
                # 剪枝权重
                pruned_weight = module.weight.data[keep_indices, :]
                module.weight = nn.Parameter(pruned_weight)
                
                # 剪枝偏置
                if module.bias is not None:
                    pruned_bias = module.bias.data[keep_indices]
                    module.bias = nn.Parameter(pruned_bias)
                
                # 更新输出特征数
                module.out_features = num_keep
                print(f"✂️ 结构化剪枝层 {name}: {module.in_features}x{module.out_features}")
        
        return self.model

    def _unstructured_prune(self, ratio: float) -> nn.Module:
        """非结构化剪枝实现"""
        print(f"✂️ 执行非结构化剪枝 (ratio={ratio})")
        
        # 收集所有权重
        all_weights = []
        for name, param in self.model.named_parameters():
            if param.dim() > 1:  # 只处理权重矩阵
                all_weights.append(param.data.view(-1))
        
        # 计算全局阈值
        all_weights = torch.cat(all_weights)
        threshold = torch.quantile(all_weights.abs(), ratio)
        
        # 应用剪枝
        for name, param in self.model.named_parameters():
            if param.dim() > 1:
                mask = param.data.abs() > threshold
                param.data.mul_(mask.float())
                pruned_count = torch.sum(mask == 0).item()
                total_count = param.numel()
                print(f"✂️ 非结构化剪枝层 {name}: 剪枝率 {pruned_count/total_count:.2%}")
        
        return self.model

    def convert_moe(
        self,
        method: Literal["mean_merge", "utilization_merge", "weighted_merge", "remove"] = "mean_merge",
        dense_size: Optional[int] = None
    ) -> nn.Module:
        """
        MOE转换功能
        Args:
            method: 专家合并方法
            dense_size: 目标Dense层大小 (None表示保持原专家总容量)
        Returns:
            转换后的模型
        """
        print(f"🔄 执行MOE转换 (method={method}, dense_size={dense_size})")
        
        if method == "remove":
            return self._remove_moe()
        else:
            return self._merge_experts(method, dense_size)
    
    def _merge_experts(self, method: str, dense_size: int) -> nn.Module:
        """专家合并核心逻辑"""
        if method == "mean_merge":
            print("📊 使用均值合并专家...")
            return self._mean_merge_experts(dense_size)
        elif method == "utilization_merge":
            print("📈 基于利用率加权合并...")
            return self._utilization_merge_experts(dense_size)
        elif method == "weighted_merge":
            print("⚖️ 基于路由权重合并...")
            return self._weighted_merge_experts(dense_size)


    def _mean_merge_experts(self, dense_size: int) -> nn.Module:
        """均值合并专家 (使用实际权重结构)"""
        # 修复配置访问方式
        num_layers = self.config.text_config.num_hidden_layers
        for i in range(num_layers):
            layer = self.model.language_model.model.layers[i]
            
            # 检查MOE结构 (修正为feed_forward)
            if hasattr(layer, 'feed_forward') and hasattr(layer.feed_forward, 'experts'):
                ff = layer.feed_forward
                num_experts = len(ff.experts)
                
                # 平均合并所有专家权重 (使用实际权重名称)
                merged_gate_up = torch.mean(torch.stack(
                    [expert.gate_up_proj.weight.data for expert in ff.experts]), dim=0)
                merged_down = torch.mean(torch.stack(
                    [expert.down_proj.weight.data for expert in ff.experts]), dim=0)
                
                # 创建新的稠密前馈层
                new_ffn = nn.Sequential(
                    nn.Linear(merged_gate_up.shape[1], merged_gate_up.shape[0]),
                    nn.GELU(),
                    nn.Linear(merged_down.shape[1], merged_down.shape[0])
                ).to(self.device)
                
                # 设置权重
                new_ffn[0].weight.data = merged_gate_up
                new_ffn[2].weight.data = merged_down
                
                # 替换原始MOE模块
                setattr(layer, 'feed_forward', new_ffn)
                print(f"🔄 合并{num_experts}个专家 (layer.{i})")
        
        return self.model
    
    def _utilization_merge_experts(self, dense_size: int) -> nn.Module:
        """基于利用率加权合并专家"""
        print("📈 基于专家利用率加权合并...")
        
        # 修复配置访问方式
        num_layers = self.config.text_config.num_hidden_layers
        # 遍历所有MOE层
        for i in range(num_layers):
            layer = self.model.language_model.model.layers[i]
            
            if hasattr(layer, 'block_sparse_moe'):
                moe = layer.block_sparse_moe
                num_experts = moe.num_experts
                
                # 获取路由器权重
                router_weights = moe.gate.weight.data
                
                # 计算每个专家的利用率（路由器权重绝对值之和）
                expert_utilization = torch.sum(torch.abs(router_weights), dim=0)
                
                # 归一化得到权重因子
                weights = expert_utilization / torch.sum(expert_utilization)
                
                # 加权合并专家权重
                merged_w1 = torch.zeros_like(moe.experts[0].w1.weight.data)
                merged_w2 = torch.zeros_like(moe.experts[0].w2.weight.data)
                merged_w3 = torch.zeros_like(moe.experts[0].w3.weight.data)
                
                for idx, expert in enumerate(moe.experts):
                    merged_w1 += weights[idx] * expert.w1.weight.data
                    merged_w2 += weights[idx] * expert.w2.weight.data
                    merged_w3 += weights[idx] * expert.w3.weight.data
                
                # 创建新的稠密前馈层
                new_ffn = nn.Sequential(
                    nn.Linear(merged_w1.shape[1], merged_w1.shape[0]),
                    nn.GELU(),
                    nn.Linear(merged_w3.shape[1], merged_w3.shape[0])
                ).to(self.device)
                
                # 设置权重
                new_ffn[0].weight.data = merged_w1
                new_ffn[2].weight.data = merged_w2
                
                # 替换原始MOE模块
                setattr(layer, 'feed_forward', new_ffn)
                delattr(layer, 'block_sparse_moe')
                print(f"📊 基于利用率合并{num_experts}个专家 (layer.{i})")
        
        return self.model
    
    def _weighted_merge_experts(self, dense_size: int) -> nn.Module:
        """基于路由权重合并专家"""
        print("⚖️ 基于路由权重加权合并...")
        
        # 修复配置访问方式
        num_layers = self.config.text_config.num_hidden_layers
        # 遍历所有MOE层
        for i in range(num_layers):
            layer = self.model.language_model.model.layers[i]
            
            if hasattr(layer, 'block_sparse_moe'):
                moe = layer.block_sparse_moe
                num_experts = moe.num_experts
                
                # 获取路由器权重
                router_weights = moe.gate.weight.data
                
                # 加权合并专家权重
                merged_w1 = torch.zeros_like(moe.experts[0].w1.weight.data)
                merged_w2 = torch.zeros_like(moe.experts[0].w2.weight.data)
                merged_w3 = torch.zeros_like(moe.experts[0].w3.weight.data)
                
                for idx, expert in enumerate(moe.experts):
                    # 使用路由器权重作为加权因子
                    weight_factor = router_weights[:, idx].unsqueeze(1)
                    merged_w1 += torch.mean(weight_factor) * expert.w1.weight.data
                    merged_w2 += torch.mean(weight_factor) * expert.w2.weight.data
                    merged_w3 += torch.mean(weight_factor) * expert.w3.weight.data
                
                # 创建新的稠密前馈层
                new_ffn = nn.Sequential(
                    nn.Linear(merged_w1.shape[1], merged_w1.shape[0]),
                    nn.GELU(),
                    nn.Linear(merged_w3.shape[1], merged_w3.shape[0])
                ).to(self.device)
                
                # 设置权重
                new_ffn[0].weight.data = merged_w1
                new_ffn[2].weight.data = merged_w2
                
                # 替换原始MOE模块
                setattr(layer, 'feed_forward', new_ffn)
                delattr(layer, 'block_sparse_moe')
                print(f"⚖️ 基于路由权重合并{num_experts}个专家 (layer.{i})")
        
        return self.model

    def _remove_moe(self) -> nn.Module:
        """完全移除MOE结构，仅保留共享专家"""
        print("🚮 移除MOE结构，仅保留共享专家...")
        
        class SimpleFFN(nn.Module):
            def __init__(self, shared_expert, device):
                super().__init__()
                self.gate_proj = nn.Linear(
                    shared_expert.gate_proj.weight.shape[1],
                    shared_expert.gate_proj.weight.shape[0],
                    bias=shared_expert.gate_proj.bias is not None
                ).to(device)
                self.up_proj = nn.Linear(
                    shared_expert.up_proj.weight.shape[1],
                    shared_expert.up_proj.weight.shape[0],
                    bias=shared_expert.up_proj.bias is not None
                ).to(device)
                self.down_proj = nn.Linear(
                    shared_expert.down_proj.weight.shape[1],
                    shared_expert.down_proj.weight.shape[0],
                    bias=shared_expert.down_proj.bias is not None
                ).to(device)
                # 复制权重
                self.gate_proj.weight.data = shared_expert.gate_proj.weight.data.clone()
                self.up_proj.weight.data = shared_expert.up_proj.weight.data.clone()
                self.down_proj.weight.data = shared_expert.down_proj.weight.data.clone()
                # 复制bias（如果有）
                if shared_expert.gate_proj.bias is not None:
                    self.gate_proj.bias.data = shared_expert.gate_proj.bias.data.clone()
                if shared_expert.up_proj.bias is not None:
                    self.up_proj.bias.data = shared_expert.up_proj.bias.data.clone()
                if shared_expert.down_proj.bias is not None:
                    self.down_proj.bias.data = shared_expert.down_proj.bias.data.clone()
            def forward(self, x):
                return self.down_proj(
                    torch.silu(self.gate_proj(x)) * self.up_proj(x)
                )
        
        # 修复配置访问方式
        num_layers = self.config.text_config.num_hidden_layers
        for i in range(num_layers):
            layer = self.model.language_model.model.layers[i]
            
            # 检查是否存在MOE结构 (修正为feed_forward)
            if hasattr(layer, 'feed_forward') and hasattr(layer.feed_forward, 'shared_expert'):
                ff = layer.feed_forward
                shared_expert = ff.shared_expert
                
                # 打印shared_expert权重和bias的shape
                print(f"Layer {i} shared_expert.gate_proj.weight.shape: {shared_expert.gate_proj.weight.shape}")
                print(f"Layer {i} shared_expert.up_proj.weight.shape: {shared_expert.up_proj.weight.shape}")
                print(f"Layer {i} shared_expert.down_proj.weight.shape: {shared_expert.down_proj.weight.shape}")
                if shared_expert.gate_proj.bias is not None:
                    print(f"Layer {i} shared_expert.gate_proj.bias.shape: {shared_expert.gate_proj.bias.shape}")
                if shared_expert.up_proj.bias is not None:
                    print(f"Layer {i} shared_expert.up_proj.bias.shape: {shared_expert.up_proj.bias.shape}")
                if shared_expert.down_proj.bias is not None:
                    print(f"Layer {i} shared_expert.down_proj.bias.shape: {shared_expert.down_proj.bias.shape}")
                
                # 新建FFN
                new_ffn = SimpleFFN(shared_expert, self.device)
                # 打印新建FFN各层的shape
                print(f"Layer {i} new_ffn.gate_proj.weight.shape: {new_ffn.gate_proj.weight.shape}")
                print(f"Layer {i} new_ffn.up_proj.weight.shape: {new_ffn.up_proj.weight.shape}")
                print(f"Layer {i} new_ffn.down_proj.weight.shape: {new_ffn.down_proj.weight.shape}")
                if new_ffn.gate_proj.bias is not None:
                    print(f"Layer {i} new_ffn.gate_proj.bias.shape: {new_ffn.gate_proj.bias.shape}")
                if new_ffn.up_proj.bias is not None:
                    print(f"Layer {i} new_ffn.up_proj.bias.shape: {new_ffn.up_proj.bias.shape}")
                if new_ffn.down_proj.bias is not None:
                    print(f"Layer {i} new_ffn.down_proj.bias.shape: {new_ffn.down_proj.bias.shape}")
                # 移除路由器权重 (如果存在) - 在替换之前删除
                if hasattr(ff, 'router'):
                    print(f"✂️ 移除路由器层 layer.{i}.router")
                
                # 替换原始MOE模块
                setattr(layer, 'feed_forward', new_ffn)
                
                print(f"✂️ 移除MOE层 layer.{i}，替换为稠密前馈层")
        
        return self.model

    
    def save(self, output_path: str):
        """保存优化后模型"""
        self.model.save_pretrained(output_path)
        print(f"💾 模型已保存至 {output_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Llama4模型优化工具 - 支持剪枝和MOE转换")
    parser.add_argument("--model_path", type=str, required=True, help="原始模型路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出路径")
    parser.add_argument("--prune_ratio", type=float, default=0.3, help="剪枝比例 (0-1，仅在启用剪枝时有效)")
    parser.add_argument("--prune_type", choices=["structured", "unstructured"], default="structured", help="剪枝类型 (仅在启用剪枝时有效)")
    parser.add_argument("--enable_prune", action="store_true", help="启用剪枝操作 (默认不剪枝)")
    parser.add_argument("--moe_method", choices=["mean_merge", "utilization_merge", "weighted_merge", "remove"], default="remove", help="MOE转换方法")
    parser.add_argument("--dense_size", type=int, default=None, help="目标Dense层大小 (None表示保持原专家总容量)")
    
    args = parser.parse_args()
    
    try:
        optimizer = ModelOptimizer(args.model_path)
        
        # 根据参数决定是否执行剪枝
        if args.enable_prune:
            print(f"🔧 执行剪枝操作 (ratio={args.prune_ratio}, type={args.prune_type})")
            optimizer.prune(ratio=args.prune_ratio, prune_type=args.prune_type)
        else:
            print("⏭️ 跳过剪枝操作")
        
        # 执行MOE转换
        optimizer.convert_moe(method=args.moe_method, dense_size=args.dense_size)
        optimizer.save(args.output_path)
        print("🎉 优化完成!")
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        warnings.warn("优化过程中出现错误，请检查模型和参数")
