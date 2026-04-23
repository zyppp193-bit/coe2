import torch
import time
import torch.nn as nn
import sys
import os
import logging as lg
import random as r
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
import time
import torch.nn as nn
import sys
import os
import logging as lg
import random as r
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchvision
import torch.cuda.amp as amp
import random
import wandb
import matplotlib.pyplot as plt
import math
import timm
from safetensors.torch import load_file
from copy import deepcopy
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE

from src.learners.base import BaseLearner
from src.learners.baselines.er import ERLearner
from src.utils.losses import WKDLoss
from src.models.resnet import ResNet18
from src.utils import name_match
from src.utils.metrics import forgetting_line
from src.utils.utils import get_device, filter_labels
from src.utils.augment import MixupAdaptative, ZetaMixup

from typing import Any, Dict, Optional

device = get_device()
scaler = amp.GradScaler()

LR_MIN = 5e-4
LR_MAX = 5e-2


class PromptedViT(nn.Module):
    """
    包装 timm 的 ViT，仅在输入层加入可学习的 Visual Prompts (VPT-Shallow)。
    原有的 Backbone 参数将被冻结。
    """
    def __init__(self, vit_model, num_prompts=5, prompt_dim=768, num_classes=100):
        super().__init__()
        self.backbone = vit_model
        self.num_prompts = num_prompts
        
        # 1. 定义可学习的 Prompts [1, N, D]
        self.prompts = nn.Parameter(torch.zeros(1, num_prompts, prompt_dim))
        nn.init.xavier_uniform_(self.prompts)
        
        # 2. 冻结 Backbone 的所有参数
        for p in self.backbone.parameters():
            p.requires_grad = False
            
        # 3. 临时分类头，用于给 Prompts 提供训练信号
        self.head = nn.Linear(prompt_dim, num_classes)

    def forward_features(self, x):
        # 1. Patch Embedding
        x = self.backbone.patch_embed(x)
        
        # 2. Add CLS token + Positional Embedding (timm 内部处理)
        # 注意：不同 timm 版本 _pos_embed 实现略有不同，但通常包含 pos_drop
        x = self.backbone._pos_embed(x)
        
        # 3. 插入 Prompts
        # 将 Prompts 扩展到 Batch 维度 [B, N_prompts, D]
        prompts = self.prompts.expand(x.shape[0], -1, -1)
        
        # 将 Prompts 拼接到序列中。
        # 结构变为: [CLS, Prompts, Patches] 或者 [CLS, Patches, Prompts]
        # 这里我们把 Prompts 放在 CLS 之后，Patches 之前
        x = torch.cat((x[:, :1, :], prompts, x[:, 1:, :]), dim=1)
        
        # 4. Pass through Blocks
        if hasattr(self.backbone, 'norm_pre'):
            x = self.backbone.norm_pre(x)
            
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        
        # 5. 提取特征
        # index 0 依然是 CLS token，因为它在拼接时被保留在最前
        return x[:, 0]

    def forward(self, x):
        feat = self.forward_features(x)
        # 使用临时分类头计算 logits，用于训练 Prompt
        logits = self.head(feat)
        return logits, feat


class SampleMoEGate(nn.Module):
    """
    一个简单的按样本 MoE gate。

    输入:
        hidden: [B, H]  每个样本的一行向量(例如学生 logits 或特征)
    输出:
        topk_idx:    [B, top_k]   每个样本选中的专家编号
        topk_weight: [B, top_k]   对应的权重(已在 top-k 内归一化)
        aux_loss:    负载均衡损失(可选, 可能为 None)
    """

    def __init__(
        self,
        in_dim: int,
        n_experts: int,
        top_k: int,
        aux_loss_alpha: float = 0.0,
        noise_std: float = 1.0,
    ):
        super().__init__()
        self.in_dim = in_dim              # 输入向量长度 H
        self.n_experts = n_experts        # 专家数
        self.top_k = top_k                # 每个样本选多少个专家
        self.alpha = aux_loss_alpha       # 负载均衡损失权重
        self.noise_std = noise_std        # logits 上的噪声强度

        # gating 线性层: [H] -> [n_experts]
        # learnable matrix: [n_experts, in_dim]
        self.weight = nn.Parameter(torch.empty(n_experts, in_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def forward(self, hidden: torch.Tensor):
        """
        hidden: [B, H]
        """
        B, H = hidden.shape

        # 1. 线性打分 -> logits: [B, n_experts]
        logits = F.linear(hidden, self.weight, None)  # hidden @ weight.T

        # 1b. 训练阶段加入噪声，鼓励均匀探索
        if self.training and self.noise_std > 0:
            logits = logits + self.noise_std * torch.randn_like(logits)

        # 2. softmax 变成概率分布
        scores = logits.softmax(dim=-1)  # 每一行加起来 = 1

        # 3. Top-k 专家选择
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 4. 在 top-k 内重新归一化权重
        denom = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denom

        # 5. 辅助负载均衡损失
        aux_loss: Optional[torch.Tensor] = None
        if self.alpha > 0.0 and self.training:
            # mask: [B * top_k, n_experts] 的 one-hot
            mask = F.one_hot(topk_idx.view(-1), num_classes=self.n_experts).float()
            # f_i: 每个专家被选中的频率
            f_i = mask.mean(dim=0)
            # P_i: gate 给每个专家的平均概率
            P_i = scores.mean(dim=0)
            # Switch Transformer 风格的负载均衡损失：alpha * N * sum(f_i * P_i)
            aux_loss = self.alpha * self.n_experts * torch.sum(f_i * P_i)

        return topk_idx, topk_weight, aux_loss


class StudentFeatureGate(nn.Module):
    """
    基于学生模型中间特征的门控：LN + MLP -> n_experts

    - 输入: hidden [B, H]，来自学生的特征（建议 detach）
    - 输出: 与 SampleMoEGate 相同 (topk_idx, topk_weight, aux_loss)
    - 特点: 语义贴近当前任务，计算轻量，适合 OCL
    """

    def __init__(
        self,
        in_dim: int,
        n_experts: int,
        top_k: int,
        aux_loss_alpha: float = 0.01,
        noise_std: float = 1.0,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.alpha = aux_loss_alpha
        self.noise_std = noise_std
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_experts),
        )

    def forward(self, hidden: torch.Tensor):
        # 1) MLP 打分
        logits = self.net(hidden)  # [B, n_experts]
        # 2) 训练期噪声，鼓励探索
        if self.training and self.noise_std > 0:
            logits = logits + self.noise_std * torch.randn_like(logits)
        # 3) softmax 概率
        scores = logits.softmax(dim=-1)
        # 4) Top-k 选择与归一化
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        denom = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denom
        # 5) 负载均衡损失
        aux_loss: Optional[torch.Tensor] = None
        if self.alpha > 0.0 and self.training:
            mask = F.one_hot(topk_idx.view(-1), num_classes=self.n_experts).float()
            f_i = mask.mean(dim=0)
            P_i = scores.mean(dim=0)
            aux_loss = self.alpha * self.n_experts * torch.sum(f_i * P_i)
        return topk_idx, topk_weight, aux_loss


class ER_EMA_MoELearner(ERLearner):
    """
    ER_EMA_MoELearner (Enhanced with Pretrained ViT Gating & Ensemble Distillation)

    主要机制:
      1. 路由专家 (Routed Experts):
         - 维护一组通过 EMA 更新的教师模型(学生模型的历史副本)。
      2. 预训练 ViT 门控 (Pretrained ViT Gating):
         - 冻结的 ViT (ImageNet-21k) 提取图像特征作为 gate 输入。
      3. MoE 蒸馏:
         - 每个样本由 gate 选择 top-k 个教师专家进行加权蒸馏。
      4. 动态扩展 (Dynamic Expansion):
         - 新任务到来时同步扩展学生和所有教师的分类头。
    """

    def __init__(self, args: Any):
        super().__init__(args)

        # 蒸馏损失
        self.wkdloss = WKDLoss(
            temperature=self.params.kd_temperature,
            use_wandb=not getattr(self.params, "no_wandb", False),
            alpha_kd=self.params.alpha_kd,
        )
        self.classes_seen_so_far = torch.LongTensor(size=(0,)).to(device)
        # 新增：用于 MOSE 风格的平衡采样，记录所有见过的类别
        self.seen_classes = set()

        # ---------------------- 超参数 ----------------------
        self.n_routed_experts = getattr(self.params, "n_routed_experts", 4) #路由教师个数
        #self.n_shared_experts = getattr(self.params, "n_shared_experts", 1)  # 保留接口, 实际不使用
        self.top_k = getattr(self.params, "top_k_experts",1) #选择的教师个数
        self.ema_alpha = getattr(self.params, "ema_alpha", 0.99)
        self.n_classes = getattr(self.params, "n_classes", None)
        self.gamma = getattr(self.params, "gamma", 0.01) #控制 EMA 更新速度
        self.gamma_unselected = getattr(self.params, "gamma_unselected", 0)  # 未选中专家的慢速 EMA 系数
        # 统计每个专家被选中的总次数
        self.expert_select_counts = torch.zeros(self.n_routed_experts, dtype=torch.long)

        if self.n_classes is None:
            raise ValueError("请在 params 中设置 n_classes (模型输出类别数).")

        # ---------------------- 路由专家 (EMA 教师) ----------------------
        # 初始化时, 每个专家都是学生模型的深拷贝
        self.routed_teachers = nn.ModuleList(
             [deepcopy(self.model) for _ in range(self.n_routed_experts)]
        )
        # ---------------------- 预训练 ViT 作为门控特征提取器 ----------------------
        print("Loading pretrained ViT for MoE Gating...")
        try:
            self.pretrained_backbone = timm.create_model(
                "vit_base_patch16_224.augreg_in21k",
                pretrained=False,
                num_classes=0,
            )
        except Exception:
            try:
                self.pretrained_backbone = timm.create_model(
                    "vit_base_patch16_224_in21k",
                    pretrained=False,
                    num_classes=0,
                )
            except Exception as e:
                raise ValueError(f"Failed to create ViT model: {e}")

        # safetensors 权重路径
        ckpt_path = "/home/ubuntu/Desktop/mkd_ocl-main/pt_m/modelvit.safetensors"
        try:
            if os.path.exists(ckpt_path):
                state_dict = load_file(ckpt_path)
                msg = self.pretrained_backbone.load_state_dict(state_dict, strict=False)
                print(f"Loaded pretrained weights from {ckpt_path} {msg}")
            else:
                print(f"Warning: Checkpoint not found at {ckpt_path}. Using random init.")
        except Exception as e:
            print(f"Error loading safetensors from {ckpt_path}: {e}")
            print("Warning: Using random initialization for Gate backbone!")

        # 包装 ViT 模型
        self.pretrained_backbone = PromptedViT(
            self.pretrained_backbone,
            num_prompts=getattr(self.params, "num_prompts", 5), # 默认5个prompt
            prompt_dim=self.pretrained_backbone.embed_dim,
            num_classes=self.n_classes
        ).to(device)

        # 确保 Backbone 冻结，但 Prompts 和 Head 开启梯度
        self.pretrained_backbone.eval() # 保持 eval 模式 (关闭 Dropout/BatchNorm 更新)
        self.pretrained_backbone.prompts.requires_grad = True
        self.pretrained_backbone.head.weight.requires_grad = True
        self.pretrained_backbone.head.bias.requires_grad = True

        # 定义 Prompt 优化器 (学习率通常比主模型大一点)
        self.prompt_optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.pretrained_backbone.parameters()),
            lr=getattr(self.params, "prompt_lr", 1e-2), 
            weight_decay=0.0
        )

        self.pretrained_backbone.to(device)

        # ViT Base 通常 embed_dim = 768
        self.feat_dim = self.pretrained_backbone.backbone.embed_dim

        # ViT 需要 224x224 输入
        self.vit_transform = torchvision.transforms.Resize((224, 224))

        # -------- 学生特征 -> ViT 空间 的 Projector --------
        with torch.no_grad():
            dummy_input = torch.randn(2, 3, 32, 32).to(device)
            if hasattr(self.model, "features"):
                stu_feat_dim = self.model.features(dummy_input).shape[1]
            else:
                # 兜底: ResNet18 一般为 512
                stu_feat_dim = 512
        # -------- 对齐维度 --------
        self.feature_projector = nn.Sequential(
            nn.Linear(stu_feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, self.feat_dim),
        ).to(device)

        # -------- 学生优化器：只优化 student --------
        self.optim = torch.optim.Adam(
            list(self.model.parameters()),
            lr=self.params.learning_rate,
            weight_decay=self.params.weight_decay,
        )
        # -------- Projector 优化器：只优化 projector --------
        proj_lr = getattr(self.params, "proj_lr", self.params.learning_rate)  # 没配就先跟学生同 lr
        proj_wd = getattr(self.params, "proj_weight_decay", self.params.weight_decay)
        self.proj_optim = torch.optim.Adam(
            self.feature_projector.parameters(),
            lr=proj_lr,
            weight_decay=proj_wd,
        )

        # MoE Gate
        gate_noise_std = getattr(self.params, "gate_noise_std", 1.0)
        # 使用学生特征维度作为门控输入维度
        '''根据代码逻辑，学生模型用于门控和特征蒸馏的中间层是 
        全局平均池化层 (Global Average Pooling) 之后，
        分类头 (FC Layer) 之前 的特征向量。'''
        
        self.gate = StudentFeatureGate(
            in_dim=stu_feat_dim,
            n_experts=self.n_routed_experts,
            top_k=self.top_k,
            aux_loss_alpha=getattr(self.params, "aux_loss_alpha", 0.01),
            noise_std=gate_noise_std,
            hidden_dim=getattr(self.params, "gate_hidden_dim", 512),
        ).to(device)
        
        #学习率
        gate_lr = getattr(self.params, "gate_lr", 1e-4)
        #优化器
        self.gate_optim = torch.optim.Adam(self.gate.parameters(), lr=gate_lr)

        print(
            f"[MoE EMA] routed_experts={self.n_routed_experts}, "
            f"top_k={self.top_k}, ema_alpha={self.ema_alpha}"
        )

        # 初始化: 所有路由专家参数先拷贝自学生
        self.update_ema_all(init=True)

        # 漂移测量(可选, 按你原逻辑)
        self.previous_model = None
        if getattr(self.params, "measure_drift", 0) > 0:
            self.drift = []
            self.previous_model = None

    # ---------------------- EMA 更新相关函数 ----------------------
    def _ema_update_one(self, ema_model: nn.Module, src_model: nn.Module, alpha: float, init: bool = False):
        """
        对单个教师模型做 EMA 更新。
        ema_model: 要更新的老师
        src_model: 参考模型(学生)
        alpha:     EMA 系数
        init:      是否为初始化(直接拷贝)
        """
        # TIES 参数: 只保留变化最大的 20% 参数
        trim_ratio = 0.2

        for ema_p, p in zip(ema_model.parameters(), src_model.parameters()):
            if init:
                ema_p.data.copy_(p.data)
            else:
                # 原始 EMA: ema_new = alpha * ema + (1-alpha) * src
                # 等价于: ema_new = ema + (1-alpha) * (src - ema)
                
                # 1. 计算更新量 delta
                delta = p.data - ema_p.data
                
                # === TIES Trim (极速版: Strided Sampling) ===
                # 全量 kthvalue 即使在 GPU 上也较慢 (涉及排序/选择)
                # 使用步长采样 (Strided Sampling) 估计阈值，速度极快且精度足够
                numel = delta.numel()
                if numel > 0:
                    # 预计算 abs，避免重复计算
                    abs_delta = delta.abs()
                    
                    # 设定采样点上限，例如 2000 点
                    # 对于大张量，只看一部分点来确定分布阈值
                    sample_max = 2000
                    
                    if numel <= sample_max:
                        # 小张量：直接精确计算
                        k = int(numel * trim_ratio)
                        if k > 0:
                            threshold = abs_delta.flatten().kthvalue(numel - k + 1).values
                            delta.masked_fill_(abs_delta < threshold, 0)
                    else:
                        # 大张量：步长采样
                        # [::stride] 创建视图，无额外内存开销，速度极快
                        stride = numel // sample_max
                        sampled_abs = abs_delta.flatten()[::stride]
                        
                        k_sample = int(sampled_abs.numel() * trim_ratio)
                        if k_sample > 0:
                            # 在小样本上计算阈值
                            threshold = sampled_abs.kthvalue(sampled_abs.numel() - k_sample + 1).values
                            # 应用到全量
                            delta.masked_fill_(abs_delta < threshold, 0)
                
                # 应用净化后的更新 (使用 add_ 的 alpha 参数避免一次乘法)
                ema_p.data.add_(delta, alpha=(1.0 - alpha))

    def update_ema_all(self, init: bool = False, routed_mask: Optional[torch.Tensor] = None):
        """
        对所有路由专家模型做 EMA 更新。

        init=True: 所有路由专家直接复制学生参数。
        init=False: 只更新 routed_mask[idx] == True 的专家。
        """
        alpha_routed = 1.0 - self.gamma
        alpha_unselected = None
        if self.gamma_unselected > 0.0:
            alpha_unselected = 1.0 - self.gamma_unselected
        
        #初始化
        if init:
            for ema_model in self.routed_teachers:
                self._ema_update_one(ema_model, self.model, alpha=1.0, init=True)
        
        else:
            if routed_mask is None:
                #没有被选中教师的信息，就都更新
                for ema_model in self.routed_teachers:
                    self._ema_update_one(ema_model, self.model, alpha=alpha_routed, init=False)
            else:
                #拥有被选中教师的信息，按需更新
                for idx, ema_model in enumerate(self.routed_teachers):
                    is_selected = bool(routed_mask[idx])
                    if is_selected:
                        # 被选中的教师: 正常 EMA（较快）
                        self._ema_update_one(ema_model, self.model, alpha=alpha_routed, init=False)
                    elif alpha_unselected is not None:
                        # 未选中的教师: 慢速 EMA（防止遗忘过快，又保持一定跟随）
                        self._ema_update_one(ema_model, self.model, alpha=alpha_unselected, init=False)
    # ---------------------- 训练主循环 ----------------------
    def train(self, dataloader, **kwargs):
        task_name = kwargs.get("task_name", "Unknown")
        task_id = kwargs.get('task_id', None)
        self.model = self.model.train()
        amp_enabled = device.type == "cuda"
        
        # MOSE 风格：记录当前任务的类别
        task_classes = set()

        for j, batch in enumerate(dataloader):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                batch_x, batch_y = batch[0], batch[1]
                self.stream_idx += len(batch_x)
                
                # 更新类别记录
                unique_y = set(batch_y.cpu().numpy().tolist())
                task_classes.update(unique_y)
                self.seen_classes.update(unique_y)

                for _ in range(self.params.mem_iters):
                    # === MOSE 风格的平衡采样 ===
                    # 1. 区分旧类和新类
                    old_classes = self.seen_classes - task_classes
                    
                    # 2. 计算采样配额
                    total_seen = len(self.seen_classes)
                    if total_seen > 0:
                        # 理想情况下，每个类别应该均分 buffer batch
                        # 新类（当前任务）占比
                        ratio_new = len(task_classes) / total_seen
                        
                        n_new = int(self.params.mem_batch_size * ratio_new)
                        n_old = self.params.mem_batch_size - n_new
                    else:
                        n_new = self.params.mem_batch_size
                        n_old = 0
                        
                    # 3. 执行采样
                    mem_x_list, mem_y_list = [], []
                    
                    # 采样旧类 (Excluding current task classes)
                    if n_old > 0 and len(old_classes) > 0:
                        # 注意：buffer.except_retrieve 需要 list
                        mx_old, my_old = self.buffer.except_retrieve(n_old, list(task_classes))
                        if mx_old.size(0) > 0:
                            mem_x_list.append(mx_old)
                            mem_y_list.append(my_old)
                            
                    # 采样新类 (Only current task classes)
                    # MOSE 也会回放当前任务的数据以增强稳定性
                    if n_new > 0 and len(task_classes) > 0:
                        mx_new, my_new = self.buffer.only_retrieve(n_new, list(task_classes))
                        if mx_new.size(0) > 0:
                            mem_x_list.append(mx_new)
                            mem_y_list.append(my_new)
                            
                    if len(mem_x_list) > 0:
                        mem_x = torch.cat(mem_x_list).to(device)
                        mem_y = torch.cat(mem_y_list).to(device)
                    else:
                        # Fallback if buffer is empty or retrieval failed
                        mem_x, mem_y = self.buffer.random_retrieve(self.params.mem_batch_size)
                        mem_x, mem_y = mem_x.to(device), mem_y.to(device)

                    if mem_x.size(0) == 0:
                        continue

                    # 拼接记忆样本 + 当前流样本
                    # 确保 batch_x 也在 device 上
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    
                    combined_x = torch.cat([mem_x, batch_x])
                    combined_y = torch.cat([mem_y, batch_y])
                    combined_aug = self.transform_train(combined_x)

                    # 学生 logits (增强 + 原始)
                    if hasattr(self.model, "logits"):
                        logits_stu = self.model.logits(combined_aug)      # [B, C]
                        logits_stu_raw = self.model.logits(combined_x)    # [B, C]
                    else:
                        logits_stu = self.model(combined_aug)
                        logits_stu_raw = self.model(combined_x)

                    B, C = logits_stu.shape

                    # -------- 1) ViT (用于蒸馏) + Gate (用于路由) --------
                    # ViT 仅用于特征蒸馏目标
                    
                    # -------- 1) 准备 ViT 输入 (Aug + Raw) 以扩大 Batch --------
                    # 拼接增强图和原图，一次性送入 ViT (Batch Size 变为 2B)
                    gate_input_aug = self.vit_transform(combined_aug)
                    gate_input_raw = self.vit_transform(combined_x)
                    gate_input_all = torch.cat([gate_input_aug, gate_input_raw], dim=0)

                    # ViT 前向传播 [2B, ...]
                    vit_logits_all, features_vit_all = self.pretrained_backbone(gate_input_all)
                    
                    # 将结果拆分回 Aug 和 Raw
                    # features_vit_aug: [B, D], features_vit_raw: [B, D]
                    features_vit_aug, features_vit_raw = torch.split(features_vit_all, B, dim=0)
                    vit_logits_aug, _ = torch.split(vit_logits_all, B, dim=0)

                    # 保持原有的逻辑用于 Prompt 训练 (只用 Aug)
                    loss_prompt_ce = self.criterion(vit_logits_aug, combined_y.long())
                    
                    # 目标特征 (Teacher) 现在包含两部分
                    vit_feat_target_aug = features_vit_aug.detach()
                    vit_feat_target_raw = features_vit_raw.detach()

                    # -------- 获取学生特征 (Aug + Raw) --------
                    if hasattr(self.model, "features"):
                        features_stu_aug = self.model.features(combined_aug)
                        features_stu_raw = self.model.features(combined_x)
                    else:
                        features_stu_aug = logits_stu
                        features_stu_raw = logits_stu_raw
                    
                    # 保持原有变量名 features_stu 指向 Aug，以免破坏后续 Gate 逻辑
                    features_stu = features_stu_aug 
                    gate_input = features_stu.detach()

                    topk_idx, topk_weight, aux_loss = self.gate(gate_input)  #选择教师
                    # 累计整个训练过程中每个专家被选中的次数
                    expert_counts_step = torch.bincount(
                        topk_idx.view(-1).detach().cpu(), minlength=self.n_routed_experts
                    )
                    self.expert_select_counts += expert_counts_step
                    
                    # -------- 2) 路由专家前向(只算被选中的那几个) --------
                    selected_experts_mask = torch.zeros(
                        self.n_routed_experts, dtype=torch.bool, device=device
                    )
                    selected_experts_mask.scatter_(0, topk_idx.view(-1), True) #选中的教师

                    # 优化：只对选中的样本进行前向传播
                    # selected_tea_logits_aug: [B, top_k, C]
                    selected_tea_logits_aug = torch.zeros(B, self.top_k, C, device=device, dtype=logits_stu.dtype)
                    selected_tea_logits_raw = torch.zeros(B, self.top_k, C, device=device, dtype=logits_stu.dtype)
                    
                    unique_experts = torch.unique(topk_idx)
                    
                    for e_idx in unique_experts:
                        e_idx = int(e_idx.item())
                        tea_model = self.routed_teachers[e_idx]
                        
                        # 找到选择该专家的样本索引 (b, k)
                        mask = (topk_idx == e_idx)
                        b_indices, k_indices = torch.where(mask)
                        
                        if len(b_indices) == 0:
                            continue
                            
                        # 提取样本 (注意去重，虽然 b_indices 对于同一个 expert 应该是唯一的)
                        sub_aug = combined_aug[b_indices]
                        sub_raw = combined_x[b_indices]
                        
                        # 前向传播
                        with torch.no_grad():
                            if hasattr(tea_model, "logits"):
                                out_aug = tea_model.logits(sub_aug)
                                out_raw = tea_model.logits(sub_raw)
                            else:
                                out_aug = tea_model(sub_aug)
                                out_raw = tea_model(sub_raw)
                        
                        # 填回结果
                        selected_tea_logits_aug[b_indices, k_indices] = out_aug
                        selected_tea_logits_raw[b_indices, k_indices] = out_raw

                    # -------- 3) MoE 蒸馏损失 (Vectorized) --------
                    # 准备数据
                    stu_logits_aug_exp = logits_stu.unsqueeze(1).expand(-1, self.top_k, -1)
                    stu_logits_raw_exp = logits_stu_raw.unsqueeze(1).expand(-1, self.top_k, -1)
                    
                    # Flatten [B*K, C]
                    flat_tea_aug = selected_tea_logits_aug.view(-1, C)
                    flat_tea_raw = selected_tea_logits_raw.view(-1, C)
                    flat_stu_aug = stu_logits_aug_exp.reshape(-1, C)
                    flat_stu_raw = stu_logits_raw_exp.reshape(-1, C)
                    
                    T = self.params.kd_temperature
                    
                    # 计算 KL Div
                    def calc_kl(tea_logits, stu_logits, temp):
                        log_stu = F.log_softmax(stu_logits / temp, dim=1)
                        prob_tea = F.softmax(tea_logits / temp, dim=1)
                        kl = F.kl_div(log_stu, prob_tea, reduction='none').sum(1) * (temp ** 2)
                        return kl # [B*K]

                    if getattr(self.params, "no_aug", False):
                        kd_val = calc_kl(flat_tea_aug, flat_stu_raw, T)
                    else:
                        kd1 = calc_kl(flat_tea_aug, flat_stu_aug, T)
                        kd2 = calc_kl(flat_tea_aug, flat_stu_raw, T)
                        kd_val = 0.5 * (kd1 + kd2)
                    
                    # Reshape back to [B, K]
                    kd_val = kd_val.view(B, self.top_k)
                    
                    # Weighted sum
                    loss_dist_moe = (kd_val * topk_weight).sum(dim=1).mean()

                    # -------- 4) ViT 特征蒸馏 (Contrastive / InfoNCE) --------
                    # 我们现在有 2B 个样本对: (Stu_Aug, Tea_Aug) 和 (Stu_Raw, Tea_Raw)
                    
                    # 1. 投影学生特征 [2B, D]
                    proj_stu_aug = self.feature_projector(features_stu_aug)
                    proj_stu_raw = self.feature_projector(features_stu_raw)
                    proj_stu_all = torch.cat([proj_stu_aug, proj_stu_raw], dim=0)

                    # 2. 准备专家特征 [2B, D]
                    vit_feat_all = torch.cat([vit_feat_target_aug, vit_feat_target_raw], dim=0)

                    # 3. 归一化
                    stu_norm = F.normalize(proj_stu_all, dim=1)  # [2B, D]
                    tea_norm = F.normalize(vit_feat_all, dim=1)  # [2B, D]

                    # 4. 计算相似度矩阵 [2B, 2B]
                    # logits[i, j] = sim(stu[i], tea[j]) / T
                    T_con = 0.1 # Temperature
                    logits_con = torch.mm(stu_norm, tea_norm.t()) / T_con
                    
                    # 5. InfoNCE Loss
                    # 目标是: stu[i] 应该匹配 tea[i] (对角线是正样本)
                    # 负样本是同一行中的所有其他 tea[j]
                    labels_con = torch.arange(2 * B, device=device)
                    loss_contrastive = F.cross_entropy(logits_con, labels_con)

                    # 使用对比损失作为特征损失 (用于日志或辅助学生)
                    loss_feat = loss_contrastive

                    loss_proj = loss_feat
                    
                    # -------- 5) 总损失 --------
                    loss_ce = self.criterion(logits_stu, combined_y.long())
                    lambda_moe = getattr(self.params, "kd_lambda", 10)
                    lambda_feat = getattr(self.params, "lambda_feat",0)
                    #loss_proj =  loss_feat
                    loss = loss_ce + lambda_feat * loss_feat  + lambda_moe * loss_dist_moe 
                    if aux_loss is not None:
                        loss = loss + aux_loss

                    loss = loss.mean()
                    self.loss = float(loss.item())

                    # --------  反向传播 + 优化 --------
                    self.proj_optim.zero_grad(set_to_none=True)
                    self.optim.zero_grad()
                    self.gate_optim.zero_grad()
                    self.prompt_optim.zero_grad() 
                    
                    # --------  更新 projector --------
                    if lambda_feat > 0:    #在进行消融实验中不能使用投影层
                        scaler.scale(loss_proj).backward(retain_graph=True)  
                        scaler.step(self.proj_optim)
                    
                    # -------- 更新 Prompts + Student + Gate --------
                    # 合并 loss_prompt_ce 和 loss
                    total_loss = loss + loss_prompt_ce
                    scaler.scale(total_loss).backward()
                    
                    scaler.step(self.prompt_optim)
                    scaler.step(self.optim)
                    scaler.step(self.gate_optim)
                    scaler.update()

                    # -------- 8) wandb 记录(与其他指标同频) --------
                    if not getattr(self.params, "no_wandb", False):
                        try:
                            log_dict = {
                                "loss/total": float(loss.item()),
                                "loss/prompt_ce": float(loss_prompt_ce.item()), 
                                "loss/moe": float(loss_dist_moe.item()),
                                "loss/feat": float(loss_feat.item()),
                                "loss/ce": float(loss_ce.item()),
                                "gate/aux_loss": float(aux_loss.item()) if aux_loss is not None else 0.0,
                                "gate/lr": float(self.gate_optim.param_groups[0]["lr"]),
                                "projector/lr": float(self.proj_optim.param_groups[0]["lr"]),
                                "projector/loss_feat": float(loss_feat.item()),
                                "experts/selected_counts_cum": self.expert_select_counts.tolist(),
                            }
                            wandb.log(log_dict)
                        except Exception:
                            pass

                    # -------- 7) EMA 更新教师参数(只更新用到的专家) --------
                    self.update_ema_all(init=False, routed_mask=selected_experts_mask)

                    # -------- 8) scheduler、drift、打印(你可以按需加回 wandb.log) --------
                    if getattr(self.params, "annealing", False):
                        self.scheduler.step()

                    if (
                        getattr(self.params, "measure_drift", 0) > 0
                        and task_id is not None
                        and task_id > 0
                    ):
                        self.measure_drift(task_id)

                    print(
                        f"Phase {task_name} | "
                        f"Loss {loss.item():.3f} | "
                        f"Loss_MoE {loss_dist_moe.item():.3f} | "
                        f"Loss_Feat {loss_feat.item():.3f} | "
                        f"batch {j}",
                        end="\r",
                    )

                # -------- 9) 更新经验缓冲区 --------
                self.buffer.update(imgs=batch_x, labels=batch_y)

                # -------- 10) 收尾 --------
                if (j == (len(dataloader) - 1)) and (j > 0):
                    if getattr(self.params, "tsne", False) and task_id == 4:
                        self.tsne()
                    print(
                        f"\nPhase {task_name} | "
                        f"batch {j}/{len(dataloader)} | "
                        f"Loss {self.loss:.4f} | "
                        f"time {time.time() - self.start:.4f}s"
                    )
                    if not getattr(self.params, "no_wandb", False):
                        try:
                            wandb.log({"experts/selected_counts_final": self.expert_select_counts.tolist()})
                        except Exception:
                            pass
                    self.save(model_name=f"ckpt_{task_name}_task{task_id}")

