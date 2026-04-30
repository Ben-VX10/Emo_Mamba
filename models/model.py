import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- 修复跨文件夹导包的核心代码 ---
import sys
from pathlib import Path
# 获取当前 model.py 所在的 models 目录
current_dir = Path(__file__).resolve().parent
# 获取上一级目录（也就是 Emo_Mamba 项目根目录）
project_root = current_dir.parent
# 将根目录加入 Python 的临时环境变量
sys.path.append(str(project_root))
# -------------------------------------

from scripts.dataset import MERRDataset

class DepthwiseSeparableConv1d(nn.Module):
    """深度可分离卷积：大幅降低参数量，非常适合低算力环境"""
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        # 注意：由于目前的特征序列长度为1，kernel_size 和 padding 设为 1 和 0
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, 
                                   padding=0, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ConvAttentionFusion(nn.Module):
    """端到端多视角卷积注意力预融合机制"""
    def __init__(self, audio_dim=1024, visual_dim=2048, hidden_dim=512):
        super().__init__()
        self.audio_conv = DepthwiseSeparableConv1d(audio_dim, hidden_dim)
        self.visual_conv = DepthwiseSeparableConv1d(visual_dim, hidden_dim)
        
        self.audio_cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        self.visual_cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        
    def forward(self, audio, visual):
        # 增加序列维度并调整为 Conv1d 需要的 (Batch, Channels, Length)
        # 输入形状从 (B, Dim) 变为 (B, Dim, 1)
        audio = audio.unsqueeze(-1)
        visual = visual.unsqueeze(-1)
        
        # 经过卷积增强并转置为 Attention 需要的 (Batch, Length, Hidden_dim)
        audio_conv = self.audio_conv(audio).transpose(1, 2)  # (B, 1, hidden_dim)
        visual_conv = self.visual_conv(visual).transpose(1, 2) # (B, 1, hidden_dim)
        
        audio_fused, _ = self.audio_cross_attn(query=audio_conv, key=visual_conv, value=visual_conv)
        visual_fused, _ = self.visual_cross_attn(query=visual_conv, key=audio_conv, value=audio_conv)
        
        # 将互补后的特征拼接 -> (Batch, 1, 1024)
        fused_features = torch.cat([audio_fused, visual_fused], dim=-1)
        return fused_features

class MockBiMamba(nn.Module):
    """
    Bi-Mamba 的本地 Windows 测试替身。
    用于在低显存下模拟 O(N) 复杂度的长程时序推理的数据流。
    部署到 Linux 服务器后可替换为真实的 mamba-ssm。
    """
    def __init__(self, d_model):
        super().__init__()
        # 用简单的线性层模拟 Mamba 的前向和后向扫描状态转换
        self.forward_scan = nn.Linear(d_model, d_model)
        self.backward_scan = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x shape: (Batch, Seq_Len, d_model)
        out_fwd = torch.relu(self.forward_scan(x))
        # 模拟反向扫描（真实 Mamba 会对序列进行 flip）
        out_bwd = torch.relu(self.backward_scan(x))
        
        # 结合双向信息
        out = self.norm(out_fwd + out_bwd)
        return out

class EmoMambaLLM_Base(nn.Module):
    """总装车间：组合预融合与时空状态空间模型"""
    def __init__(self, audio_dim=1024, visual_dim=2048, fusion_dim=512):
        super().__init__()
        self.fusion_module = ConvAttentionFusion(audio_dim, visual_dim, fusion_dim)
        # 融合后拼接的维度是 fusion_dim * 2 = 1024
        self.bi_mamba = MockBiMamba(d_model=fusion_dim * 2)
        
    def forward(self, audio, visual):
        fused_features = self.fusion_module(audio, visual)
        temporal_features = self.bi_mamba(fused_features)
        return temporal_features

# ----------------- 本地全流程连通性测试 -----------------
if __name__ == "__main__":
    print("开始数据流通道与网络架构连通性测试...")
    
    # 1. 实例化 Dataset 和 DataLoader
    json_path = "../data/annotations/MERR_fine_grained.json"
    features_dir = "../data/features/"
    
    dataset = MERRDataset(json_path, features_dir)
    # Batch Size 设为 2，避免本地 12GB 显存溢出
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    # 2. 实例化我们的核心架构
    model = EmoMambaLLM_Base(audio_dim=1024, visual_dim=2048, fusion_dim=512)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 3. 抽取一个 Batch 的真实数据进行前向传播
    audio_batch, visual_batch, text_batch = next(iter(dataloader))
    audio_batch, visual_batch = audio_batch.to(device), visual_batch.to(device)
    
    # 跑通数据流！
    output_features = model(audio_batch, visual_batch)
    
    print("网络前向传播成功！")
    print(f"输入音频 Batch: {audio_batch.shape}")
    print(f"输入视觉 Batch: {visual_batch.shape}")
    print(f"输出特征 (即将送入 QLoRA 和 LLaMA): {output_features.shape}")
    print(f"对应的微调目标文本 (前 30 个字符): {text_batch[0][:30]}...")