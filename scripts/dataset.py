import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from pathlib import Path

class MERRDataset(Dataset):
    def __init__(self, json_path, features_dir):
        self.json_path = Path(json_path)
        self.features_dir = Path(features_dir)
        
        # 加载标注文本
        with open(self.json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # 兼容处理：如果加载出来是字典，将其键值对拆解成列表
        if isinstance(raw_data, dict):
            self.video_ids = list(raw_data.keys())   # 保存类似 'sample_00000047' 的真实键名
            self.data = list(raw_data.values())      # 保存对应的内容字典
        else:
            self.data = raw_data
            # 如果碰巧是列表，就回退到基础编号策略
            self.video_ids = [f"sample_{i:08d}" for i in range(len(self.data))]
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_id = self.video_ids[idx]
        
        # 动态获取对应的真实文件名
        file_name = f"{video_id}.npy"
        
        # 1. 加载听觉特征 (HuBERT)
        audio_path = self.features_dir / "HL-UTT" / file_name
        audio_feat = torch.from_numpy(np.load(audio_path)).float()
        
        # 2. 加载视觉局部特征 (MAE)
        mae_path = self.features_dir / "mae_340_UTT" / file_name
        mae_feat = torch.from_numpy(np.load(mae_path)).float()
        
        # 3. 加载视觉时序特征 (VideoMAE)
        vmae_path = self.features_dir / "maeV_399_UTT" / file_name
        vmae_feat = torch.from_numpy(np.load(vmae_path)).float()
        
        # 视觉特征初步拼接 (沿着特征维度)
        visual_feat = torch.cat([mae_feat, vmae_feat], dim=-1)
        
        # 获取最终用于大模型微调的目标文本
        target_text = item.get("smp_reason_caption", "")
        
        return audio_feat, visual_feat, target_text

# 本地简单测试模块
if __name__ == "__main__":
    # 根据你的实际存放位置修改路径
    json_path = "../data/annotations/MERR_fine_grained.json"
    features_dir = "../data/features/"
    
    dataset = MERRDataset(json_path, features_dir)
    audio, visual, text = dataset[0]
    
    print(f"数据集加载成功！总样本数: {len(dataset)}")
    print(f"音频特征维度: {audio.shape}")
    print(f"视觉特征维度: {visual.shape}")
    print(f"目标推理文本: {text[:50]}...")