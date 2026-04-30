import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import sys
# 将项目根目录（当前文件的上一级）动态加入 Python 的环境变量中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 导入我们之前写好的模块 (根据你实际的文件结构调整导入路径)
from dataset import MERRDataset
from models.llm_plugin import EmoMambaLLM_Complete

def train():
    print("初始化本地训练流水线...")
    
    # ================= 1. 超参数与路径配置 =================
    JSON_PATH = "../data/annotations/MERR_fine_grained.json"
    FEATURES_DIR = "../data/features/"
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # 本地测试替身
    
    BATCH_SIZE = 1        # 本地 12GB 显存保命设置
    LEARNING_RATE = 2e-4  # QLoRA 微调的经典学习率
    MAX_STEPS = 10        # 本地仅试跑 10 个 Step 验证反向传播，不跑完全程
    
    # ================= 2. 加载数据 =================
    print("正在构建 DataLoader...")
    dataset = MERRDataset(JSON_PATH, FEATURES_DIR)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # ================= 3. 加载模型 =================
    model = EmoMambaLLM_Complete(
        llm_model_name_or_path=MODEL_NAME, 
        audio_dim=1024, 
        visual_dim=2048, 
        fusion_dim=512
    )
    
    device = model.llm.device # 获取模型被分配到的 GPU (通常是 cuda:0)
    
    # ================= 4. 提取可训练参数 =================
    # 极度重要：我们绝对不能把冻结的参数交给优化器，否则会报错或浪费算力
    print("正在筛查可训练参数 (Adapter + QLoRA)...")
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            # print(f"  - 加入优化器: {name}") # 可以取消注释查看具体是哪些层
            
    optimizer = AdamW(trainable_params, lr=LEARNING_RATE)
    
    # ================= 5. 正式开启训练循环 =================
    print("\n" + "="*50)
    print(f"开始训练测试! 最大测试步数: {MAX_STEPS}")
    print("="*50)
    
    model.train() # 开启训练模式
    
    for step, (audio_batch, visual_batch, text_batch) in enumerate(dataloader):
        if step >= MAX_STEPS:
            print(f"\n已达到本地测试设定步数 ({MAX_STEPS})，停止训练。")
            break
            
        # 将多模态输入推送到显卡
        audio_batch = audio_batch.to(device)
        visual_batch = visual_batch.to(device)
        
        # 梯度清零 (PyTorch 训练标准起手式)
        optimizer.zero_grad()
        
        # 前向传播，计算 Loss
        loss = model(audio_batch, visual_batch, text_batch)
        
        # 反向传播，计算梯度
        loss.backward()
        
        # 更新权重参数
        optimizer.step()
        
        print(f"Step [{step+1}/{MAX_STEPS}] | Loss: {loss.item():.4f}")
        
        # 清理一下显存碎片 (对于显存处于极限边缘的 4070 很有用)
        torch.cuda.empty_cache()

    # ================= 6. 模拟保存模型 =================
    save_dir = "../checkpoints/local_test/"
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n正在保存跨模态对齐适配器与 LoRA 权重...")
    # 注意：我们只保存自己写的 Adapter 网络和 PEFT 权重，不保存 7B 基座
    torch.save(model.base_model.state_dict(), os.path.join(save_dir, "base_model.pth"))
    torch.save(model.multimodal_projector.state_dict(), os.path.join(save_dir, "projector.pth"))
    model.llm.save_pretrained(os.path.join(save_dir, "lora_weights"))
    
    print(f"训练脚本跑通！权重已存入: {save_dir}")
    print("下一步：可以打包项目前往 AutoDL Linux 服务器跑全量实验了！")

if __name__ == "__main__":
    train()