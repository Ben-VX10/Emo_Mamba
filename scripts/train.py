import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from dataset import MERRDataset
from models.llm_plugin import EmoMambaLLM_Complete

def train():
    print("初始化服务器全量训练流水线...")
    
    # ================= 1. 超参数与路径配置 =================
    JSON_PATH = "../data/annotations/MERR_fine_grained.json"
    FEATURES_DIR = "../data/features/"
    
    # 换成完整的 7B 大模型 (请确保你已经获取了访问权限或使用本地路径)
    MODEL_NAME = "/root/autodl-fs/Llama-2-7b-chat-hf" 
    
    # 释放算力，调大 Batch Size，设置完整 Epoch 数
    BATCH_SIZE = 16
    GRADIENT_ACCUMULATION_STEPS = 4  # 累加 4 次再更新梯度 (4 x 4 = 16，等效于 BATCH_SIZE=16！)
    LEARNING_RATE = 5e-5  
    NUM_EPOCHS = 3   # 跑完整的 3 轮数据集
    
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
    
    device = model.llm.device 
    
    # ================= 4. 提取可训练参数 =================
    print("正在筛查可训练参数 (Adapter + QLoRA)...")
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            
    optimizer = AdamW(trainable_params, lr=LEARNING_RATE)
    
    # ================= 5. 正式开启全量训练循环 =================
    print("\n" + "="*50)
    print(f"开始全量训练! 总 Epoch 数: {NUM_EPOCHS}")
    print("="*50)
    
    model.train() 
    
    # 增加 Epoch 外层循环
    for epoch in range(NUM_EPOCHS):
        print(f"\n开始 Epoch [{epoch+1}/{NUM_EPOCHS}]")
        
        # 使用 tqdm 包装 dataloader，在终端显示漂亮的进度条
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for step, (audio_batch, visual_batch, text_batch) in enumerate(progress_bar):
            audio_batch = audio_batch.to(device)
            visual_batch = visual_batch.to(device)
            
            # 把情绪词包装成一句话，迎合 LLaMA-2 的语言习惯
            # 比如 "angry" 变成 "The emotion is angry."
            formatted_texts = [f"The emotion is {emo}." for emo in text_batch]
            
            # 前向传播，计算 Loss (传入包装后的文本)
            loss = model(audio_batch, visual_batch, formatted_texts)
            
            # 反向传播，计算梯度
            # 将 Loss 除以累加步数，使梯度的量级保持稳定
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            # 只有当累加够了步数，才进行一次真正的参数更新
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                
                # 梯度裁剪！如果梯度大于 1.0，直接强行砍掉，防止把大模型烧傻
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                
                optimizer.step() # 更新权重参数
                optimizer.zero_grad() # 更新完后再清空梯度
            
            # 实时更新进度条上的 Loss 显示
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            torch.cuda.empty_cache()

        # ================= 每个 Epoch 结束模拟保存模型 =================
        save_dir = f"../checkpoints/epoch_{epoch+1}/"
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n正在保存 Epoch {epoch+1} 的权重到 {save_dir}...")
        torch.save(model.base_model.state_dict(), os.path.join(save_dir, "base_model.pth"))
        torch.save(model.multimodal_projector.state_dict(), os.path.join(save_dir, "projector.pth"))
        model.llm.save_pretrained(os.path.join(save_dir, "lora_weights"))

    print("\n恭喜！所有 Epoch 训练圆满结束！")

if __name__ == "__main__":
    train()
