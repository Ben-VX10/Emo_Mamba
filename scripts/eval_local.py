import sys
import os
# 开天眼，让脚本找到上一级的 models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from dataset import MERRDataset
from models.llm_plugin import EmoMambaLLM_Complete
from torch.utils.data import DataLoader
from peft import PeftModel

# 提取情感标签的工具函数
def extract_emotion(text):
    emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    text_lower = text.lower()
    for emo in emotions:
        if emo in text_lower:
            return emo
    return "neutral" # 默认给个 neutral，防止没找到标签报错

def evaluate_local():
    print("📊 初始化本地测试评测流水线 (12GB显存保命版)...")
    
    # ================= 1. 本地路径与参数 =================
    TEST_JSON_PATH = "../data/annotations/MERR_fine_grained.json" 
    FEATURES_DIR = "../data/features/"
    
    # 换回你本地的替身大模型
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 
    
    # 读取你之前在本地成功跑完并保存的 10 个 Step 的权重
    CHECKPOINT_DIR = "../checkpoints/local_test/" 
    
    # ================= 2. 加载数据 =================
    dataset = MERRDataset(TEST_JSON_PATH, FEATURES_DIR)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) 
    
    # ================= 3. 加载模型基座 =================
    model = EmoMambaLLM_Complete(llm_model_name_or_path=MODEL_NAME)
    device = model.llm.device
    
    # ================= 4. 挂载本地权重 =================
    print(f"⏳ 正在注入本地测试权重: {CHECKPOINT_DIR}")
    model.base_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "base_model.pth"), weights_only=True))
    model.multimodal_projector.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "projector.pth"), weights_only=True))
    
    # 挂载 LoRA 插件
    model.llm = PeftModel.from_pretrained(model.llm, os.path.join(CHECKPOINT_DIR, "lora_weights"))
    
    model.eval() # 极其重要：切换到测试模式，关闭 Dropout 等
    
    y_true = []
    y_pred = []
    
    print("\n📝 开始本地闭卷考试推理测试...")
    # ================= 5. 真实推理生成逻辑 =================
    with torch.no_grad():
        for i, (audio_batch, visual_batch, text_batch) in enumerate(dataloader):
            if i >= 5: # 本地仅跑 5 个样本跑通流水线即可
                break
                
            audio_batch = audio_batch.to(device)
            visual_batch = visual_batch.to(device)
            
            # (1) 提取多模态特征
            multimodal_features = model.base_model(audio_batch, visual_batch)
            multimodal_embeds = model.multimodal_projector(multimodal_features)
            
            # (2) 准备提示词并转成 Embeddings
            prompt = "Please reason about the emotion based on the given visual and audio cues. Answer: "
            prompt_inputs = model.tokenizer(prompt, return_tensors="pt").to(device)
            prompt_embeds = model.llm.get_input_embeddings()(prompt_inputs.input_ids)
            
            # (3) 拼接输入 (精度必须对齐！)
            multimodal_embeds = multimodal_embeds.to(prompt_embeds.dtype)
            inputs_embeds = torch.cat([multimodal_embeds, prompt_embeds], dim=1)
            
            # (4) 呼叫大模型 generate 函数进行生成
            generated_ids = model.llm.generate(
                inputs_embeds=inputs_embeds, 
                max_new_tokens=20, # 只让他生成20个词，加快本地测试速度
                pad_token_id=model.tokenizer.eos_token_id
            )
            
            # (5) 解码输出文本
            generated_text = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # (6) 提取情绪词
            pred_emo = extract_emotion(generated_text)
            true_emo = extract_emotion(text_batch[0])
            
            print(f"\n--- 样本 {i+1} ---")
            print(f"真实情感: {true_emo} | 预测情感: {pred_emo}")
            print(f"模型原话: {generated_text.strip()}")
            
            y_pred.append(pred_emo)
            y_true.append(true_emo)
            
    # ================= 6. 绘图与出成绩 =================
    print("\n" + "="*40)
    acc = accuracy_score(y_true, y_pred)
    # 本地跑的数据极少，算 F1 可能会报警告，这里忽略
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) 
    print(f"🎯 准确率: {acc*100:.2f}% (本地仅训练10步，瞎猜很正常)")
    print(f"🥇 加权 F1: {f1*100:.2f}%")
    print("="*40)
    
    # 为了防止这5个样本预测出的类别太少报错，统一下坐标轴标签
    labels = sorted(list(set(y_true + y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Local Test Confusion Matrix', fontsize=16)
    plt.ylabel('True Emotion', fontsize=12)
    plt.xlabel('Predicted Emotion', fontsize=12)
    
    # 保存图片
    plt.savefig('../local_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n📸 混淆矩阵图已保存至项目根目录: local_confusion_matrix.png")

if __name__ == "__main__":
    evaluate_local()