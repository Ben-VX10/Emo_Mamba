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
    # 构建同义词/词性映射表
    emotion_mapping = {
        "anger": "angry", "angry": "angry", "mad": "angry", "frustration": "angry",
        "disgust": "disgust", "disgusted": "disgust",
        "fear": "fear", "fearful": "fear", "scared": "fear", "anxiety": "fear", "anxious": "fear",
        "happiness": "happy", "happy": "happy", "joy": "happy", "cheerful": "happy",
        "sadness": "sad", "sad": "sad", "melancholy": "sad", "despair": "sad", "sorrow": "sad",
        "surprise": "surprise", "surprised": "surprise",
        "neutral": "neutral", "normal": "neutral", "calm": "neutral"
    }
    
    text_lower = text.lower()
    
    # 遍历字典，只要模型原话里包含这些词，就映射到标准 7 分类
    for key, target in emotion_mapping.items():
        if key in text_lower:
            return target
            
    return "neutral" # 实在没找到再给 neutral

def evaluate_local():
    # print("初始化本地测试评测流水线 (12GB显存保命版)...")
    print("初始化")
    
    # ================= 1. 本地路径与参数 =================
    TEST_JSON_PATH = "../data/annotations/MERR_fine_grained.json" 
    FEATURES_DIR = "../data/features/"
    
    # 大模型路径
    MODEL_NAME = "/root/autodl-fs/Llama-2-7b-chat-hf" 
    
     # 假设你想测试第3轮的结果
    CHECKPOINT_DIR = "../checkpoints/epoch_3/"
    
    # ================= 2. 加载数据 =================
    dataset = MERRDataset(TEST_JSON_PATH, FEATURES_DIR)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) 
    
    # ================= 3. 加载模型基座 =================
    model = EmoMambaLLM_Complete(llm_model_name_or_path=MODEL_NAME)
    device = model.llm.device
    
    # ================= 4. 挂载本地权重 =================
    print(f"正在注入本地测试权重: {CHECKPOINT_DIR}")
    model.base_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "base_model.pth"), weights_only=True))
    model.multimodal_projector.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "projector.pth"), weights_only=True))
    
    # 挂载 LoRA 插件：直接将训练好的权重加载到现有的 LoRA 结构中，彻底解决套娃报错！
    model.llm.load_adapter(os.path.join(CHECKPOINT_DIR, "lora_weights"), adapter_name="default")
    
    model.eval() # 极其重要：切换到测试模式，关闭 Dropout 等
    
    y_true = []
    y_pred = []
    
    print("\n开始本地闭卷考试推理测试...")
    # ================= 5. 真实推理生成逻辑 =================
    with torch.no_grad():
        for i, (audio_batch, visual_batch, text_batch) in enumerate(dataloader):
            # if i >= 5: # 本地仅跑 5 个样本跑通流水线即可
            #     break
                
            audio_batch = audio_batch.to(device)
            visual_batch = visual_batch.to(device)
            
            # (1) 提取多模态特征
            multimodal_features = model.base_model(audio_batch, visual_batch)
            multimodal_embeds = model.multimodal_projector(multimodal_features)

            # 👇 --- 新增：诊断特征坍塌的“探针” --- 👇
            print(f"探针: 多模态特征总和 = {multimodal_embeds.sum().item():.4f}")
            # 👆 ------------------------------------- 👆
            
            # (2) 回归最纯净的引导：不加废话，不加指令！
            prompt = "The emotion is "
            prompt_inputs = model.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
            prompt_embeds = model.llm.get_input_embeddings()(prompt_inputs.input_ids)
            
            # (3) 拼接输入
            multimodal_embeds = multimodal_embeds.to(prompt_embeds.dtype)
            inputs_embeds = torch.cat([multimodal_embeds, prompt_embeds], dim=1)
            
            # (4) 呼叫大模型 generate 函数进行生成
            generated_ids = model.llm.generate(
                inputs_embeds=inputs_embeds, 
                max_new_tokens=50,          # 留足空间，让它自然写完一句话
                do_sample=True,             # 🚨 核心修改：开启采样！彻底打破 The The The 死循环！
                temperature=0.4,            # 加上微小的随机性，让它思维活跃起来
                top_p=0.9,                  # 截断长尾生僻词，防止它胡言乱语
                repetition_penalty=1.1,     # 🚨 极其温柔的重复惩罚：1.1 既不会产生  乱码，又能防止结巴
                pad_token_id=model.tokenizer.eos_token_id
            )
            
            # (5) 解码输出文本
            generated_text = model.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # 👇 新增：原封不动地打印大模型的真实原始输出（带括号方便看清空格和换行） 👇
            print(f"【模型原始生肉输出】: [{generated_text}]")
            
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
    print(f"准确率: {acc*100:.2f}% (本地仅训练10步，瞎猜很正常)")
    print(f"加权 F1: {f1*100:.2f}%")
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
    print("\n混淆矩阵图已保存至项目根目录: local_confusion_matrix.png")

if __name__ == "__main__":
    evaluate_local()
