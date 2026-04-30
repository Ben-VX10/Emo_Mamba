import json
import pickle
import numpy as np
from pathlib import Path

def explore_merr_json(json_path):
    print("="*50)
    print(f"正在读取 MERR JSON 文件: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    print(f"成功加载！数据总条数: {len(data)}")
    print("第一条数据的结构和内容展示")
    
    # 获取第一条数据
    first_item = data[0] if isinstance(data, list) else next(iter(data.values()))
    
    # 格式化打印第一条数据的内容
    print(json.dumps(first_item, indent=4, ensure_ascii=False))
    print("="*50)

def explore_feature_pkl(pkl_path):
    print("="*50)
    print(f"正在读取特征 PKL 文件: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    # 判断数据是列表还是字典
    if isinstance(data, list):
        print(f"成功加载！这是一个 List，包含 {len(data)} 个样本。")
        first_item = data[0]
    elif isinstance(data, dict):
        print(f"成功加载！这是一个 Dict，包含 {len(data)} 个样本。")
        first_key = list(data.keys())[0]
        first_item = data[first_key]
        print(f"第一个样本的 Key (比如对话ID) 是: {first_key}")
    else:
        print(f"未知的数据结构: {type(data)}")
        return

    print("第一条样本包含的特征维度")
    # 遍历打印特征名称和它的维度
    for key, value in first_item.items():
        if isinstance(value, np.ndarray):
            print(f" - [{key}]: Numpy Array, 形状 Shape = {value.shape}, 数据类型 = {value.dtype}")
        elif hasattr(value, 'shape'): # 兼容 PyTorch Tensor
            print(f" - [{key}]: Tensor, 形状 Shape = {value.shape}")
        else:
            # 可能是文本标签或ID，截取显示前50个字符
            val_str = str(value)
            print(f" - [{key}]: {type(value).__name__}, 内容 = {val_str[:50]}{'...' if len(val_str)>50 else ''}")
    print("="*50)

if __name__ == "__main__":
    # 请确保这里的路径和你的实际存放路径一致
    # 假设你把下载的文件放到了 data/ 目录下
    
    merr_path = Path("../data/annotations/MERR_fine_grained.json") 
    # 如果你在根目录运行，路径应该是 Path("data/MERR_fine_grained.json")
    
    meld_path = Path("../data/benchmarks/meld_multimodal_features.pkl")
    # 如果你在根目录运行，路径应该是 Path("data/meld_multimodal_features.pkl")

    # 1. 探索 MERR 细粒度推理文本
    if merr_path.exists():
        explore_merr_json(merr_path)
    else:
        print(f"找不到文件: {merr_path}")

    # 2. 探索 MELD 多模态特征底座
    if meld_path.exists():
        explore_feature_pkl(meld_path)
    else:
        print(f"找不到文件: {meld_path}")