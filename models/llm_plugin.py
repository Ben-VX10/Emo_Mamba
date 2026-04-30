import torch
import torch.nn as nn

# 👇 新增：针对 PyTorch 2.4.0 缺失 set_submodule 的兼容性补丁 (Monkey Patch) 👇
if not hasattr(nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: nn.Module) -> None:
        atoms = target.split(".")
        name = atoms.pop(-1)
        mod = self
        for item in atoms:
            mod = getattr(mod, item)
        setattr(mod, name, module)
    nn.Module.set_submodule = _set_submodule
# 👆 ------------------------------------------------------------------- 👆

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 导入我们刚刚跑通的底层模块
from models.model import EmoMambaLLM_Base

class EmoMambaLLM_Complete(nn.Module):
    def __init__(self, llm_model_name_or_path, audio_dim=1024, visual_dim=2048, fusion_dim=512):
        super().__init__()
        
        print("正在加载基础特征提取模块 (Conv-Attention + Bi-Mamba)...")
        self.base_model = EmoMambaLLM_Base(audio_dim, visual_dim, fusion_dim)
        
        print("正在配置 4-bit QLoRA 大模型量化参数...")
        # 4-bit 量化配置，极限压缩显存
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        print(f"正在加载大语言模型基座: {llm_model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name_or_path, trust_remote_code=True)
        # LLaMA 默认没有 pad_token，需要手动设置
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto" # 自动将模型分配到有显存的 GPU 上
        )
        
        # 配置 LoRA 插件
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"], # 针对注意力机制注入可训练参数
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters() # 打印一下参数，你会发现可训练参数极少！
        
        # 获取 LLM 的隐藏层维度 (TinyLlama 通常是 2048, LLaMA-7B 是 4096)
        llm_hidden_size = self.llm.config.hidden_size
        
        print("正在构建跨模态对齐适配器 (Adapter)...")
        # 将我们 Bi-Mamba 输出的 1024 维，映射到 LLM 认识的维度
        self.multimodal_projector = nn.Linear(fusion_dim * 2, llm_hidden_size)

        # 新增这块代码：将自定义的网络模块对齐到大模型所在的设备 (GPU) 上
        self.base_model.to(self.llm.device)
        self.multimodal_projector.to(self.llm.device)
        # -----------------------------------------------------------
        
    def forward(self, audio, visual, text_targets):
        # 1. 提取多模态时空特征 (Batch, 1, 1024)
        multimodal_features = self.base_model(audio, visual)
        
        # 2. 映射到大模型空间 (Batch, 1, LLM_Hidden_Size)
        multimodal_embeds = self.multimodal_projector(multimodal_features)
        
        # 3. 处理文本 Prompt
        # 我们设计一个引导大模型进行推理的 prompt
        prompts = ["Please reason about the emotion based on the given visual and audio cues. Answer: " for _ in range(len(text_targets))]
        
        # 将 Prompt 和真实目标文本 tokenize
        prompt_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(multimodal_embeds.device)
        target_inputs = self.tokenizer(text_targets, return_tensors="pt", padding=True, truncation=True, max_length=128).to(multimodal_embeds.device)
        
        # 获取文本的 embeddings
        prompt_embeds = self.llm.model.model.embed_tokens(prompt_inputs.input_ids)
        target_embeds = self.llm.model.model.embed_tokens(target_inputs.input_ids)

        # 新增这一行：将多模态特征的数据类型（dtype）强行对齐到大模型文本特征的类型
        multimodal_embeds = multimodal_embeds.to(prompt_embeds.dtype)
        #  ------------------------------------------------------------------
        
        # 4. 拼接！将 [多模态特征] + [Prompt] + [目标答案] 拼接在一起送给大模型
        # (Batch, 1 + Prompt_Len + Target_Len, LLM_Hidden_Size)
        inputs_embeds = torch.cat([multimodal_embeds, prompt_embeds, target_embeds], dim=1)
        
        # 构造对应的 Labels (为了计算 Loss)
        # 前面的多模态特征和 prompt 我们不计算 loss (设为 -100)
        batch_size = inputs_embeds.shape[0]
        ignore_index = -100
        labels_multimodal = torch.full((batch_size, 1), ignore_index, dtype=torch.long).to(multimodal_embeds.device)
        labels_prompt = torch.full(prompt_inputs.input_ids.shape, ignore_index, dtype=torch.long).to(multimodal_embeds.device)
        labels_target = target_inputs.input_ids
        
        labels = torch.cat([labels_multimodal, labels_prompt, labels_target], dim=1)
        
        # 5. 送入大模型前向传播并计算 Loss
        outputs = self.llm(inputs_embeds=inputs_embeds, labels=labels)
        
        return outputs.loss

if __name__ == "__main__":
    from scripts.dataset import MERRDataset
    from torch.utils.data import DataLoader
    
    # 我们使用极小的 TinyLlama 作为本地测试替身
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print("开始终极融合测试...")
    dataset = MERRDataset("../data/annotations/MERR_fine_grained.json", "../data/features/")
    # 注意：包含了 LLM 后，即使是 Batch=1 也需要吃显存
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True) 
    
    model = EmoMambaLLM_Complete(llm_model_name_or_path=MODEL_NAME)
    
    audio_batch, visual_batch, text_batch = next(iter(dataloader))
    
    # 因为使用了 device_map="auto"，我们手动把多模态输入也移动到对应的 GPU 上
    device = model.llm.device
    audio_batch = audio_batch.to(device)
    visual_batch = visual_batch.to(device)
    
    # 终极一跑！
    loss = model(audio_batch, visual_batch, text_batch)
    
    print("\n" + "="*50)
    print("恭喜！整个链路已彻底打通！")
    print(f"跑出的第一笔 Loss 值: {loss.item():.4f}")
    print("="*50)
