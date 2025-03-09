from transformers import LogitsProcessor, GPT2LMHeadModel, PreTrainedTokenizerFast
from rdkit import Chem
from rdkit.Chem import QED
import torch
import re
from transformers import LogitsProcessor
from rdkit import Chem
from collections import OrderedDict
import re
import torch
from rdkit.Chem import QED


from transformers import LogitsProcessor, GPT2LMHeadModel, PreTrainedTokenizerFast, BertModel, BertTokenizer
from rdkit import Chem
from collections import OrderedDict
import re
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn
app = FastAPI()
class QEDPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = torch.nn.Dropout(0.1)
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(768, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        return self.regressor(self.dropout(pooled_output)).squeeze()

def init_qed_predictor(model_path, device="cpu"):
    model = QEDPredictor()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

class QEDGuidedEOSProcessor(LogitsProcessor):
    def __init__(self, 
                 tokenizer,
                 eos_boost=30.0,
                 qed_threshold=0.6,
                 min_length=15,
                 top_k=100,
                 qed_model=None,
                 qed_tokenizer=None):
        
        self.tokenizer = tokenizer
        self.eos_boost = eos_boost
        self.qed_threshold = qed_threshold
        self.min_length = min_length
        self.top_k = top_k
        self.eos_token_id = tokenizer.eos_token_id
        self.bos_token_id = tokenizer.bos_token_id
        self.qed_model = qed_model
        self.qed_tokenizer = qed_tokenizer

    def _clean_smiles(self, text):
        # """清洗SMILES并截断到第一个EOS"""
        # eos_pos = text.find(self.tokenizer.eos_token)
        # clean = text[:eos_pos] if eos_pos != -1 else text
        # return re.sub(r'[^A-Za-z0-9@+\-$$$$=#%]', '', clean)
        pass

    def __call__(self, input_ids, scores):
        batch_size, _ = input_ids.shape
        
        # 获取所有序列的top_k候选 [batch_size, top_k]
        topk_scores, topk_tokens = torch.topk(scores, self.top_k, dim=-1)
        
        for batch_idx in range(batch_size):
            current_ids = input_ids[batch_idx].tolist()
            
            # 解码当前序列（不含特殊token）
            filtered_ids = [x for x in current_ids 
                           if x not in [self.bos_token_id, self.eos_token_id]]
            current_smiles = self.tokenizer.decode(filtered_ids)
            
            # 最小长度保护
            if len(current_smiles) < self.min_length:
                continue
                
            # 检查EOS是否在当前候选的top_k中
            if self.eos_token_id not in topk_tokens[batch_idx]:
                if len(filtered_ids)%10==0:
                    inputs =self.qed_tokenizer(
                current_smiles,
                add_special_tokens=True,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_token_type_ids=False  # 关键修复
            ).to("cuda")
                    qed = self.qed_model(**inputs).item()
                    
                    # 2. 根据QED调整候选token概率
                    scores[batch_idx] *= (1 + qed * 3)  # QED越高，增强当前候选概率
                    continue
                    
                else:
                
                    continue
                
            try:
                # 模拟添加EOS后的完整SMILES
                clean_smi = current_smiles 
                # clean_smi = self._clean_smiles(candidate_smiles)
                print("clean_smiclean_smiclean_smiclean_smiclean_smiclean_smiclean_smi",clean_smi)
                
                # 验证有效性
                mol = Chem.MolFromSmiles(clean_smi)
                if not mol:
                    continue
                
                # 计算QED
                qed = QED.qed(mol)
                print(f"候选评估 =========================================| QED: {qed:.2f} | SMILES: {clean_smi}")
                
                if qed >= self.qed_threshold:
                    # 定位EOS在候选中的位置
                    eos_pos = (topk_tokens[batch_idx] == self.eos_token_id).nonzero()[0].item()
                    original_score = topk_scores[batch_idx, eos_pos].item()
                    
                    # 仅修改当前序列的EOS logits（关键逻辑）
                    scores[batch_idx, self.eos_token_id] = original_score * self.eos_boost
                                    # 将其他所有token的概率设为负无穷
                    # scores[batch_idx, :] = -float("inf")
                    # # 仅保留EOS的有效概率
                    # scores[batch_idx, self.eos_token_id] = 1.0  # 任意正值均可
                    
            except Exception as e:
                print(f"序列{batch_idx}处理异常: {str(e)}")

        return scores
class SmilesRequest(BaseModel):
    prompt: str
    num:int
@app.post("/predict")
async def predict(request:SmilesRequest ):
    generation_config = {
        "max_length": 80,
        # "num_beams": 1,
        "num_return_sequences": request.num,
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.2,
        "pad_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "logits_processor": [
            QEDGuidedEOSProcessor(
                tokenizer=tokenizer,
                eos_boost=50000.0,       # EOS增强倍数
                qed_threshold=0.6,    # QED阈值
                min_length=9 ,
                qed_model=qed_model ,
                qed_tokenizer=qed_tokenizer       # 最小生成长度
            )
        ]
    }
        # 4. 生成分子
    inputs = torch.tensor([[tokenizer.bos_token_id]])
    # inputs = tokenizer.encode(request.prompt, return_tensors='pt')
    outputs = model.generate(inputs, **generation_config)

    # 5. 解码结果
    generated_smiles = [
        tokenizer.decode(seq, skip_special_tokens=True) 
        for seq in outputs
    ]
    return generated_smiles


if __name__ == "__main__":


    qed_model, qed_tokenizer = init_qed_predictor("./saved_model/pytorch_model.bin", device="cuda")

    # 1. 加载并修复分词器
    checkpoint_path = "checkpoints/benchmark-10m"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint_path)

    # 手动添加特殊标记（如果缺失）
    if tokenizer.bos_token is None:
        special_tokens = {"bos_token": "<s>", "eos_token": "</s>"}
        tokenizer.add_special_tokens(special_tokens)

    # 2. 加载并调整模型
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
    model.resize_token_embeddings(len(tokenizer))

    # 3. 生成配置


    uvicorn.run(app, host="0.0.0.0", port=8000)

    # 在原有代码基础上添加重复统计
    # smiles_counter = {}  # 用于统计SMILES出现次数

    # valid_qeds = []
    # invalid_count = 0

    # for i, smi in enumerate(generated_smiles):
    #     print(f"\n分子 {i+1}:")
    #     print(f"  SMILES: {smi}")
        
    #     # 统计SMILES出现次数（在验证前统计原始生成）
    #     smiles_counter[smi] = smiles_counter.get(smi, 0) + 1
        
    #     try:
    #         mol = Chem.MolFromSmiles(smi)
    #         if mol is None:
    #             raise ValueError("Invalid SMILES")
            
    #         # 计算QED值
    #         qed = QED.qed(mol)
    #         valid_qeds.append(qed)
    #         print(f"  QED值: {qed:.2f}")
    #         print("  有效性: Valid")
    #     except Exception as e:
    #         invalid_count += 1
    #         print(f"  有效性: Invalid - {str(e)}")

    # # 统计结果
    # if valid_qeds:
    #     avg_qed = sum(valid_qeds) / len(valid_qeds)
    #     print(f"\n有效分子平均QED: {avg_qed:.2f}")
    # else:
    #     print("\n没有有效的分子可供计算QED。")

    # print(f"无效SMILES数目: {invalid_count}")
    
    # # 新增重复统计
    # unique_smiles = len(smiles_counter)
    # total_generated = len(generated_smiles)
    # duplicate_count = total_generated - unique_smiles
    
    # # 统计完全重复的条目（出现次数>1的）
    # duplicate_details = {k:v for k,v in smiles_counter.items() if v > 1}
    # duplicate_entries = sum(v-1 for v in duplicate_details.values())
    
    # print(f"\n重复统计:")
    # print(f"生成总数: {total_generated}")
    # print(f"唯一SMILES数: {unique_smiles}")
    # print(f"完全重复条目数: {len(duplicate_details)}")
    # print(f"重复实例总数: {duplicate_entries}")
    
    # # 可选：打印重复详情（最多显示5个）
    # if duplicate_details:
    #     print("\n重复最多的SMILES:")
    #     for smi, count in sorted(duplicate_details.items(), 
    #                            key=lambda x: x[1], reverse=True)[:5]:
    #         print(f" 出现{count}次: {smi}")




import requests

# 定义API的URL
url = "http://127.0.0.1:8000/predict"

# 创建请求数据
data = {
    "prompt": "C"
}

# 发送POST请求
response = requests.post(url, json=data)

# 检查响应状态码
if response.status_code == 200:
    # 解析响应内容
    generated_smiles = response.json()
    print("Generated SMILES:", generated_smiles)
else:
    print("Failed to get response. Status code:", response.status_code)
    print("Response:", response.text)