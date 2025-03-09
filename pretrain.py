import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from transformers import AdamW, get_linear_schedule_with_warmup
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import os

# 设置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class CustomTokenizer:
    @staticmethod
    def train_new_tokenizer(texts, vocab_size=5000, min_frequency=2, padding=True):
        """从头训练新的tokenizer"""
        # 初始化tokenizer
        tokenizer = Tokenizer(models.BPE())
        
        # 设置预分词器
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        
        # 配置训练器
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=["<|endoftext|>", "<|pad|>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )
        
        # 训练tokenizer
        logger.info("Starting tokenizer training...")
        tokenizer.train_from_iterator(texts, trainer=trainer)
        
        # 设置后处理器
        if padding:
            tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        
        # 设置解码器
        tokenizer.decoder = decoders.ByteLevel()
        
        # 转换为transformers格式
        wrapped_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            pad_token="<|pad|>"
        )
        
        return wrapped_tokenizer

class SMILESDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.smiles = []
        
        # 从CSV文件读取SMILES数据
        logger.info(f"Reading SMILES data from {file_path}")
        df = pd.read_csv(file_path)
        
        # 假设SMILES在名为'smiles'的列中
        if 'smiles' not in df.columns:
            raise ValueError("CSV文件中未找到'smiles'列")
        
        self.smiles = df['smiles'].dropna().tolist()  # 去掉空值
        logger.info(f"Loaded {len(self.smiles)} SMILES strings")

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles[idx]
        
        # 编码SMILES
        encoding = self.tokenizer(
            smiles,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

class GPT2Trainer:
    def __init__(
        self,
        train_file,
        output_dir,
        vocab_size=5000,
        max_length=128,
        batch_size=32,
        epochs=10,
        learning_rate=5e-5,
        warmup_steps=1000,
        save_steps=1000,
        model_size='small'  # 可选: 'small', 'medium', 'large'
    ):
        self.train_file = train_file
        self.output_dir = output_dir
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.save_steps = save_steps
        self.model_size = model_size
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def _get_model_config(self):
        """根据模型大小返回相应的配置"""
        configs = {
            'small': {
                'n_layer': 6,
                'n_head': 8,
                'n_embd': 512
            },
            'medium': {
                'n_layer': 12,
                'n_head': 12,
                'n_embd': 768
            },
            'large': {
                'n_layer': 24,
                'n_head': 16,
                'n_embd': 1024
            }
        }
        
        config = configs.get(self.model_size, configs['small'])
        return GPT2Config(
            vocab_size=self.vocab_size,
            n_positions=self.max_length,
            n_ctx=self.max_length,
            **config
        )

    def train(self):

        df = pd.read_csv(self.train_file)
        if 'smiles' not in df.columns:
            raise ValueError("'smiles'Not found in the CSV file")
        
        texts = df['smiles'].dropna().tolist()  
        

        logger.info("Training new tokenizer...")
        tokenizer = CustomTokenizer.train_new_tokenizer(
            texts,
            vocab_size=self.vocab_size
        )
        
        tokenizer_path = os.path.join(self.output_dir, 'tokenizer')
        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_path)
        logger.info(f"Saved tokenizer to {tokenizer_path}")
        
        logger.info("Initializing new GPT-2 model...")
        config = self._get_model_config()
        model = GPT2LMHeadModel(config)
        model.to(self.device)
        
        # 准备数据集
        dataset = SMILESDataset(self.train_file, tokenizer, self.max_length)
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        # 准备优化器和调度器
        optimizer = AdamW(model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        # 开始训练
        logger.info("Starting training...")
        global_step = 0
        
        for epoch in range(self.epochs):
            model.train()
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for batch in progress_bar:
                # 准备输入数据
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # 前向传播
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # 反向传播
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # 更新进度条
                progress_bar.set_postfix({'loss': loss.item()})
                
                global_step += 1
                
                # 保存检查点
                if global_step % self.save_steps == 0:
                    checkpoint_dir = os.path.join(
                        self.output_dir,
                        f'checkpoint-{global_step}'
                    )
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    
                    model.save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                    
                    logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
            # 输出每个epoch的平均损失
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")
        
        # 保存最终模型
        logger.info("Saving final model...")
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)
        
        return model, tokenizer

class SMILESGenerator:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = GPT2LMHeadModel.from_pretrained(model_path).to(self.device)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        self.model.eval()

    def generate(
        self,
        prompt="",
        max_length=128,
        num_samples=5,
        temperature=0.8,
        top_p=0.95,
        top_k=50
    ):
        generated_smiles = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_smiles.append(generated_text)
        
        return generated_smiles

def main():
    # 训练配置
    config = {
        'train_file': './output22.csv',  # 你的CSV数据文件路径
        'output_dir': 'gpt2-smiles-model',        # 模型保存路径
        'vocab_size': 5000,                       # 词汇表大小
        'max_length': 128,                        # 最大序列长度
        'batch_size': 32,                         # 批次大小
        'epochs': 10,                             # 训练轮数
        'learning_rate': 5e-5,                    # 学习率
        'warmup_steps': 1000,                     # 预热步数
        'save_steps': 1000,                       # 保存检查点间隔
        'model_size': 'small'                     # 模型大小
    }
    
    # 训练模型
    trainer = GPT2Trainer(**config)
    model, tokenizer = trainer.train()
    
    # 生成示例
    generator = SMILESGenerator(config['output_dir'])
    generated = generator.generate(
        prompt="C",
        num_samples=5
    )
    
    # 打印生成的SMILES
    for i, smiles in enumerate(generated, 1):
        print(f"Generated SMILES {i}: {smiles}")

if __name__ == "__main__":
    main()
