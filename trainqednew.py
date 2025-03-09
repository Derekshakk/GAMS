# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, r2_score
# from transformers import BertTokenizer, BertConfig, BertModel

# # 配置参数
# class Config:
#     # 数据参数
#     data_path = "expanded_smiles1.csv"
#     max_length = 64  # 根据片段最大长度调整
    
#     # 模型参数
#     pretrained_model = "bert-base-uncased"  # 可替换为更小的模型
#     dropout = 0.1
#     hidden_size = 128
    
#     # 训练参数
#     batch_size = 64
#     lr = 2e-5
#     epochs = 50
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 自定义数据集
# class SMILESDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_len):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_len = max_len
        
#     def __len__(self):
#         return len(self.texts)
    
#     def __getitem__(self, idx):
#         text = str(self.texts[idx])
#         encoding = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'label': torch.tensor(self.labels[idx], dtype=torch.float)
#         }

# # 自定义Transformer模型
# class QEDPredictor(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(config.pretrained_model)
#         self.dropout = nn.Dropout(config.dropout)
#         self.regressor = nn.Sequential(
#             nn.Linear(self.bert.config.hidden_size, config.hidden_size),
#             nn.ReLU(),
#             nn.Linear(config.hidden_size, 1)
#         )
        
#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
#         pooled_output = outputs.last_hidden_state[:, 0, :]
#         output = self.dropout(pooled_output)
#         return self.regressor(output).squeeze()

# # 训练函数
# def train_model(model, dataloader, criterion, optimizer, device):
#     model.train()
#     total_loss = 0
#     for batch in dataloader:
#         optimizer.zero_grad()
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['label'].to(device)
        
#         outputs = model(input_ids, attention_mask)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     return total_loss / len(dataloader)

# # 评估函数
# def evaluate_model(model, dataloader, criterion, device):
#     model.eval()
#     total_loss = 0
#     predictions = []
#     true_labels = []
#     with torch.no_grad():
#         for batch in dataloader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['label'].to(device)
            
#             outputs = model(input_ids, attention_mask)
#             loss = criterion(outputs, labels)
#             total_loss += loss.item()
            
#             predictions.extend(outputs.cpu().numpy())
#             true_labels.extend(labels.cpu().numpy())
#     return total_loss / len(dataloader), predictions, true_labels

# # 主流程
# if __name__ == "__main__":
#     # 数据加载
#     df = pd.read_csv(Config.data_path)
#     texts = df['smiles'].values
#     labels = df['qed'].values
    
#     # 数据分割
#     train_texts, test_texts, train_labels, test_labels = train_test_split(
#         texts, labels, test_size=0.2, random_state=42
#     )
    
#     # 初始化tokenizer
#     tokenizer = BertTokenizer.from_pretrained(Config.pretrained_model)
    
#     # 创建数据集
#     train_dataset = SMILESDataset(train_texts, train_labels, tokenizer, Config.max_length)
#     test_dataset = SMILESDataset(test_texts, test_labels, tokenizer, Config.max_length)
    
#     # 创建数据加载器
#     train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=Config.batch_size)
    
#     # 初始化模型
#     model = QEDPredictor(Config).to(Config.device)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=Config.lr)
    
#     # 训练循环
#     best_mae = float('inf')
#     for epoch in range(Config.epochs):
#         train_loss = train_model(model, train_loader, criterion, optimizer, Config.device)
#         test_loss, preds, truths = evaluate_model(model, test_loader, criterion, Config.device)
        
#         mae = mean_absolute_error(truths, preds)
#         r2 = r2_score(truths, preds)
        
#         print(f"Epoch {epoch+1}/{Config.epochs}")
#         print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
#         print(f"MAE: {mae:.4f} | R²: {r2:.4f}\n")
        
#         if mae < best_mae:
#             best_mae = mae
#             torch.save(model.state_dict(), "best_model1.pth")

#     print(f"Best MAE: {best_mae:.4f}")




















import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from transformers import BertTokenizer, BertModel
import os
import json
from tqdm.auto import tqdm
import matplotlib.pyplot as plt  # 新增matplotlib导入

# 配置参数
class Config:
    # 数据参数
    data_path = "expanded_smiles1.csv"
    max_length = 128
    
    # 模型参数
    pretrained_model = "bert-base-uncased"
    dropout = 0.1
    hidden_size = 128
    
    # 训练参数
    batch_size = 64
    lr = 2e-5
    epochs = 40
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 保存参数
    save_dir = "saved_model"
    metrics_file = "training_metrics.txt"
    metrics_figure = "training_metrics.png"

# 自定义数据集（保持不变）
class SMILESDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }

# 自定义模型（保持不变）
class QEDPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.pretrained_model)
        self.dropout = nn.Dropout(config.dropout)
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.dropout(pooled_output)
        return self.regressor(output).squeeze()


def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
        labels = batch['label'].to(device)
        
        outputs = model(**inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)

# 带进度条的评估函数
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)
            
            outputs = model(**inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix(loss=loss.item())
            
    return total_loss / len(dataloader), predictions, true_labels

# 保存完整模型
def save_model(model, tokenizer, config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存模型权重
    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    
    # 保存分词器
    tokenizer.save_pretrained(save_dir)
    
    # 保存配置
    config_dict = {
        "max_length": config.max_length,
        "hidden_size": config.hidden_size,
        "dropout": config.dropout,
        "pretrained_model": config.pretrained_model
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config_dict, f)

# 主流程修改
if __name__ == "__main__":
    # 初始化配置
    config = Config()
    os.makedirs(config.save_dir, exist_ok=True)  # 确保保存目录存在
    
    # 初始化指标文件
    metrics_path = os.path.join(config.save_dir, config.metrics_file)
    with open(metrics_path, "w") as f:
        f.write("epoch,train_loss,test_loss,mae,r2\n")
    
    # 数据加载与分割
    df = pd.read_csv(config.data_path)
    texts = df['smiles'].values
    labels = df['qed'].values
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # 初始化分词器
    tokenizer = BertTokenizer.from_pretrained(config.pretrained_model)
    
    # 创建数据集和数据加载器
    train_dataset = SMILESDataset(train_texts, train_labels, tokenizer, config.max_length)
    test_dataset = SMILESDataset(test_texts, test_labels, tokenizer, config.max_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    # 初始化模型和训练参数
    model = QEDPredictor(config).to(config.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    
    # 初始化指标存储列表
    train_losses = []
    test_losses = []
    mae_scores = []
    r2_scores = []
    
    # 训练循环
    best_mae = float('inf')
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        
        # 训练阶段
        train_loss = train_model(model, train_loader, criterion, optimizer, config.device)
        
        # 评估阶段
        test_loss, preds, truths = evaluate_model(model, test_loader, criterion, config.device)
        
        # 计算指标
        mae = mean_absolute_error(truths, preds)
        r2 = r2_score(truths, preds)
        
        # 存储指标
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        mae_scores.append(mae)
        r2_scores.append(r2)
        
        # 写入指标文件
        with open(metrics_path, "a") as f:
            f.write(f"{epoch+1},{train_loss:.4f},{test_loss:.4f},{mae:.4f},{r2:.4f}\n")
        
        # 打印结果
        print(f"\nTrain Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        print(f"MAE: {mae:.4f} | R²: {r2:.4f}")
        
        # 保存最佳模型
        if mae < best_mae:
            best_mae = mae
            save_model(model, tokenizer, config, config.save_dir)
            print(f"New best model saved to {config.save_dir}")

    # 训练结束后绘制指标变化图
    plt.figure(figsize=(12, 8))
    
    # 训练损失
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    
    # 测试损失
    plt.subplot(2, 2, 2)
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test Loss')
    
    # MAE
    plt.subplot(2, 2, 3)
    plt.plot(mae_scores, label='MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Mean Absolute Error')
    
    # R²
    plt.subplot(2, 2, 4)
    plt.plot(r2_scores, label='R²')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.title('R² Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.save_dir, config.metrics_figure))
    plt.show()

    print(f"\nTraining Complete! Best MAE: {best_mae:.4f}")