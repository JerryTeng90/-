import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm
import time

class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 300, 
                 filter_sizes: List[int] = [2, 3, 4], num_filters: int = 100,
                 dropout: float = 0.5, num_classes: int = 2):
        """
        初始化TextCNN模型
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            filter_sizes: 卷积核尺寸列表
            num_filters: 每种尺寸的卷积核数量
            dropout: Dropout比率
            num_classes: 类别数
        """
        super(TextCNN, self).__init__()
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) 
            for k in filter_sizes
        ])
        
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        
        # 全连接层
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
        # 保存参数
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.embedding_dim = embedding_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入张量，shape为(batch_size, sequence_length)
        Returns:
            输出张量，shape为(batch_size, num_classes)
        """
        # 词嵌入，shape: (batch_size, sequence_length, embedding_dim)
        x = self.embedding(x)
        
        # 添加通道维度，shape: (batch_size, 1, sequence_length, embedding_dim)
        x = x.unsqueeze(1)
        
        # 对不同尺寸的卷积核分别进行卷积和池化操作
        pooled_outputs = []
        for conv in self.convs:
            # 卷积，shape: (batch_size, num_filters, sequence_length-filter_size+1, 1)
            conv_out = F.relu(conv(x))
            
            # 最大池化，shape: (batch_size, num_filters, 1, 1)
            pooled = F.max_pool2d(conv_out, (conv_out.shape[2], 1))
            
            # 去除多余维度，shape: (batch_size, num_filters)
            pooled = pooled.squeeze(3).squeeze(2)
            
            pooled_outputs.append(pooled)
        
        # 拼接所有池化输出，shape: (batch_size, num_filters * len(filter_sizes))
        cat = torch.cat(pooled_outputs, dim=1)
        
        # Dropout
        cat = self.dropout(cat)
        
        # 全连接层，shape: (batch_size, num_classes)
        logits = self.fc(cat)
        
        return logits

class TextCNNTrainer:
    def __init__(self, model: TextCNN, device: torch.device, learning_rate: float = 0.001):
        """
        初始化训练器
        Args:
            model: TextCNN模型
            device: 设备
            learning_rate: 学习率
        """
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_step(self, batch_texts: torch.Tensor, batch_labels: torch.Tensor) -> Tuple[float, float]:
        """
        训练一个批次
        Args:
            batch_texts: 文本批次
            batch_labels: 标签批次
        Returns:
            (loss, accuracy) 损失和准确率
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        batch_texts = batch_texts.to(self.device)
        batch_labels = batch_labels.to(self.device)
        
        # 使用混合精度训练
        with torch.cuda.amp.autocast():
            outputs = self.model(batch_texts)
            loss = self.criterion(outputs, batch_labels)
        
        # 反向传播
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # 计算准确率
        predictions = torch.argmax(outputs, dim=1)
        correct = (predictions == batch_labels).sum().item()
        accuracy = correct / len(batch_labels)
        
        return loss.item(), accuracy
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """
        评估模型
        Args:
            data_loader: 数据加载器
        Returns:
            (avg_loss, accuracy) 平均损失和准确率
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_texts, batch_labels in data_loader:
                batch_texts = batch_texts.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                with torch.cuda.amp.autocast():
                    outputs = self.model(batch_texts)
                    loss = self.criterion(outputs, batch_labels)
                
                total_loss += loss.item() * len(batch_labels)
                
                predictions = torch.argmax(outputs, dim=1)
                total_correct += (predictions == batch_labels).sum().item()
                total_samples += len(batch_labels)
        
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
             num_epochs: int = 10) -> Dict[str, List[float]]:
        """
        训练模型
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
        Returns:
            训练历史
        """
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(num_epochs):
            # 训练阶段
            train_loss = 0.0
            train_acc = 0.0
            train_steps = 0
            
            train_iterator = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
            for batch_texts, batch_labels in train_iterator:
                loss, accuracy = self.train_step(batch_texts, batch_labels)
                train_loss += loss
                train_acc += accuracy
                train_steps += 1
                
                # 更新进度条
                train_iterator.set_postfix({
                    'loss': f'{train_loss / train_steps:.4f}',
                    'acc': f'{train_acc / train_steps:.4f}'
                })
            
            # 计算训练集平均指标
            train_loss /= train_steps
            train_acc /= train_steps
            
            # 记录训练集指标
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # 验证阶段
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                print(f'\nEpoch {epoch + 1}/{num_epochs}:')
                print(f'train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f}')
                print(f'val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}\n')
            else:
                print(f'\nEpoch {epoch + 1}/{num_epochs}:')
                print(f'train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f}\n')
        
        return history

