import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from typing import List, Tuple, Optional
import nltk

print("Initializing NLTK resources...")
# 下载所有必要的NLTK数据
resources = [
    'punkt',
    'averaged_perceptron_tagger',
    'wordnet',
    'omw-1.4'
]
for resource in resources:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        print(f'Downloading NLTK {resource}...')
        nltk.download(resource)
print("NLTK resources initialized.")

from data.data_reader import DataReader
from preprocessors.lstm_preprocessor import LSTMPreprocessor
from models.bilstm_attention import BiLSTMAttention

class SentimentDataset(Dataset):
    """
    情感分类数据集
    """
    def __init__(self, texts: List[str], labels: List[int], preprocessor: LSTMPreprocessor):
        """
        初始化数据集
        Args:
            texts: 文本列表
            labels: 标签列表
            preprocessor: 预处理器
        """
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        
        # 预处理所有数据
        self.sequences, self.attention_masks = preprocessor.process(texts)
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取一个样本
        Args:
            idx: 样本索引
        Returns:
            sequence: 序列张量
            attention_mask: 注意力掩码张量
            label: 标签张量
        """
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        attention_mask = torch.tensor(self.attention_masks[idx], dtype=torch.bool)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence, attention_mask, label

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_grad_norm: float = 1.0  # 添加梯度裁剪阈值
) -> float:
    """
    训练一个epoch
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 设备
        epoch: 当前epoch
        max_grad_norm: 梯度裁剪阈值
    Returns:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch in progress_bar:
        # 获取数据
        sequences, attention_masks, labels = [x.to(device) for x in batch]
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(sequences, attention_masks)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        # 计算准确率
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # 更新统计
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    return total_loss / len(train_loader)

def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, List[int], List[int]]:
    """
    评估模型
    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 设备
    Returns:
        avg_loss: 平均损失
        all_preds: 所有预测结果
        all_labels: 所有真实标签
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating'):
            # 获取数据
            sequences, attention_masks, labels = [x.to(device) for x in batch]
            
            # 前向传播
            outputs = model(sequences, attention_masks)
            loss = criterion(outputs, labels)
            
            # 获取预测结果
            preds = torch.argmax(outputs, dim=1)
            
            # 更新统计
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(val_loader), all_preds, all_labels

def train(args):
    """
    训练模型
    Args:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建保存目录
    os.makedirs(os.path.join('checkpoints', args.language), exist_ok=True)
    
    # 加载数据
    data_reader = DataReader(base_path="data")
    train_texts, train_labels, val_texts, val_labels = data_reader.load_train_val_data(args.language)
    
    print(f'Loaded {len(train_texts)} training samples')
    print(f'Loaded {len(val_texts)} validation samples')
    
    # 初始化预处理器
    preprocessor = LSTMPreprocessor(
        max_sequence_length=args.max_seq_len,
        vocab_size=args.vocab_size,
        language=args.language
    )
    
    # 构建词汇表
    print('Building vocabulary...')
    preprocessor.build_vocab(train_texts)
    
    # 创建数据集
    train_dataset = SentimentDataset(train_texts, train_labels, preprocessor)
    val_dataset = SentimentDataset(val_texts, val_labels, preprocessor)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True  # 丢弃不完整的批次
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 初始化模型
    model = BiLSTMAttention(
        vocab_size=preprocessor.get_vocab_size(),
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        padding_idx=preprocessor.get_pad_idx()
    ).to(device)
    
    # 使用 weight decay 的 AdamW 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # 余弦退火学习率调度器
    total_steps = len(train_loader) * args.num_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate,
        total_steps=total_steps,
        pct_start=0.1,  # 预热阶段占比
        anneal_strategy='cos',
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,  # 初始学习率 = max_lr/25
        final_div_factor=10000.0  # 最终学习率 = max_lr/10000
    )
    
    # 带权重的交叉熵损失
    if args.class_weights:
        # 计算类别权重
        label_counts = np.bincount(train_labels)
        total = len(train_labels)
        weights = torch.FloatTensor([total / (2 * count) for count in label_counts]).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    patience = args.patience
    patience_counter = 0
    
    for epoch in range(1, args.num_epochs + 1):
        # 训练一个epoch
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            max_grad_norm=args.max_grad_norm
        )
        
        # 评估
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 打印评估结果
        print(f'\nEpoch {epoch}:')
        print(f'Learning Rate: {current_lr:.6f}')
        print(f'Train Loss: {train_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}')
        print('\nClassification Report:')
        print(classification_report(val_labels, val_preds, target_names=['Negative', 'Positive']))
        
        # 计算F1分数
        val_report = classification_report(val_labels, val_preds, output_dict=True)
        val_f1 = val_report['macro avg']['f1-score']
        
        # 保存最佳模型（同时考虑F1分数和损失）
        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_loss < best_val_loss):
            best_val_f1 = val_f1
            best_val_loss = val_loss
            patience_counter = 0
            
            # 保存完整的模型状态
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1': val_f1,
                'val_loss': val_loss,
                'vocab': preprocessor.word2idx
            }
            torch.save(
                checkpoint,
                os.path.join('checkpoints', args.language, 'bilstm.pt')
            )
            print(f'Saved new best model with F1: {val_f1:.4f}')
        else:
            patience_counter += 1
            print(f'F1 did not improve. Patience: {patience_counter}/{patience}')
        
        # 早停（同时考虑轮数和性能）
        if patience_counter >= patience and epoch > args.min_epochs:
            print('Early stopping triggered')
            break

def main():
    parser = argparse.ArgumentParser(description='Train BiLSTM model')
    
    # 数据参数
    parser.add_argument('--max_seq_len', type=int, default=200,
                      help='Maximum sequence length')
    parser.add_argument('--vocab_size', type=int, default=50000,
                      help='Vocabulary size')
    
    # 模型参数
    parser.add_argument('--embedding_dim', type=int, default=300,
                      help='Word embedding dimension')
    parser.add_argument('--hidden_size', type=int, default=256,
                      help='LSTM hidden size')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                      help='Dropout rate')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30,
                      help='Maximum number of epochs')
    parser.add_argument('--min_epochs', type=int, default=5,
                      help='Minimum number of epochs before early stopping')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Maximum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                      help='Weight decay for AdamW')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                      help='Maximum gradient norm for clipping')
    parser.add_argument('--patience', type=int, default=5,
                      help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=0,
                      help='Number of data loading workers')
    parser.add_argument('--class_weights', action='store_true',
                      help='Use class weights in loss function')
    
    args = parser.parse_args()
    
    # 训练英文模型
    print("\n=== Training English Model ===")
    args.language = 'en'
    train(args)
    
    # 训练中文模型
    print("\n=== Training Chinese Model ===")
    args.language = 'cn'
    train(args)

if __name__ == '__main__':
    main() 