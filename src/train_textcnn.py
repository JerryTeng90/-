import os
import torch
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

from data.data_reader import DataReader
from preprocessors.cnn_preprocessor import CNNPreprocessor
from models.text_cnn import TextCNN, TextCNNTrainer

class SentimentDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], preprocessor: CNNPreprocessor):
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
        
        # 预处理文本
        self.processed_texts = []
        batch_size = 100
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        with tqdm(total=len(texts), desc="Processing texts", ncols=100) as pbar:
            for i in range(0, len(texts), batch_size):
                end_idx = min(i + batch_size, len(texts))
                batch_texts = texts[i:end_idx]
                batch_processed = self.preprocessor.process(batch_texts)
                self.processed_texts.extend(batch_processed)
                
                # 更新进度条
                pbar.update(len(batch_texts))
                
                # 主动清理内存
                if i % 1000 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
        self.processed_texts = np.array(self.processed_texts)
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个样本
        Args:
            idx: 索引
        Returns:
            (processed_text, label)元组
        """
        return (
            torch.tensor(self.processed_texts[idx], dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

def train(data_reader: DataReader, preprocessor: CNNPreprocessor,
          model: TextCNN, trainer: TextCNNTrainer,
          train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
          num_epochs: int = 10, save_dir: str = 'checkpoints') -> Dict:
    """
    训练模型
    Args:
        data_reader: 数据读取器
        preprocessor: 预处理器
        model: TextCNN模型
        trainer: 训练器
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        save_dir: 模型保存目录
    Returns:
        训练历史
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练模型
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs
    )
    
    # 保存模型和预处理器
    model_path = os.path.join(save_dir, 'textcnn.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': preprocessor.word2idx,
        'config': {
            'embedding_dim': model.embedding_dim,
            'filter_sizes': model.filter_sizes,
            'num_filters': model.num_filters
        }
    }, model_path)
    
    return history

def train_single_language(language: str, base_dir: str, data_dir: str,
                     max_seq_len: int = 128, embedding_dim: int = 100,
                     batch_size: int = 32, num_epochs: int = 10,
                     learning_rate: float = 0.001) -> Tuple[float, float]:
    """
    训练单个语言的模型
    Args:
        language: 语言类型 ('cn' 或 'en')
        base_dir: 项目根目录
        data_dir: 数据目录
        max_seq_len: 最大序列长度
        embedding_dim: 词嵌入维度
        batch_size: 批次大小
        num_epochs: 训练轮数
        learning_rate: 学习率
    Returns:
        (train_acc, val_acc) 训练集和验证集的准确率
    """
    print(f'\n{"="*20} Training {language.upper()} Model {"="*20}')
    
    # 设置检查点目录
    checkpoint_dir = os.path.join(base_dir, 'checkpoints', language)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # 初始化数据读取器和预处理器
    data_reader = DataReader(base_path=data_dir)
    preprocessor = CNNPreprocessor(
        max_sequence_length=max_seq_len,
        min_freq=5,
        language=language
    )
    
    # 加载数据
    print('Loading data...')
    train_texts, train_labels, val_texts, val_labels = data_reader.load_train_val_data(language=language)
    print(f'Raw data loaded:')
    print(f'- Training samples: {len(train_texts)}')
    print(f'- Validation samples: {len(val_texts)}')
    
    # 构建词汇表
    print('Building vocabulary...')
    preprocessor.build_vocab(train_texts)
    print(f'Vocabulary size: {preprocessor.get_vocab_size()}')
    
    # 创建数��集
    print('Creating datasets...')
    train_dataset = SentimentDataset(train_texts, train_labels, preprocessor)
    val_dataset = SentimentDataset(val_texts, val_labels, preprocessor)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 初始化模型
    model = TextCNN(
        vocab_size=preprocessor.get_vocab_size(),
        embedding_dim=embedding_dim,
        num_filters=50
    ).to(device)
    
    # 初始化训练器
    trainer = TextCNNTrainer(
        model=model,
        device=device,
        learning_rate=learning_rate
    )
    
    # 训练模型
    print('Starting training...')
    history = train(
        data_reader=data_reader,
        preprocessor=preprocessor,
        model=model,
        trainer=trainer,
        train_loader=train_loader,
        val_loader=val_loader,
        save_dir=checkpoint_dir,
        num_epochs=num_epochs
    )
    
    # 获取最终准确率
    _, train_acc = trainer.evaluate(train_loader)
    _, val_acc = trainer.evaluate(val_loader)
    
    print(f'\n{language.upper()} Model Results:')
    print(f'Training Accuracy: {train_acc:.4f}')
    print(f'Validation Accuracy: {val_acc:.4f}')
    
    return train_acc, val_acc

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing device: {device}')
    if device.type == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    # 设置路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    # 训练参数
    params = {
        'max_seq_len': 128,
        'embedding_dim': 100,
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 0.001
    }
    
    # 先训练英文模型
    en_train_acc, en_val_acc = train_single_language(
        language='en',
        base_dir=base_dir,
        data_dir=data_dir,
        **params
    )
    
    # 再训练中文模型
    cn_train_acc, cn_val_acc = train_single_language(
        language='cn',
        base_dir=base_dir,
        data_dir=data_dir,
        **params
    )
    
    # 输出总体结果
    print('\n' + '='*50)
    print('Overall Results:')
    print('-'*50)
    print('English Model:')
    print(f'- Training Accuracy: {en_train_acc:.4f}')
    print(f'- Validation Accuracy: {en_val_acc:.4f}')
    print('\nChinese Model:')
    print(f'- Training Accuracy: {cn_train_acc:.4f}')
    print(f'- Validation Accuracy: {cn_val_acc:.4f}')
    print('\nAverage Performance:')
    print(f'- Average Training Accuracy: {(cn_train_acc + en_train_acc) / 2:.4f}')
    print(f'- Average Validation Accuracy: {(cn_val_acc + en_val_acc) / 2:.4f}')
    print('='*50)

if __name__ == '__main__':
    main()

