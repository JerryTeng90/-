import os
import torch
import argparse
import logging
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm

from data.data_reader import DataReader
from models.bert_model import BertForSentiment
from preprocessors.bert_preprocessor import BertPreprocessor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_dataloader(texts, labels, preprocessor, batch_size, shuffle=True):
    """
    创建数据加载器
    """
    # 处理文本数据
    encoded_dict, labels_tensor = preprocessor.process(texts, labels)
    
    # 创建数据集
    dataset = TensorDataset(
        encoded_dict['input_ids'],
        encoded_dict['attention_mask'],
        encoded_dict['token_type_ids'],
        labels_tensor
    )
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )
    
    return dataloader

def evaluate_model(model, dataloader, device):
    """
    评估模型性能
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, token_type_ids, labels = [b.to(device) for b in batch]
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    
    # 计算分类报告
    report = classification_report(
        all_labels,
        all_preds,
        target_names=['Negative', 'Positive'],
        digits=4
    )
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, report, conf_matrix, all_preds, all_labels

def train(args):
    """
    训练模型
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 加载数据
    data_reader = DataReader()
    train_texts, train_labels = data_reader.load_train_data(args.language)
    val_texts, val_labels = data_reader.load_validation_data(args.language)
    
    # 初始化预处理器
    preprocessor = BertPreprocessor(
        pretrained_model_name=args.pretrained_model_name,
        max_length=args.max_length,
        language=args.language
    )
    
    # 创建数据加载器
    train_dataloader = create_dataloader(
        train_texts, train_labels,
        preprocessor, args.batch_size
    )
    val_dataloader = create_dataloader(
        val_texts, val_labels,
        preprocessor, args.batch_size,
        shuffle=False
    )
    
    # 初始化模型
    model = BertForSentiment(
        pretrained_model_name=args.pretrained_model_name,
        num_classes=2,
        dropout=args.dropout,
        freeze_bert=args.freeze_bert
    ).to(device)
    
    # 定义优化器和损失函数
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
    # 训练循环
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_path = os.path.join(
        'checkpoints',
        args.language,
        'bert.pt'
    )
    
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        train_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{args.num_epochs}')
        
        for batch in train_bar:
            input_ids, attention_mask, token_type_ids, labels = [b.to(device) for b in batch]
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # 更新参数
            optimizer.step()
            
            # 更新进度条
            train_bar.set_postfix({'loss': loss.item()})
        
        # 计算平均训练损失
        avg_train_loss = total_loss / len(train_dataloader)
        
        # 验证
        val_loss, val_report, val_conf_matrix, _, _ = evaluate_model(
            model, val_dataloader, device
        )
        
        # 打印训练信息
        logger.info(f'\nEpoch {epoch+1}/{args.num_epochs}:')
        logger.info(f'Average training loss: {avg_train_loss:.4f}')
        logger.info(f'Validation loss: {val_loss:.4f}')
        logger.info('\nValidation Report:')
        logger.info(f'\n{val_report}')
        logger.info('\nConfusion Matrix:')
        logger.info(f'\n{val_conf_matrix}')
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            
            # 确保目录存在
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'preprocessor_config': {
                    'pretrained_model_name': args.pretrained_model_name,
                    'max_length': args.max_length
                }
            }, best_model_path)
            logger.info(f'Saved best model to {best_model_path}')
        else:
            early_stopping_counter += 1
            
        # 早停
        if early_stopping_counter >= args.patience:
            logger.info(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    logger.info('Training completed!')
    return best_model_path

def main():
    parser = argparse.ArgumentParser(description='Train BERT model for sentiment classification')
    
    # 模型参数
    parser.add_argument('--pretrained_model_name', type=str, default='bert-base-multilingual-cased',
                        help='Pretrained BERT model name')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--freeze_bert', action='store_true',
                        help='Freeze BERT parameters')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                        help='Maximum gradient norm')
    
    args = parser.parse_args()
    
    # 训练中英文模型
    for language in ['en', 'cn']:
        logger.info(f'\nTraining {language} model...')
        args.language = language
        train(args)

if __name__ == '__main__':
    main() 