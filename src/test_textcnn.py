import os
import torch
import argparse
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

from data.data_reader import DataReader
from preprocessors.cnn_preprocessor import CNNPreprocessor
from models.text_cnn import TextCNN
from utils.print_utils import print_test_results, print_section_header

# 设置日志配置，删除时间信息
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # 只保留消息内容
)
logger = logging.getLogger(__name__)

def evaluate_model(model, test_texts, test_labels, preprocessor, batch_size, device):
    """
    评估模型
    Args:
        model: 模型
        test_texts: 测试文本
        test_labels: 测试标签
        preprocessor: 预处理器
        batch_size: 批次大小
        device: 设备
    Returns:
        predictions: 预测结果
        labels: 真实标签
        loss: 损失值
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    # 处理所有文本
    logger.info("Processing test texts...")
    processed_texts = preprocessor.process(test_texts)
    
    # 批量处理
    num_batches = (len(processed_texts) + batch_size - 1) // batch_size
    
    logger.info("Starting evaluation...")
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Testing"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(processed_texts))
            
            # 准备批次数据
            batch_texts = processed_texts[start_idx:end_idx]
            batch_labels = test_labels[start_idx:end_idx]
            
            # 转换为tensor
            inputs = torch.tensor(batch_texts, dtype=torch.long).to(device)
            labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # 获取预测结果
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / num_batches
    return all_preds, all_labels, avg_loss

def main():
    parser = argparse.ArgumentParser(description='Test TextCNN model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--max_seq_len', type=int, default=200, help='Maximum sequence length')
    parser.add_argument('--vocab_size', type=int, default=50000, help='Vocabulary size')
    parser.add_argument('--embed_dim', type=int, default=100, help='Word embedding dimension')
    parser.add_argument('--num_filters', type=int, default=100, help='Number of filters')
    parser.add_argument('--filter_sizes', type=str, default='3,4,5', help='Filter sizes')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    args = parser.parse_args()

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    data_reader = DataReader(base_path="data")
    
    # 测试中文和英文模型
    for language in ['en', 'cn']:
        print_section_header(language)
        try:
            # 加载测试数据
            logger.info("Loading test data...")
            test_texts, test_labels = data_reader.load_test_data(language=language)
            logger.info(f"Loaded {len(test_texts)} test samples")
            
            # 初始化预处理器
            logger.info("Initializing preprocessor...")
            preprocessor = CNNPreprocessor(
                max_sequence_length=args.max_seq_len,
                language=language
            )
            
            # 加载训练数据以构建词汇表
            logger.info("Loading training data for vocabulary...")
            train_texts, train_labels, _, _ = data_reader.load_train_val_data(language=language)
            preprocessor.build_vocab(train_texts)
            logger.info("Vocabulary built successfully")
            
            # 加载训练好的模型
            logger.info("Loading trained model...")
            model_path = os.path.join('checkpoints', language, 'textcnn.pt')
            checkpoint = torch.load(model_path, map_location=device)
            
            # 更新预处理器的词汇表
            preprocessor.word2idx = checkpoint['vocab']
            vocab_size = len(checkpoint['vocab'])
            
            # 使用保存的配置初始化模型
            config = checkpoint['config']
            model = TextCNN(
                vocab_size=vocab_size,
                embedding_dim=config['embedding_dim'],
                num_filters=config['num_filters'],
                filter_sizes=config['filter_sizes'],
                num_classes=2,
                dropout=args.dropout
            ).to(device)
            
            # 加载模型参数
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Model loaded successfully")
            
            # 评估模型
            predictions, labels, test_loss = evaluate_model(
                model, test_texts, test_labels, preprocessor, 
                args.batch_size, device
            )
            
            # 使用统一的输出格式
            print_test_results(
                language=language,
                loss=test_loss,
                predictions=predictions,
                labels=labels,
                model_name='textcnn'
            )
            
        except Exception as e:
            logger.error(f'Error testing {language} model: {str(e)}')
            continue

if __name__ == '__main__':
    main() 