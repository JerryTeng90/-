import os
import torch
import torch.nn as nn
import argparse
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
import numpy as np

from data.data_reader import DataReader
from preprocessors.lstm_preprocessor import LSTMPreprocessor
from models.bilstm_attention import BiLSTMAttention
from train_bilstm import SentimentDataset
from utils.print_utils import print_test_results, print_section_header

# 设置日志配置，删除时间信息
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # 只保留消息内容
)
logger = logging.getLogger(__name__)

def evaluate_model(model, test_loader, criterion, device):
    """
    评估模型
    Args:
        model: 模型
        test_loader: 测试数据加载器
        criterion: 损失函数
        device: 设备
    Returns:
        avg_loss: 平均损失
        all_preds: 所有预测结果
        all_labels: 所有真实标签
        attention_weights: 注意力权重
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    attention_weights = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            # 获取数据
            sequences, attention_masks, labels = [x.to(device) for x in batch]
            
            # 前向传播
            outputs, attn_weights = model(sequences, attention_masks, return_attention=True)
            loss = criterion(outputs, labels)
            
            # 获取预测结果
            preds = torch.argmax(outputs, dim=1)
            
            # 更新统计
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            attention_weights.extend(attn_weights.cpu().numpy())
    
    return total_loss / len(test_loader), all_preds, all_labels, attention_weights

def test(args):
    """
    测试模型
    Args:
        args: 命令行参数
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载测试数据
    data_reader = DataReader(base_path="data")
    test_texts, test_labels = data_reader.load_test_data(args.language)
    
    # 初始化预处理器
    preprocessor = LSTMPreprocessor(
        max_sequence_length=args.max_seq_len,
        vocab_size=args.vocab_size,
        language=args.language
    )
    
    # 构建词汇表
    train_texts, train_labels, _, _ = data_reader.load_train_val_data(args.language)
    preprocessor.build_vocab(train_texts)
    
    # 创建测试数据集和加载器
    test_dataset = SentimentDataset(test_texts, test_labels, preprocessor)
    test_loader = DataLoader(
        test_dataset,
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
        dropout=0.0,  # 测试时不使用dropout
        padding_idx=preprocessor.get_pad_idx()
    ).to(device)
    
    # 加载训练好的模型
    model_path = os.path.join('checkpoints', args.language, 'bilstm.pt')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 初始化损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 评估模型
    test_loss, test_preds, test_labels, attention_weights = evaluate_model(
        model, test_loader, criterion, device
    )
    
    # 使用统一的输出格式
    print_test_results(
        language=args.language,
        loss=test_loss,
        predictions=test_preds,
        labels=test_labels,
        model_name='bilstm'
    )
    
    # 分析错误预测的样本
    errors = []
    for text, true_label, pred_label, attn_weight in zip(
        test_texts, test_labels, test_preds, attention_weights
    ):
        if true_label != pred_label:
            # 获取最重要的词
            tokens = preprocessor.tokenize(text)
            if len(tokens) > 0:  # 确保文本不为空
                top_k = min(5, len(tokens))  # 取前5个最重要的词
                top_indices = np.argsort(attn_weight[:len(tokens)])[-top_k:]
                important_words = [tokens[i] for i in top_indices]
                
                errors.append({
                    'text': text,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'important_words': important_words
                })
    
    # 保存错误分析结果
    error_dir = os.path.join('results', 'bilstm')
    os.makedirs(error_dir, exist_ok=True)
    with open(os.path.join(error_dir, f'error_analysis_{args.language}.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Total errors: {len(errors)}\n\n')
        for i, error in enumerate(errors, 1):
            f.write(f'Error {i}:\n')
            f.write(f'Text: {error["text"]}\n')
            f.write(f'True label: {error["true_label"]}\n')
            f.write(f'Predicted label: {error["pred_label"]}\n')
            f.write(f'Important words: {", ".join(error["important_words"])}\n')
            f.write('\n')

def main():
    parser = argparse.ArgumentParser(description='Test BiLSTM model')
    
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
    
    # 测试参数
    parser.add_argument('--batch_size', type=int, default=128,
                      help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                      help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # 测试中英文模型
    for language in ['en', 'cn']:
        args.language = language
        print_section_header(language)
        try:
            test(args)
        except Exception as e:
            logger.error(f'Error testing {language} model: {str(e)}')
            continue

if __name__ == '__main__':
    main() 