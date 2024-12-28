import os
import torch
import logging
import argparse
from tqdm import tqdm
import numpy as np
import nltk
import tensorflow as tf

# 设置tensorflow日志级别
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from data.data_reader import DataReader
from models.bert_model import BertForSentiment
from preprocessors.bert_preprocessor import BertPreprocessor
from utils.print_utils import print_test_results, print_section_header

# 修改日志配置，删除时间信息
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # 只保留消息内容
)
logger = logging.getLogger(__name__)

def test_model(language, batch_size, device):
    """
    测试模型
    """
    # 加载测试数据
    data_reader = DataReader()
    test_texts, test_labels = data_reader.load_test_data(language)
    
    # 加载模型和预处理器配置
    model_path = os.path.join('checkpoints', language, 'bert.pt')
    checkpoint = torch.load(model_path, map_location=device)
    preprocessor_config = checkpoint['preprocessor_config']
    
    # 初始化预处理器和模型
    preprocessor = BertPreprocessor(
        pretrained_model_name=preprocessor_config['pretrained_model_name'],
        max_length=preprocessor_config['max_length'],
        language=language
    )
    
    model = BertForSentiment(
        pretrained_model_name=preprocessor_config['pretrained_model_name']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 批量处理测试数据
    all_preds = []
    all_labels = test_labels
    total_loss = 0
    criterion = torch.nn.CrossEntropyLoss()
    
    # 创建进度条
    test_bar = tqdm(range(0, len(test_texts), batch_size), desc=f'Testing {language} model')
    
    with torch.no_grad():
        for i in test_bar:
            # 获取当前批次数据
            batch_texts = test_texts[i:i + batch_size]
            batch_labels = test_labels[i:i + batch_size]
            
            # 处理文本
            encoded_dict = preprocessor.process(batch_texts)
            
            # 将数据移动到设备
            input_ids = encoded_dict['input_ids'].to(device)
            attention_mask = encoded_dict['attention_mask'].to(device)
            token_type_ids = encoded_dict['token_type_ids'].to(device)
            batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # 计算损失
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            
            # 获取预测结果
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
    
    # 计算平均损失
    avg_loss = total_loss / len(test_bar)
    
    # 使用统一的输出格式
    print_test_results(
        language=language,
        loss=avg_loss,
        predictions=all_preds,
        labels=all_labels,
        model_name='bert'
    )

def main():
    parser = argparse.ArgumentParser(description='Test BERT model for sentiment classification')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 测试中英文模型
    for language in ['en', 'cn']:
        print_section_header(language)
        try:
            test_model(language, args.batch_size, device)
        except Exception as e:
            logger.error(f'Error testing {language} model: {str(e)}')
            continue

if __name__ == '__main__':
    main() 