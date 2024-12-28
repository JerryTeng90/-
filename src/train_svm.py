import os
import argparse
import logging
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import pickle
from tqdm import tqdm

from data.data_reader import DataReader
from preprocessors.svm_preprocessor import SVMPreprocessor

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def train(args):
    """
    训练SVM模型
    """
    # 加载数据
    data_reader = DataReader()
    train_texts, train_labels = data_reader.load_train_data(args.language)
    val_texts, val_labels = data_reader.load_validation_data(args.language)
    
    logger.info(f'Loaded {len(train_texts)} training samples')
    logger.info(f'Loaded {len(val_texts)} validation samples')
    
    # 初始化预处理器
    preprocessor = SVMPreprocessor(
        max_features=args.max_features,
        language=args.language
    )
    
    # 特征提取
    logger.info('Extracting features...')
    X_train = preprocessor.fit_transform(train_texts)
    X_val = preprocessor.transform(val_texts)
    
    # 初始化并训练模型
    logger.info('Training SVM model...')
    model = SVC(
        kernel=args.kernel,
        C=args.C,
        probability=True,
        random_state=42
    )
    
    model.fit(X_train, train_labels)
    
    # 验证
    val_pred = model.predict(X_val)
    
    # 打印评估结果
    logger.info(f'\n=== {args.language.upper()} Validation Results ===')
    report = classification_report(
        val_labels,
        val_pred,
        target_names=['Negative', 'Positive']
    )
    logger.info('\nClassification Report:')
    logger.info(report)
    
    # 保存模型和预处理器
    os.makedirs(os.path.join('checkpoints', args.language), exist_ok=True)
    model_path = os.path.join('checkpoints', args.language, 'svm.pkl')
    preprocessor_path = os.path.join('checkpoints', args.language, 'svm_preprocessor.pkl')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    preprocessor.save(preprocessor_path)
    
    logger.info(f'Model saved to {model_path}')
    logger.info(f'Preprocessor saved to {preprocessor_path}')

def main():
    parser = argparse.ArgumentParser(description='Train SVM model for sentiment classification')
    
    # 数据参数
    parser.add_argument('--max_features', type=int, default=5000,
                      help='Maximum number of features')
    
    # 模型参数
    parser.add_argument('--kernel', type=str, default='rbf',
                      choices=['linear', 'rbf', 'poly'],
                      help='SVM kernel function')
    parser.add_argument('--C', type=float, default=1.0,
                      help='SVM regularization parameter')
    
    args = parser.parse_args()
    
    # 训练中英文模型
    for language in ['en', 'cn']:
        logger.info(f'\n{"="*50}')
        logger.info(f'Training {language.upper()} model')
        logger.info(f'{"="*50}')
        
        args.language = language
        train(args)

if __name__ == '__main__':
    main() 