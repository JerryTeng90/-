import os
import argparse
import logging
import numpy as np
from sklearn.metrics import classification_report
import pickle
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
import time

from data.data_reader import DataReader
from preprocessors.svm_preprocessor import SVMPreprocessor
from models.adaboost_svm import AdaBoostSVM

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def train(args):
    """
    训练 AdaBoost-SVM 模型
    """
    start_time = time.time()
    
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
    
    # 定义参数组合
    param_combinations = [
        {
            'base_estimator_params': {'kernel': 'rbf', 'C': 1.0, 'probability': True},
            'n_estimators': 30,
            'learning_rate': 0.1
        },
        {
            'base_estimator_params': {'kernel': 'rbf', 'C': 10.0, 'probability': True},
            'n_estimators': 30,
            'learning_rate': 1.0
        }
    ]
    
    # 手动进行参数搜索
    logger.info('Performing parameter search...')
    search_start = time.time()
    
    best_score = -float('inf')
    best_model = None
    best_params = None
    
    for params in param_combinations:
        logger.info(f'\nTrying parameters: {params}')
        
        # 初始化模型
        model = AdaBoostSVM(
            base_estimator_params=params['base_estimator_params'],
            n_estimators=params['n_estimators'],
            learning_rate=params['learning_rate'],
            random_state=42
        )
        
        # 训练模型
        model.fit(X_train, train_labels)
        
        # 在验证集上评估
        val_pred = model.predict(X_val)
        score = classification_report(
            val_labels,
            val_pred,
            target_names=['Negative', 'Positive'],
            output_dict=True
        )['macro avg']['f1-score']
        
        logger.info(f'Validation F1 score: {score:.4f}')
        
        # 更新最佳模型
        if score > best_score:
            best_score = score
            best_model = model
            best_params = params
    
    search_time = time.time() - search_start
    logger.info(f'\nParameter search completed in {search_time:.2f} seconds')
    logger.info(f'Best parameters: {best_params}')
    logger.info(f'Best validation score: {best_score:.4f}')
    
    # 使用最佳模型
    model = best_model
    
    # 验证
    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)
    
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
    model_path = os.path.join('checkpoints', args.language, 'adaboost_svm.pkl')
    preprocessor_path = os.path.join('checkpoints', args.language, 'adaboost_svm_preprocessor.pkl')
    
    # 保存模型
    model.save(model_path)
    # 保存预处理器
    preprocessor.save(preprocessor_path)
    
    logger.info(f'Model saved to {model_path}')
    logger.info(f'Preprocessor saved to {preprocessor_path}')
    
    # 打印每个基分类器的信息
    logger.info('\nBase Classifier Information:')
    for i, (estimator, weight) in enumerate(zip(model.estimators_, model.estimator_weights_)):
        logger.info(f'Classifier {i+1}: weight = {weight:.4f}')
    
    total_time = time.time() - start_time
    logger.info(f'\nTotal training time: {total_time:.2f} seconds')

def main():
    parser = argparse.ArgumentParser(description='Train AdaBoost-SVM model for sentiment classification')
    
    # 数据参数
    parser.add_argument('--max_features', type=int, default=10000,
                      help='Maximum number of features')
    
    # 基分类器参数
    parser.add_argument('--kernel', type=str, default='rbf',
                      choices=['linear', 'rbf', 'poly'],
                      help='SVM kernel function')
    parser.add_argument('--C', type=float, default=10.0,
                      help='SVM regularization parameter')
    
    # AdaBoost参数
    parser.add_argument('--n_estimators', type=int, default=50,
                      help='Number of base classifiers')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                      help='AdaBoost learning rate')
    
    args = parser.parse_args()
    
    # 训练中英文模型
    for language in ['en', 'cn']:
        logger.info(f'\n{"="*50}')
        logger.info(f'Training {language.upper()} model')
        logger.info(f'{"="*50}')
        
        args.language = language
        try:
            train(args)
        except Exception as e:
            logger.error(f'Error training {language} model: {str(e)}')
            continue

if __name__ == '__main__':
    main() 