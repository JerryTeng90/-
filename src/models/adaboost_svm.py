import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import List, Tuple, Optional
import pickle
import os
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class AdaBoostSVM(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 50,
        base_estimator_params: dict = None,
        learning_rate: float = 0.1,
        random_state: int = None
    ):
        """
        初始化 AdaBoost-SVM 模型
        Args:
            n_estimators: 基分类器数量
            base_estimator_params: 基分类器参数
            learning_rate: 学习率
            random_state: 随机种子
        """
        self.n_estimators = n_estimators
        self.base_estimator_params = base_estimator_params or {}
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # 使用不同核函数的SVM组合
        self.base_estimator_params_list = [
            {'kernel': 'rbf', 'C': 10.0},
            {'kernel': 'linear', 'C': 1.0}
        ]
        
        # ���始化模型列表和权重
        self.estimators_: List[SVC] = []
        self.estimator_weights_: Optional[np.ndarray] = None
        self.estimator_errors_: Optional[np.ndarray] = None
        
        # 样本权重
        self.sample_weights_: Optional[np.ndarray] = None

    def fit(self, X, y):
        """
        训练模型
        Args:
            X: 特征矩阵
            y: 标签向量
        Returns:
            self: 训练好的模型
        """
        # 初始化
        n_samples = X.shape[0]
        self.estimators_ = []
        self.estimator_weights_ = []
        
        # 初始化样本权重
        self.sample_weights_ = np.ones(n_samples) / n_samples
        
        # 使用tqdm显示训练进度
        pbar = tqdm(range(self.n_estimators), desc='Training AdaBoost-SVM')
        
        # 添加提前停止条件
        no_improvement_count = 0
        best_error = float('inf')
        
        # 训练每个基分类器
        for i in pbar:
            # 随机选择一个基分类器配置
            base_params = self.base_estimator_params_list[i % len(self.base_estimator_params_list)]
            base_params.update({'probability': True, 'random_state': self.random_state})
            
            # 训练基分类器
            estimator = SVC(**base_params)
            estimator.fit(X, y, sample_weight=self.sample_weights_)
            
            # 获取预测结果
            y_pred = estimator.predict(X)
            
            # 计算错误率
            incorrect = y_pred != y
            estimator_error = np.mean(incorrect * self.sample_weights_)
            
            # 更新进度条信息
            pbar.set_postfix({
                'error': f'{estimator_error:.4f}',
                'n_estimators': len(self.estimators_)
            })
            
            # 如果错误率太高，跳过这个分类器
            if estimator_error > 0.5:
                continue
                
            # 计算分类器权重
            estimator_weight = self.learning_rate * np.log((1 - estimator_error) / estimator_error)
            
            # 更新样本权重
            self.sample_weights_ *= np.exp(estimator_weight * incorrect)
            self.sample_weights_ /= np.sum(self.sample_weights_)
            
            # 保存基分类器和权重
            self.estimators_.append(estimator)
            self.estimator_weights_.append(estimator_weight)
            
            # 检查是否有改善
            if estimator_error < best_error:
                best_error = estimator_error
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # 如果连续5个分类器没有改善，提前停止
            if no_improvement_count >= 5:
                logger.info(f'Early stopping at iteration {i} due to no improvement')
                break
        
        return self

    def predict_proba(self, X):
        """
        预测概率
        Args:
            X: 特征矩阵
        Returns:
            预测概率矩阵
        """
        # 检查是否已训练
        if not self.estimators_:
            raise ValueError("请先训练模型")
        
        # 获取每个基分类器的预测概率
        proba = np.zeros((X.shape[0], 2))
        
        # 将列表转换为numpy数组
        estimator_weights = np.array(self.estimator_weights_)
        
        # 获取每个基分类器的预测概率并加权
        for estimator, weight in zip(self.estimators_, estimator_weights):
            proba += weight * estimator.predict_proba(X)
        
        # 归一化
        proba /= np.sum(estimator_weights)
        return proba

    def predict(self, X):
        """
        预测类别
        Args:
            X: 特征矩阵
        Returns:
            预测类别
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def save(self, path: str):
        """
        保存模型
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'estimators': self.estimators_,
                'estimator_weights': self.estimator_weights_,
                'sample_weights': self.sample_weights_,
                'params': {
                    'n_estimators': self.n_estimators,
                    'base_estimator_params': self.base_estimator_params,
                    'learning_rate': self.learning_rate,
                    'random_state': self.random_state
                }
            }, f)

    @classmethod
    def load(cls, path: str) -> 'AdaBoostSVM':
        """
        加载模型
        Args:
            path: 加载路径
        Returns:
            加载的模型
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        # 创建模型实例
        model = cls(
            n_estimators=data['params']['n_estimators'],
            base_estimator_params=data['params']['base_estimator_params'],
            learning_rate=data['params']['learning_rate'],
            random_state=data['params']['random_state']
        )
        
        # 加载模型参数
        model.estimators_ = data['estimators']
        model.estimator_weights_ = data['estimator_weights']
        model.sample_weights_ = data['sample_weights']
        
        return model