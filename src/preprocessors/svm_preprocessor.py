from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import nltk
from typing import List, Dict, Union
import pickle
import os
import logging

class SVMPreprocessor:
    def __init__(self, max_features: int = 5000, language: str = 'cn'):
        """
        初始化SVM预处理器
        Args:
            max_features: 最大特征数量
            language: 语言类型 ('cn' 或 'en')
        """
        self.language = language
        self.max_features = max_features
        # 配置TF-IDF向量化器，使用unigram和bigram特征
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # 使用unigram和bigram特征
            token_pattern=r"(?u)\b\w+\b"  # 支持中文分词
        )
        
        # 初始化分词器
        if language == 'cn':
            # 设置jieba的日志级别为WARNING，避免重复输出初始化信息
            import jieba.analyse
            jieba.setLogLevel(logging.WARNING)
            jieba.initialize()
        else:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
    
    def preprocess_text(self, text: str) -> str:
        """
        对文本进行预处理和分词
        Args:
            text: 输入文本
        Returns:
            处理后的文本
        """
        if self.language == 'cn':
            tokens = jieba.cut(text)
        else:
            tokens = nltk.word_tokenize(text.lower())
        return ' '.join(tokens)
    
    def fit_transform(self, texts: List[str]) -> Union[List[List[float]], List[str]]:
        """
        拟合TF-IDF向量化器并转换文本数据
        Args:
            texts: 文本列表
        Returns:
            特征矩阵
        """
        processed_texts = [self.preprocess_text(text) for text in texts]
        return self.vectorizer.fit_transform(processed_texts)
    
    def transform(self, texts: List[str]) -> Union[List[List[float]], List[str]]:
        """
        使用已训练的向量化器转换文本数据
        Args:
            texts: 文本列表
        Returns:
            特征矩阵
        """
        processed_texts = [self.preprocess_text(text) for text in texts]
        return self.vectorizer.transform(processed_texts)
    
    def save(self, path: str):
        """
        保存预处理器的配置和向量化器
        Args:
            path: 保存路径
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'max_features': self.max_features,
                'language': self.language
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'SVMPreprocessor':
        """
        从文件加载预处理器
        Args:
            path: 加载路径
        Returns:
            预处理器实例
        """
        with open(path, 'rb') as f:
            config = pickle.load(f)
        
        preprocessor = cls(
            max_features=config['max_features'],
            language=config['language']
        )
        preprocessor.vectorizer = config['vectorizer']
        return preprocessor 