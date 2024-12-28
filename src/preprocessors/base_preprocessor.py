import re
import jieba
import nltk
from typing import List, Union
from abc import ABC, abstractmethod

# 全局初始化
jieba.initialize()

class BasePreprocessor(ABC):
    def __init__(self, language: str = 'cn'):
        """
        初始化基础预处理器
        Args:
            language: 语言类型 ('cn' 或 'en')
        """
        self.language = language

    def clean_text(self, text: str) -> str:
        """
        基础文本清理，包括XML标签移除、空白字符统一和特殊字符处理
        Args:
            text: 输入文本
        Returns:
            清理后的文本
        """
        if not text:
            return ""
            
        # 移除XML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 统一空白字符
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符，但保留中文标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？、；：""''（）【】《》]', '', text)
        
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        """
        根据语言类型对文本进行分词
        Args:
            text: 输入文本
        Returns:
            tokens: 分词结果
        """
        if self.language == 'cn':
            return list(jieba.cut(text))
        else:
            # 使用更简单的分词方式，避免依赖NLTK的复杂功能
            text = text.lower()
            # 基本的标点符号处理
            for punct in ',.!?;:':
                text = text.replace(punct, f' {punct} ')
            return text.split()

    @abstractmethod
    def process(self, texts: Union[str, List[str]], language: str = 'cn') -> Union[List[str], List[List[str]]]:
        """
        处理文本的抽象方法，需要被子类实现
        Args:
            texts: 输入文本或文本列表
            language: 语言类型
        Returns:
            处理后的文本或文本列表
        """
        pass

