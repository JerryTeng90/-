import re
import jieba
import numpy as np
from typing import List, Set, Dict
from collections import Counter
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging

# 设置jieba的日志级别为WARNING
jieba.setLogLevel(logging.WARNING)

# 设置日志配置
logger = logging.getLogger(__name__)

class CNNPreprocessor:
    def __init__(self, max_sequence_length: int = 128, min_freq: int = 5,
                 vocab_size: int = None, language: str = 'cn'):
        """
        初始化预处理器
        Args:
            max_sequence_length: 最大序列长度
            min_freq: 最小词频
            vocab_size: 词汇表大小限制
            language: 语言类型 ('cn' 或 'en')
        """
        self.max_sequence_length = max_sequence_length
        self.min_freq = min_freq
        self.vocab_size = vocab_size
        self.language = language
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = Counter()
        
        # 下载必要的NLTK数据（静默模式）
        if self.language == 'en':
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
        
        # 加载停用词
        self.stop_words = self._load_stop_words()
    
    def _load_stop_words(self) -> Set[str]:
        """
        加载对应语言的停用词表
        Returns:
            停用词集合
        """
        if self.language == 'cn':
            return set()  # 中文暂时不使用停用词
        else:
            try:
                return set(stopwords.words('english'))
            except LookupError:
                nltk.download('stopwords', quiet=True)
                return set(stopwords.words('english'))

    def _clean_text(self, text: str) -> str:
        """
        清理文本，包括移除HTML标签、特殊字符和数字，统一大小写
        Args:
            text: 原始文本
        Returns:
            清理后的文本
        """
        text = re.sub(r'<[^>]+>', '', text)
        
        if self.language == 'cn':
            text = re.sub(r'[^\u4e00-\u9fa5]', ' ', text)
        else:
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            text = text.lower()
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _tokenize(self, text: str) -> List[str]:
        """
        对文本进行分词，包括文本清理、分词和停用词过滤
        Args:
            text: 文本
        Returns:
            词列表
        """
        text = self._clean_text(text)
        
        if self.language == 'cn':
            tokens = list(jieba.cut(text))
        else:
            try:
                tokens = text.split()  # 使用简单的空格分词
                if not tokens:  # 如果分词结果为空，尝试使用NLTK的word_tokenize
                    tokens = word_tokenize(text)
            except Exception as e:
                logger.warning(f'Tokenization error: {e}')
                tokens = text.split()  # 如果出错，回退到简单的空格分词
        
        # 过滤停用词
        tokens = [token for token in tokens if token not in self.stop_words]
        
        return tokens

    def build_vocab(self, texts: List[str]) -> None:
        """
        构建词汇表，包括词频统计、低频词过滤和词汇表大小限制
        Args:
            texts: 文本列表
        """
        # 统计词频
        for text in texts:
            tokens = self._tokenize(text)
            self.word_freq.update(tokens)
        
        # 过滤低频词
        words = [word for word, freq in self.word_freq.items() 
                if freq >= self.min_freq]
        
        # 限制词汇表大小
        if self.vocab_size is not None:
            words = sorted(words, key=lambda x: self.word_freq[x], reverse=True)
            words = words[:self.vocab_size-2]  # 预留PAD和UNK标记位置
        
        # 构建映射
        for word in words:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def _convert_to_indices(self, tokens: List[str]) -> List[int]:
        """
        将词列表转换为索引列表，未知词转换为UNK标记
        Args:
            tokens: 词列表
        Returns:
            索引列表
        """
        return [self.word2idx.get(token, self.word2idx['<UNK>']) 
                for token in tokens]
    
    def _pad_sequence(self, indices: List[int]) -> List[int]:
        """
        对序列进行截断或填充到指定长度
        Args:
            indices: 索引列表
        Returns:
            处理后的定长索引列表
        """
        if len(indices) > self.max_sequence_length:
            return indices[:self.max_sequence_length]
        else:
            return indices + [self.word2idx['<PAD>']] * (self.max_sequence_length - len(indices))

    def process(self, texts: List[str]) -> List[List[int]]:
        """
        处理文本列表，包括分词、转换为索引和序列填充
        Args:
            texts: 文本列表
        Returns:
            处理后的数值序列
        """
        processed = []
        for text in texts:
            tokens = self._tokenize(text)
            indices = self._convert_to_indices(tokens)
            padded = self._pad_sequence(indices)
            processed.append(padded)
        return processed
    
    def get_vocab_size(self) -> int:
        """
        获取词汇表大小
        Returns:
            词汇表大小
        """
        return len(self.word2idx)

