import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from .base_preprocessor import BasePreprocessor

class LSTMPreprocessor(BasePreprocessor):
    """
    LSTM模型的数据预处理器
    继承自BasePreprocessor，添加了序列填充和注意力掩码生成功能
    """
    def __init__(
        self,
        max_sequence_length: int = 200,
        vocab_size: int = 50000,
        language: str = 'cn',
        pad_token: str = '<PAD>',
        unk_token: str = '<UNK>'
    ):
        """
        初始化预处理器
        Args:
            max_sequence_length: 序列最大长度
            vocab_size: 词汇表大小
            language: 语言类型 ('cn' 或 'en')
            pad_token: 填充标记
            unk_token: 未知词标记
        """
        super().__init__(language=language)
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.pad_token = pad_token
        self.unk_token = unk_token
        
        # 词汇表相关
        self.word2idx: Dict[str, int] = {pad_token: 0, unk_token: 1}
        self.idx2word: Dict[int, str] = {0: pad_token, 1: unk_token}
        self.word_freq: Dict[str, int] = {}
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        构建词汇表，按词频选择前vocab_size个词
        Args:
            texts: 文本列表
        """
        for text in texts:
            for token in self.tokenize(text):
                self.word_freq[token] = self.word_freq.get(token, 0) + 1
        
        # 选择词频最高的词构建词汇表，预留PAD和UNK位置
        sorted_words = sorted(
            self.word_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.vocab_size - 2]  # 减去PAD和UNK
        
        for word, _ in sorted_words:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def text_to_sequence(self, text: str) -> List[int]:
        """
        将文本转换为定长序列，包含截断或填充操作
        Args:
            text: 输入文本
        Returns:
            sequence: 转换后的序列
        """
        tokens = self.tokenize(text)
        sequence = []
        
        # 转换为索引序列
        for token in tokens:
            if token in self.word2idx:
                sequence.append(self.word2idx[token])
            else:
                sequence.append(self.word2idx[self.unk_token])
        
        # 截断或填充
        if len(sequence) > self.max_sequence_length:
            sequence = sequence[:self.max_sequence_length]
        else:
            sequence.extend([self.word2idx[self.pad_token]] * (self.max_sequence_length - len(sequence)))
        
        return sequence
    
    def process(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理文本列表，生成序列和注意力掩码
        Args:
            texts: 文本列表
        Returns:
            sequences: 序列数组，shape: [batch_size, max_sequence_length]
            attention_mask: 注意力掩码数组，shape: [batch_size, max_sequence_length]
        """
        sequences = []
        attention_masks = []
        
        for text in texts:
            # 转换为序列
            sequence = self.text_to_sequence(text)
            sequences.append(sequence)
            
            # 生成注意力掩码，非填充位置为1，填充位置为0
            mask = [1 if idx != self.word2idx[self.pad_token] else 0 for idx in sequence]
            attention_masks.append(mask)
        
        return np.array(sequences), np.array(attention_masks)
    
    def decode_sequence(self, sequence: List[int]) -> str:
        """
        将序列解码为文本，忽略填充标记
        Args:
            sequence: 输入序列
        Returns:
            text: 解码后的文本
        """
        tokens = []
        for idx in sequence:
            if idx == self.word2idx[self.pad_token]:
                continue
            tokens.append(self.idx2word.get(idx, self.unk_token))
        
        if self.language == 'cn':
            return ''.join(tokens)
        else:
            return ' '.join(tokens)
    
    def get_vocab_size(self) -> int:
        """
        获取词汇表大小
        Returns:
            vocab_size: 词汇表大小
        """
        return len(self.word2idx)
    
    def get_pad_idx(self) -> int:
        """
        获取填充标记的索引
        Returns:
            pad_idx: 填充标记的索引
        """
        return self.word2idx[self.pad_token]
    
    def get_attention_mask(self, sequence: List[int]) -> List[int]:
        """
        生成注意力掩码，用于标识序列中的有效位置
        Args:
            sequence: 输入序列
        Returns:
            mask: 注意力掩码，1表示有效位置，0表示填充位置
        """
        return [1 if idx != self.word2idx[self.pad_token] else 0 for idx in sequence]
    
    def save_vocab(self, vocab_path: str) -> None:
        """
        保存词汇表相关信息到文件
        Args:
            vocab_path: 保存路径
        """
        vocab_dict = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'word_freq': self.word_freq
        }
        torch.save(vocab_dict, vocab_path)
    
    def load_vocab(self, vocab_path: str) -> None:
        """
        从文件加载词汇表信息
        Args:
            vocab_path: 词汇表路径
        """
        vocab_dict = torch.load(vocab_path)
        self.word2idx = vocab_dict['word2idx']
        self.idx2word = vocab_dict['idx2word']
        self.word_freq = vocab_dict['word_freq'] 