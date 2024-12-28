import torch
from transformers import BertTokenizer
import jieba
import nltk
from typing import List, Dict, Union, Optional
import logging

logger = logging.getLogger(__name__)

class BertPreprocessor:
    def __init__(self, pretrained_model_name: str, max_length: int = 128, language: str = 'cn'):
        """
        初始化BERT预处理器
        Args:
            pretrained_model_name: 预训练模型名称
            max_length: 最大序列长度
            language: 语言类型 ('cn' 或 'en')
        """
        try:
            # 优先使用本地缓存加载tokenizer以提高性能
            self.tokenizer = BertTokenizer.from_pretrained(
                pretrained_model_name,
                local_files_only=True,  # 只使用本地缓存
                do_lower_case=True,     # 转小写以节省内存
                model_max_length=max_length  # 限制最大长度
            )
        except Exception as e:
            logger.error(f"Error loading tokenizer: {str(e)}")
            # 本地加载失败时从在线下载，并使用内存优化选项
            self.tokenizer = BertTokenizer.from_pretrained(
                pretrained_model_name,
                do_lower_case=True,
                model_max_length=max_length,
                use_fast=True,  # 使用快速tokenizer
                cache_dir='./cache'  # 指定缓存目录
            )
        
        self.max_length = max_length
        self.language = language
        
        # 初始化分词器
        if language == 'cn':
            # 设置jieba的日志级别为WARNING，避免重复输出初始化信息
            jieba.setLogLevel(logging.WARNING)
            jieba.initialize()
        else:
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')

    def process(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        处理文本列表，返回BERT模型所需的输入格式
        Args:
            texts: 文本列表
        Returns:
            包含input_ids、attention_mask和token_type_ids的字典
        """
        # 对文本进行分词
        if self.language == 'cn':
            tokenized_texts = [' '.join(jieba.cut(text)) for text in texts]
        else:
            # 英文文本预处理：转小写并分词
            tokenized_texts = []
            for text in texts:
                if not isinstance(text, str):
                    text = str(text)
                # 转小写并分词
                tokens = nltk.word_tokenize(text.lower())
                tokenized_texts.append(' '.join(tokens))
        
        # 使用BERT tokenizer进行编码
        encoded = self.tokenizer(
            tokenized_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'token_type_ids': encoded['token_type_ids']
        }

    def decode(self, token_ids: torch.Tensor) -> List[str]:
        """
        将token ID转换回文本
        Args:
            token_ids: token ID张量
        Returns:
            解码后的文本列表
        """
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

    def get_vocab_size(self) -> int:
        """
        获取词汇表大小
        Returns:
            词汇表大小
        """
        return len(self.tokenizer)
    
    def get_special_tokens(self) -> Dict[str, str]:
        """
        获取BERT的特殊标记
        Returns:
            special_tokens: 特殊标记字典
        """
        return {
            'pad_token': self.tokenizer.pad_token,
            'cls_token': self.tokenizer.cls_token,
            'sep_token': self.tokenizer.sep_token,
            'unk_token': self.tokenizer.unk_token,
            'mask_token': self.tokenizer.mask_token
        }
    
    def get_special_token_ids(self) -> Dict[str, int]:
        """
        获取BERT特殊标记的token ids
        Returns:
            special_token_ids: 特殊标记的token ids字典
        """
        return {
            'pad_token_id': self.tokenizer.pad_token_id,
            'cls_token_id': self.tokenizer.cls_token_id,
            'sep_token_id': self.tokenizer.sep_token_id,
            'unk_token_id': self.tokenizer.unk_token_id,
            'mask_token_id': self.tokenizer.mask_token_id
        }
    
    def save_tokenizer(self, path: str) -> None:
        """
        保存tokenizer到指定路径
        Args:
            path: 保存路径
        """
        self.tokenizer.save_pretrained(path)

    def load_tokenizer(self, path: str) -> None:
        """
        从指定路径加载tokenizer
        Args:
            path: 加载路径
        """
        self.tokenizer = BertTokenizer.from_pretrained(path)
    
    @property
    def pad_token_id(self) -> int:
        """
        获取填充标记的token id
        Returns:
            pad_token_id: 填充标记的token id
        """
        return self.tokenizer.pad_token_id 