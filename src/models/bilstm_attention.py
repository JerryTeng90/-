import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple

class AttentionLayer(nn.Module):
    """
    注意力层
    """
    def __init__(self, hidden_size: int):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, lstm_output: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        计算注意力权重并返回加权后的特征向量
        Args:
            lstm_output: BiLSTM的输出，shape: [batch_size, seq_len, hidden_size * 2]
            mask: 序列的mask，shape: [batch_size, seq_len]
        Returns:
            attention_weights: 注意力权重，shape: [batch_size, seq_len]
        """
        # 计算注意力分数
        attention_weights = self.attention(lstm_output)  # [batch_size, seq_len, 1]
        attention_weights = attention_weights.squeeze(-1)  # [batch_size, seq_len]
        
        # 如果提供了mask，将padding部分的注意力分数设为一个很小的负数
        if mask is not None:
            attention_weights = attention_weights.masked_fill(~mask, float('-inf'))
        
        # 使用softmax获取注意力权重
        attention_weights = F.softmax(attention_weights, dim=1)  # [batch_size, seq_len]
        
        return attention_weights

class BiLSTMAttention(nn.Module):
    """
    BiLSTM + Attention 模型
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.5,
        padding_idx: int = 0,
        num_classes: int = 2
    ):
        """
        初始化模型
        Args:
            vocab_size: 词汇表大小
            embedding_dim: 词嵌入维度
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            dropout: Dropout比率
            padding_idx: 填充token的索引
            num_classes: 分类类别数
        """
        super(BiLSTMAttention, self).__init__()
        
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = AttentionLayer(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化模型参数
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    # LSTM权重使用正交初始化
                    nn.init.orthogonal_(param)
                else:
                    # 其他权重使用xavier初始化
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor, return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        Args:
            x: 输入序列 [batch_size, seq_len]
            mask: 注意力掩码 [batch_size, seq_len]
            return_attention: 是否返回注意力权重
        Returns:
            output: 分类输出 [batch_size, num_classes]
            attention_weights: 注意力权重 [batch_size, seq_len] (如果return_attention=True)
        """
        # 词嵌入
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # BiLSTM
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_size * 2]
        
        # 注意力机制
        attention_weights = self.attention(lstm_out, mask)  # [batch_size, seq_len]
        
        # 加权求和
        weighted_sum = torch.bmm(attention_weights.unsqueeze(1), lstm_out)  # [batch_size, 1, hidden_size * 2]
        context = weighted_sum.squeeze(1)  # [batch_size, hidden_size * 2]
        
        # Dropout
        context = self.dropout(context)
        
        # 分类
        output = self.fc(context)  # [batch_size, num_classes]
        
        if return_attention:
            return output, attention_weights
        return output

    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """
        预测函数
        Args:
            input_ids: 输入序列，shape: [batch_size, seq_len]
            attention_mask: 注意力mask，shape: [batch_size, seq_len]
        Returns:
            predictions: 预测类别，shape: [batch_size]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        获取注意力权重（用于可视化）
        Args:
            input_ids: 输入序列，shape: [batch_size, seq_len]
            attention_mask: 注意力mask，shape: [batch_size, seq_len]
        Returns:
            attention_weights: 注意力权重，shape: [batch_size, seq_len]
        """
        self.eval()
        with torch.no_grad():
            # 词嵌入
            embedded = self.embedding(input_ids)
            
            # BiLSTM编码
            lstm_output, _ = self.lstm(embedded)
            
            # 计算注意力分数
            attention_weights = self.attention.attention(lstm_output).squeeze(2)
            
            # 应用mask
            if attention_mask is not None:
                attention_weights = attention_weights.masked_fill(~attention_mask, float('-inf'))
            
            # 使用softmax获取注意力权重
            attention_weights = F.softmax(attention_weights, dim=1)
            
        return attention_weights 