import torch
import torch.nn as nn
from transformers import BertModel
from typing import Optional, Dict

class BertForSentiment(nn.Module):
    def __init__(self, pretrained_model_name: str, dropout_rate: float = 0.1):
        """
        初始化BERT情感分类模型
        Args:
            pretrained_model_name: 预训练模型名称
            dropout_rate: Dropout率
        """
        super().__init__()
        
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        
        # 获取BERT隐藏层大小
        self.hidden_size = self.bert.config.hidden_size
        
        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 分类器
        self.classifier = nn.Linear(self.hidden_size, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        前向传播
        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            token_type_ids: 标记类型ID
        Returns:
            logits: 分类logits
        """
        # 获取BERT输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 使用[CLS]标记的输出进行分类
        pooled_output = outputs.pooler_output
        
        # Dropout
        pooled_output = self.dropout(pooled_output)
        
        # 分类
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_bert_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取BERT特征（用于特征分析）
        Args:
            input_ids: 输入序列的token ids
            attention_mask: 注意力掩码
            token_type_ids: token类型ids
        Returns:
            pooled_output: BERT的池化输出
        """
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True
            )
            return outputs.pooler_output
    
    def freeze_bert_layers(self, num_layers: int = 0) -> None:
        """
        冻结BERT的底层
        Args:
            num_layers: 要冻结的层数，0表示不冻结，-1表示冻结所有层
        """
        if num_layers == 0:
            return
        
        # 冻结嵌入层
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        
        # 冻结编码层
        if num_layers == -1:
            num_layers = len(self.bert.encoder.layer)
        
        for i in range(num_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
    
    def unfreeze_bert_layers(self) -> None:
        """
        解冻所有BERT层
        """
        for param in self.bert.parameters():
            param.requires_grad = True
    
    @property
    def device(self) -> torch.device:
        """
        获取模型所在设备
        Returns:
            device: 模型所在设备
        """
        return next(self.parameters()).device