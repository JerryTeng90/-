import os
import re
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

class DataReader:
    def __init__(self, base_path: str = "data", val_ratio: float = 0.2, random_seed: int = 42):
        """
        初始化DataReader
        Args:
            base_path: 数据集的根目录
            val_ratio: 验证集比例
            random_seed: 随机种子
        """
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.base_path = os.path.join(project_root, base_path)
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        self.train_path = os.path.join(self.base_path, "evaltask2_sample_data")
        self.test_labeled_path = os.path.join(self.base_path, "Sentiment Classification with Deep Learning")
        self.test_unlabeled_path = os.path.join(self.base_path, "Test Data RELEASE 140523V1")

    def read_xml_file(self, file_path: str) -> List[Tuple[str, Optional[int]]]:
        """
        读取XML格式的文件
        Args:
            file_path: XML文件路径
        Returns:
            包含(文本, 标签)元组的列表，如果没有标签则标签为None
        """
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return []
            
        reviews = []
        
        try:
            # 直接以二进制模式读取文件
            with open(file_path, 'rb') as f:
                raw_content = f.read()
            
            # 尝试检测编码
            import chardet
            result = chardet.detect(raw_content)
            encoding = result['encoding']
            confidence = result['confidence']
            
            print(f"Detected encoding: {encoding} with confidence: {confidence}")
            
            # 如果检测到的编码可信度较低，尝试常用编码
            if confidence < 0.8:
                encodings = ['utf-8', 'utf-8-sig', 'utf-16', 'utf-16le', 'utf-16be', 'ascii', 'iso-8859-1']
                for enc in encodings:
                    try:
                        content = raw_content.decode(enc)
                        if '<review' in content:
                            encoding = enc
                            break
                    except:
                        continue
            else:
                content = raw_content.decode(encoding)
            
            if not content or '<review' not in content:
                print(f"Error: Could not decode file content properly")
                return []

            print(f"Successfully read {file_path} with {encoding} encoding")
            
            content = self._clean_xml_content(content)
            
            # 匹配带标签的评论，使用非贪婪模式匹配，并处理多行内容
            pattern = r'<review[^>]*?\s+label="(\d+)"[^>]*?>\s*([\s\S]*?)\s*</review>'
            matches = list(re.finditer(pattern, content))
            
            if matches:
                for match in matches:
                    try:
                        label = int(match.group(1))
                        text = match.group(2).strip()
                        text = re.sub(r'\n\s*\n', '\n', text)
                        text = text.strip()
                        if text:
                            reviews.append((text, label))
                    except Exception as e:
                        print(f"Error parsing review: {e}")
                        continue
            else:
                # 如果没有找到带标签的评论，尝试匹配无标签的评论
                pattern = r'<review[^>]*?>\s*([\s\S]*?)\s*</review>'
                matches = list(re.finditer(pattern, content))
                for match in matches:
                    try:
                        text = match.group(1).strip()
                        # 移除多余的空行
                        text = re.sub(r'\n\s*\n', '\n', text)
                        text = text.strip()
                        if text:
                            reviews.append((text, None))
                    except Exception as e:
                        print(f"Error parsing review: {e}")
                        continue

            print(f"Found {len(reviews)} reviews in {file_path}")
            if len(reviews) == 0:
                print("Warning: No reviews found. Content sample:")
                print(content[:500])  # 打印前500个字符用于调试
            
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            return []
            
        return reviews

    def _clean_xml_content(self, content: str) -> str:
        """
        清理XML内容
        Args:
            content: 原始XML内容
        Returns:
            清理后的XML内容
        """
        # 移除XML声明
        content = re.sub(r'<\?xml[^>]*\?>', '', content)
        # 移除DOCTYPE
        content = re.sub(r'<!DOCTYPE[^>]*>', '', content)
        # 移除注释
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        # 处理特殊字符
        content = content.replace('&', '&amp;')
        # 确保review标签格式正确
        content = re.sub(r'<review\s*>', '<review>', content)
        return content.strip()

    def load_raw_training_data(self, language: str = 'cn') -> Tuple[List[str], List[int]]:
        """
        加载原始训练数据
        Args:
            language: 语言类型 ('cn' 或 'en')
        Returns:
            (texts, labels)元组
        """
        data_dir = os.path.join(self.train_path, f'{language}_sample_data')
        print(f"\nLoading training data from directory: {data_dir}")
        
        # 读取正面评论
        pos_file = os.path.join(data_dir, 'sample.positive.txt')
        print(f"Loading positive samples from: {pos_file}")
        print(f"File exists: {os.path.exists(pos_file)}")
        pos_texts, pos_labels = self._read_file(pos_file)
        print(f"Loaded {len(pos_texts)} positive samples")
        
        # 读取负面评论
        neg_file = os.path.join(data_dir, 'sample.negative.txt')
        print(f"Loading negative samples from: {neg_file}")
        print(f"File exists: {os.path.exists(neg_file)}")
        neg_texts, _ = self._read_file(neg_file)
        neg_labels = [0] * len(neg_texts)
        print(f"Loaded {len(neg_texts)} negative samples")
        
        texts = pos_texts + neg_texts
        labels = pos_labels + neg_labels
        
        if not texts:
            print(f"Warning: No data found in {data_dir}")
            print(f"Current working directory: {os.getcwd()}")
            raise ValueError(f'No data found in {data_dir}')
        
        print(f"Total samples loaded: {len(texts)}")
        return texts, labels

    def load_train_val_data(self, language: str = 'cn') -> Tuple[List[str], List[int], List[str], List[int]]:
        """
        加载并划分训练集和验证集
        Args:
            language: 'cn' 或 'en'
        Returns:
            (train_texts, train_labels, val_texts, val_labels) 元组
        """
        texts, labels = self.load_raw_training_data(language)
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels,
            test_size=self.val_ratio,
            random_state=self.random_seed,
            stratify=labels  # 确保标签分布一致
        )
        
        return train_texts, train_labels, val_texts, val_labels

    def load_test_data(self, language: str = 'cn') -> Tuple[List[str], Optional[List[int]]]:
        """
        加载测试数据
        Args:
            language: 'cn' 或 'en'
        Returns:
            (texts, labels) 元组，如果是无标签数据，labels为None
        """
        if language not in ['cn', 'en']:
            raise ValueError(f"Language must be 'cn' or 'en', got {language}")
            
        file_path = os.path.join(self.test_labeled_path, f"test.label.{language}.txt")
        logger.info(f"Loading test data from: {file_path}")
        
        reviews = self.read_xml_file(file_path)
        if not reviews:
            raise ValueError(f"No reviews found in {file_path}")
        
        texts = [text for text, _ in reviews]
        labels = [label for _, label in reviews]
        
        # 移除重复的输出，只保留一条
        logger.info(f"Loaded {len(texts)} test samples")
        
        return texts, labels

    def get_dataset_stats(self, language: str = 'cn') -> Dict[str, Dict[str, int]]:
        """
        获取数据集统计信息
        Args:
            language: 'cn' 或 'en'
        Returns:
            包含各个数据集统计信息的字典
        """
        stats = {}
        
        # 训练集和验证集统计
        train_texts, train_labels, val_texts, val_labels = self.load_train_val_data(language)
        stats['train'] = {
            'total': len(train_texts),
            'positive': sum(1 for label in train_labels if label == 1),
            'negative': sum(1 for label in train_labels if label == 0)
        }
        stats['val'] = {
            'total': len(val_texts),
            'positive': sum(1 for label in val_labels if label == 1),
            'negative': sum(1 for label in val_labels if label == 0)
        }
        
        # 带标签测试集统计
        test_texts, test_labels = self.load_test_data(labeled=True, language=language)
        if test_labels:
            stats['test'] = {
                'total': len(test_texts),
                'positive': sum(1 for label in test_labels if label == 1),
                'negative': sum(1 for label in test_labels if label == 0)
            }
        
        # 无标签测试集统计
        unlabeled_texts, _ = self.load_test_data(labeled=False, language=language)
        stats['unlabeled'] = {
            'total': len(unlabeled_texts)
        }
        
        return stats

    def _read_file(self, file_path: str) -> Tuple[List[str], List[int]]:
        """
        读取文件并提取评论内容
        Args:
            file_path: 文件路径
        Returns:
            (texts, labels)元组
        """
        texts = []
        labels = []
        
        # 尝试不同的编码方式
        encodings = ['utf-8', 'utf-16']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                    if '<review' in content:
                        print(f'Successfully read {file_path} with {encoding} encoding')
                        # 使用正则表达式提取评论内容
                        reviews = re.findall(r'<review.*?>(.*?)</review>', content, re.DOTALL)
                        texts.extend([review.strip() for review in reviews if review.strip()])
                        print(f'Found {len(reviews)} reviews in {file_path}')
                        return texts, [1] * len(texts)
            except UnicodeError:
                continue
            except Exception as e:
                print(f'Error reading {file_path} with {encoding} encoding: {str(e)}')
                continue
        
        print(f'Failed to read {file_path} with any encoding')
        return [], []

    def load_train_data(self, language: str = 'cn') -> Tuple[List[str], List[int]]:
        """
        加载训练数据
        Args:
            language: 'cn' 或 'en'
        Returns:
            (texts, labels) 元组
        """
        train_texts, train_labels, _, _ = self.load_train_val_data(language)
        return train_texts, train_labels

    def load_validation_data(self, language: str = 'cn') -> Tuple[List[str], List[int]]:
        """
        加载验证数据
        Args:
            language: 'cn' 或 'en'
        Returns:
            (texts, labels) 元组
        """
        _, _, val_texts, val_labels = self.load_train_val_data(language)
        return val_texts, val_labels

