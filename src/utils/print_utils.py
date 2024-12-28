import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)

def print_test_results(language: str, loss: float, predictions: list, labels: list, model_name: str):
    """
    统一的测试结果输出格式
    Args:
        language: 语言类型 ('en' 或 'cn')
        loss: 测试损失值
        predictions: 预测结果列表
        labels: 真实标签列表
        model_name: 模型名称，用于保存混淆矩阵图像
    """
    # 计算评估指标
    report = classification_report(
        labels,
        predictions,
        target_names=['Negative', 'Positive']
    )
    cm = confusion_matrix(labels, predictions)
    
    logger.info(f'\n=== {language.upper()} Test Results ===')
    logger.info(f'Test Loss: {loss:.4f}')
    
    # 过滤掉分类报告中的accuracy行，只保留各类别的详细指标
    report_lines = report.split('\n')
    filtered_report = '\n'.join([line for line in report_lines if 'accuracy' not in line.lower()])
    logger.info('\nClassification Report:')
    logger.info(filtered_report)
    
    # 打印混淆矩阵
    logger.info('\nConfusion Matrix:')
    logger.info('                  Predicted Negative  Predicted Positive')
    logger.info(f'Actual Negative             {cm[0][0]:<4}               {cm[0][1]:<4}')
    logger.info(f'Actual Positive             {cm[1][0]:<4}               {cm[1][1]:<4}')
    
    # 保存混淆矩阵图像
    save_dir = os.path.join('results', model_name)
    os.makedirs(save_dir, exist_ok=True)
    plot_confusion_matrix(cm, language, save_dir)
    logger.info(f'\nConfusion matrix plot saved to {os.path.join(save_dir, f"confusion_matrix_{language}.png")}')

def plot_confusion_matrix(conf_matrix, language, save_dir):
    """
    绘制并保存混淆矩阵热力图
    Args:
        conf_matrix: 混淆矩阵
        language: 语言类型
        save_dir: 保存目录
    """
    plt.figure(figsize=(8, 6))
    # 使用seaborn的热力图可视化混淆矩阵，annot=True显示具体数值
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive']
    )
    plt.title(f'Confusion Matrix - {language.upper()} Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, f'confusion_matrix_{language}.png'))
    plt.close()

def print_section_header(language: str, section: str = 'Testing'):
    """
    打印统一的分节标题
    Args:
        language: 语言类型 ('en' 或 'cn')
        section: 节标题类型
    """
    title = f'{section} {language.upper()} model'
    decoration = '=' * (35 - len(title)//2)
    logger.info(f'\n{decoration}{title}{decoration}') 