import matplotlib.pyplot as plt
import numpy as np
def plot_history(history):
    # 绘制并保存训练损失图表
    plt.figure(figsize=(6, 9))
    plt.plot(history['epoch'], history['loss'], label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.xticks(np.arange(min(history['epoch']), max(history['epoch']) + 1, 1.0))  # 设置横轴的刻度为整数
    plt.savefig('training_loss.png')  # 保存图表
    plt.show()

    # 绘制并保存训练和验证准确率图表
    plt.figure(figsize=(6, 9))
    plt.plot(history['epoch'], history['accuracy'], label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.xticks(np.arange(min(history['epoch']), max(history['epoch']) + 1, 1.0))  # 设置横轴的刻度为整数
    plt.savefig('training_validation_accuracy.png')  # 保存图表
    plt.show()

    # 绘制并保存验证准确率图表
    plt.figure(figsize=(6, 9))
    plt.plot(history['test_epoch'], history['valid_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.xticks(np.arange(min(history['test_epoch']), max(history['test_epoch']) + 1, 1.0))  # 设置横轴的刻度为整数
    plt.savefig('validation_accuracy.png')  # 保存图表
    plt.show()