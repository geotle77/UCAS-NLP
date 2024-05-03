import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

__path__ = "./homework2/train_info"

# load loss info
with open(__path__+'/loss_info_FNN', 'rb') as f:
    fnn_data = pickle.load(f)
with open(__path__+'/loss_info_RNN', 'rb') as f:
    rnn_data = pickle.load(f)
with open(__path__+'/loss_info_LSTM', 'rb') as f:
    lstm_data = pickle.load(f)


fnn_x_axi = lstm_data["x_record"]
fnn_y_record = lstm_data["y_record"]
fnn_y_axi_correct = [y[0] for y in fnn_y_record]
fnn_y_axi_loss = [y[1] for y in fnn_y_record]
fnn_y_axi_lr = [y[2] for y in fnn_y_record]

fig,axs = plt.subplots(3)
# 在第一个子图中绘制正确率
axs[0].plot(fnn_x_axi, fnn_y_axi_correct, label='FNN')
axs[0].set_title('Correct Rate')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Correct Rate')

# 在第二个子图中绘制损失
axs[1].plot(fnn_x_axi, fnn_y_axi_loss, label='FNN')
axs[1].set_title('Loss')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')

# 在第三个子图中绘制学习率
axs[2].plot(fnn_x_axi, fnn_y_axi_lr, label='FNN')
axs[2].set_title('Learning Rate')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Learning Rate')


# 显示图形
plt.tight_layout()
plt.show()
