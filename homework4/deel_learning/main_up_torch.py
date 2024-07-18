from main_torch import yield_tokens, text_pipeline, label_pipeline, collate_batch
import os
import re
import jieba
import time
import numpy as np
import pickle
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

from load_data import load_raw_datasets
from utils import plot_history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# 加载数据
if os.path.exists('../data/data_nn.pkl'):
    with open('../data/data_nn.pkl', 'rb') as f:
        texts, labels,labels_index = pickle.load(f)
else:
    texts, labels,labels_index = load_raw_datasets()
    with open('../data/data_nn.pkl', 'wb') as f:
        pickle.dump((texts, labels,labels_index), f)

def space_tokenizer(text):
    return text.split(' ')

# 使用自定义的空格分词器
tokenizer = space_tokenizer

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

if os.path.exists('../data/vocab.pkl'):
    with open('../data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
else:
    with open('../data/vocab.pkl', 'wb') as f:
        vocab = build_vocab_from_iterator(yield_tokens(texts), specials=["<unk>"])
        vocab.set_default_index(vocab["<unk>"])
        pickle.dump(vocab, f)

MAX_SEQUENCE_LEN = 2000
def text_pipeline(x):
    indices = vocab(tokenizer(x))
    # 裁剪
    if len(indices) > MAX_SEQUENCE_LEN:
        return indices[:MAX_SEQUENCE_LEN]
    # 填充
    elif len(indices) < MAX_SEQUENCE_LEN:
        return indices + [0] * (MAX_SEQUENCE_LEN - len(indices))
    return indices

label_pipeline = lambda x: torch.tensor(x, dtype=torch.long)
        

    
EMBEDDING_DIM = 300 # embedding dimension
MAX_WORDS_NUM = 20000  # 词典的个数
embedding_matrix = np.zeros((MAX_WORDS_NUM+1, EMBEDDING_DIM)) # row 0 for 0
with open("../data/word_vectors.pkl", "rb") as f:
    embeddings_index = pickle.load(f)

for word, i in vocab.get_stoi().items():
    embedding_vector = embeddings_index.get(word)
    if i < MAX_WORDS_NUM:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            

num_class = len(labels_index)
vocab_size = len(vocab)
emsize = EMBEDDING_DIM

import torch.nn.functional as F
class ConvTextClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, embedding_matrix, sequence_length, num_classes):
        super(ConvTextClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # 设置嵌入层为不可训练
        
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=5)
        self.pool1 = nn.MaxPool1d(kernel_size=5)
        
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=5)
        
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5)
        self.pool3 = nn.MaxPool1d(kernel_size=35)  # 根据实际输入长度调整
        
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.embedding(x)  # [batch_size, sequence_length, embedding_dim]
        x = x.permute(0, 2, 1)  # 调整为[batch_size, embedding_dim, sequence_length]以适应Conv1d
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        
        x = torch.flatten(x, 1)  # 展平除batch维度外的所有维度
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = ConvTextClassifier(vocab_size, emsize, embedding_matrix, MAX_SEQUENCE_LEN, num_class).to(device)

def collate_batch(batch):
    label_list, text_list = [], []

    for (_text, _label) in batch:
        # 标签列表
        label_list.append(label_pipeline(_label))

        # 文本列表
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)

    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.cat(text_list)

    return text_list.to(device), label_list.to(device)


EPOCHS = 15  # epoch
LR = 5  # learning rate
BATCH_SIZE = 128  # batch size for training

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
sceduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None

class CustomDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label

    def __len__(self):
        return len(self.texts)

train_dataset = CustomDataset(texts, labels)
split_train , split_valid = random_split(train_dataset, [int(len(train_dataset)*0.9), len(train_dataset) - int(len(train_dataset)*0.9)])

train_dataloader = DataLoader(split_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

# 定义损失函数和优化器
import  time
history = {
    'epoch': [],
    'loss': [],
    'accuracy': [],
    'valid_accuracy': []
}
def train(dataloader):
    model.train()
    total_acc, total_count ,train_loss = 0, 0, 0
    log_interval = 50
    start_time = time.time()

    for idx, (text, cls) in enumerate(dataloader):
        predited_label = model(text)

        optimizer.zero_grad()
        loss = criterion(predited_label, cls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        total_acc += (predited_label.argmax(1) == cls).sum().item()
        train_loss += loss.item()
        total_count += cls.size(0)

        history['epoch'].append(epoch)
        history['loss'].append(loss.item())
        history['accuracy'].append(total_acc/total_count)

        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | accuracy {:5.2f}'.format(
                    epoch, idx, len(dataloader), optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / log_interval,
                    train_loss/total_count,
                    total_acc/total_count))
            # 在for epoch in range(1, EPOCHS + 1):循环的末尾添加
            total_acc, total_count,train_loss = 0, 0, 0
            start_time = time.time()


def evaluate(data_loader):
    model.eval()  # 切换为测试模式
    total_acc, train_loss, total_count = 0, 0, 0

    with torch.no_grad():
        for idx, (text, label, offsets) in enumerate(data_loader):
            predicted_label = model(text,offsets)

            loss = criterion(predicted_label, label)  # 计算loss值
            # 记录测试数据
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            train_loss += loss.item()
            total_count += label.size(0)

    return total_acc / total_count, train_loss / total_count

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val, loss_val = evaluate(valid_dataloader)
    history['valid_accuracy'].append(accu_val)

    lr = optimizer.state_dict()['param_groups'][0]['lr']
    if total_accu is not None and total_accu > accu_val:
        sceduler.step()
    else:
        total_accu = accu_val
    print('-' * 69)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           (time.time() - epoch_start_time),
                                           loss_val,
                                           accu_val))
    print('-' * 69)
plot_history(history)