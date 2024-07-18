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

import os,PIL,pathlib,warnings

from load_data import load_raw_datasets

warnings.filterwarnings("ignore")
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

from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset

MAX_WORDS_NUM = 20000  # 词典的个数

# 自定义中文分词函数
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


text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: torch.tensor(x, dtype=torch.long)

def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]

    for (_text, _label) in batch:
        # 标签列表
        label_list.append(label_pipeline(_label))

        # 文本列表
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)

        # 偏移量，即语句的总词汇量
        offsets.append(processed_text.size(0))

    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.cat(text_list)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)  # 返回维度dim中输入元素的累计和

    return text_list.to(device), label_list.to(device), offsets.to(device)

EMBEDDING_DIM = 300 # embedding dimension


# 定义模型
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text,offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

num_class = len(labels_index)
vocab_size = len(vocab)
emsize = 64
epoch =6
model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
# 定义损失函数和优化器
import  time

def train(dataloader):
    model.train()
    total_acc, total_count ,train_loss = 0, 0, 0
    log_interval = 50
    start_time = time.time()

    for idx, (text, cls,offsets) in enumerate(dataloader):
        predited_label = model(text,offsets)

        optimizer.zero_grad()
        loss = criterion(predited_label, cls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        total_acc += (predited_label.argmax(1) == cls).sum().item()
        train_loss += loss.item()
        total_count += cls.size(0)

        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | accuracy {:5.2f}'.format(
                    epoch, idx, len(dataloader), optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / log_interval,
                    train_loss/total_count,
                    total_acc/total_count))
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



EPOCHS = 15  # epoch
LR = 5  # learning rate
BATCH_SIZE = 64  # batch size for training

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

for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val, loss_val = evaluate(valid_dataloader)

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
