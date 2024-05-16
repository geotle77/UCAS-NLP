import torch.nn as nn
import torch
import numpy as np
import pickle
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import os
from transformers import BertForTokenClassification,BertConfig
from datasets import load_from_disk
from torch.utils.data import BatchSampler, DataLoader,RandomSampler
from DataLoader import collate_fn

file_path = "./homework3/data/"
# class BertForTokenClassification(nn.Module):
#     def __init__(self, bert, num_classes,droupout=None):
#         super(BertForTokenClassification, self).__init__()
#         self.num_classes = num_classes
#         self.bert = bert
#         self.dropout = nn.Dropout(droupout if droupout is not None else self.bert.config["hidden_dropout_prob"])
#         self.classifier = nn.Linear(self.bert.config["hidden_size"], num_classes)
        
#     def forward(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None,):
#         outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids,position_ids=position_ids, attention_mask=attention_mask)
#         sequence_output = outputs[0]
#         sequence_output = self.dropout(sequence_output)
        
#         logits = self.classifier(sequence_output)
#         return logits



# 创建一个配置对象，并设置标记分类的类别数量
config = BertConfig.from_pretrained('bert-base-chinese', num_labels=5)
model = BertForTokenClassification(config)

if os.path.exists(file_path+'train_loader.pkl' and file_path+'dev_loader.pkl' and file_path+'test_loader.pkl'):
    with open(file_path+'train_loader.pkl', 'rb') as f:
        train_loader = pickle.load(f)
    with open(file_path+'dev_loader.pkl', 'rb') as f:
        dev_loader = pickle.load(f)
    with open(file_path+'test_loader.pkl', 'rb') as f:
        test_loader = pickle.load(f)

else:
    train_dataset = load_from_disk(file_path+'train_dataset')
    dev_dataset = load_from_disk(file_path+'dev_dataset')
    test_dataset = load_from_disk(file_path+'test_dataset')

    batch_size = 12

    train_sampler = RandomSampler(train_dataset)
    dev_sampler = RandomSampler(dev_dataset)
    test_sampler = RandomSampler(test_dataset)

    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=batch_size, collate_fn=collate_fn)
    dev_loader = DataLoader(dataset=dev_dataset, sampler=dev_sampler, batch_size=batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(dataset=test_dataset, sampler=test_sampler, batch_size=batch_size, collate_fn=collate_fn)
    with open(file_path+'train_loader.pkl', 'wb') as f:
        pickle.dump(train_loader, f)
    with open(file_path+'dev_loader.pkl', 'wb') as f:
        pickle.dump(dev_loader, f)
    with open(file_path+'test_loader.pkl', 'wb') as f:
        pickle.dump(test_loader, f)

num_epochs = 3
learning_rate = 3e-5
eval_steps = 100
log_steps = 10 
save_idrs = "./homework3/checkpoint"

weight_decay = 0.01
warmup_proportion = 0.1
num_train_steps = (len(train_loader) * num_epochs)
decay_parameters = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]
no_decay = ["bias", "norm"]

# 将需要权重衰减的参数和不需要权重衰减的参数分开
parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

# 创建优化器
optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

label2id = {"O":0,"B":1,"M":2,"E":3,"S":4}
# 创建一个映射，将BMESO标签映射到IOBES标签
bmseo_to_iobes = {'B': 'B', 'M': 'I', 'E': 'E', 'S': 'S', 'O': 'O'}


inverse_label_map = {v: k for k, v in label2id.items()}
def evaluate(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    true_labels, pred_labels = [], []

    for batch_data in data_loader:
        input_ids, token_type_ids, labels, attention_mask = batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device), batch_data[3].to(device)
        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        # 将标签和预测结果添加到列表中
        true_labels.extend(labels.tolist())
        pred_labels.extend(predictions.tolist())

    # 计算各种评估指标
    true_labels_str = [[inverse_label_map[label] for label in sequence] for sequence in true_labels]
    pred_labels_str = [[inverse_label_map[label] for label in sequence] for sequence in pred_labels]
    
    true_labels_iobes = [[bmseo_to_iobes[label] for label in sequence] for sequence in true_labels_str]
    pred_labels_iobes = [[bmseo_to_iobes[label] for label in sequence] for sequence in pred_labels_str]
    
    precision = precision_score(true_labels_iobes, pred_labels_iobes)
    recall = recall_score(true_labels_iobes, pred_labels_iobes)
    f1 = f1_score(true_labels_iobes, pred_labels_iobes)
    
    return precision, recall, f1


def train(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    global_step = 0
    best_score = 0
    
    train_loss_record = []
    train_score_record = []
    
    for epoch in range(num_epochs):
        for step,batch_data in enumerate(train_loader):
            input_ids, token_type_ids, labels ,attention_mask = batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device), batch_data[3].to(device)
            
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits.view(-1, model.num_labels), labels.view(-1))
            
            train_loss_record.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if global_step % log_steps == 0:
                print(f"[Train] epoch:{epoch}/{num_epochs}, step:{global_step}/{num_train_steps}, loss:{loss.item():.5f}")
            if global_step != 0 and (global_step % eval_steps == 0 or global_step == num_train_steps-1):
                precition,recall,f1 = evaluate(model, dev_loader) 
                train_score_record.append(f1)
                model.train()
                model = model.to(device)  # move model back to GPU for training
                if f1 > best_score:
                    print(f"[Evalate] best accuracy performace has been updated:{best_score} -> {f1}")
                    best_score = f1
                    torch.save(model.state_dict(),os.path.join(save_idrs,"best.pth"))   
                    print(f"[Evalate] precision:{precition:.5f}, recall:{recall:.5f}, f1:{f1:.5f}")
            global_step += 1
    save_path = os.path.join(save_idrs,"last.pth")
    torch.save(model.state_dict(),save_path)
    print(f"[Train] last model has been saved:{save_path}")
    
    return train_loss_record,train_score_record
        
train_loss_record,train_score_record = train(model)
    
        
