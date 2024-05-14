import torch.nn as nn
import torch
import numpy as np
from sklearn.metrics import ChunkEvaluator
import os
from transformers import BertForTokenClassification
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




model_name = "bert-base-chinese"
model = BertForTokenClassification.from_pretrained(model_name, num_classes=5)


train_dataset = load_from_disk(file_path+'train_dataset')
dev_dataset = load_from_disk(file_path+'dev_dataset')
test_dataset = load_from_disk(file_path+'test_dataset')
batch_size = 16
train_sampler = RandomSampler(train_dataset)
dev_sampler = RandomSampler(dev_dataset)
test_sampler = RandomSampler(test_dataset)

train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, batch_size=batch_size, collate_fn=collate_fn)
dev_loader = DataLoader(dataset=dev_dataset, sampler=dev_sampler, batch_size=batch_size, collate_fn=collate_fn)
test_loader = DataLoader(dataset=test_dataset, sampler=test_sampler, batch_size=batch_size, collate_fn=collate_fn)

num_epochs = 3
learning_rate = 5e-5
eval_steps = 100
log_steps = 10
save_idrs = "./checkpoints"

weight_decay = 0.01
warmup_proportion = 0.1
num_train_steps = (len(train_loader) * num_epochs)
decay_parameters = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "norm"])]

optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay,apply_decay_param_fun=lambda x: x in decay_parameters)
loss_fn = nn.CrossEntropyLoss()

metric = ChunkEvaluator(label_list=["B","M","E","S"],suffix=False)


def evaluate(model, data_loader, metric):
    model.eval()
    metric.reset()
    precition,recall,f1 = 0,0,0
    for batch_data in data_loader:
        input_ids, token_type_ids, labels ,seq_len = batch_data["input_ids"], batch_data["token_type_ids"], batch_data["labels"],batch_data["seq_len"]
        
        logits = model(input_ids=input_ids, token_type_ids=token_type_ids)
        precition = torch.argmax(logits,dim=-1)
        num_infer_chuncks,num_label_chuncks,num_correct_chuncks = metric.compute(seq_len,precition,labels)
        metric.update(num_infer_chuncks.numpy(),num_label_chuncks.numpy(),num_correct_chuncks.numpy())
        metric(logits, labels)
        precition,recall,f1 = metric.accumulate()
    return precition,recall,f1





def train(model):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    global_step = 0
    best_score = 0
    
    train_loss_record = []
    train_score_record = []
    
    for epoch in range(num_epochs):
        for step,batch_data in enumerate(train_loader):
            input_ids, token_type_ids, labels = batch_data["input_ids"], batch_data["token_type_ids"], batch_data["labels"]
            
            logits = model(input_ids=input_ids, token_type_ids=token_type_ids)
            loss = loss_fn(logits.view(-1, model.num_classes), labels.view(-1))
            
            train_loss_record.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if global_step % log_steps == 0:
                print(f"[Train] epoch:{epoch}/{num_epochs}, step:{global_step}/{num_train_steps}, loss:{loss.item():.5f}")
            if global_step != 0 and (global_step % eval_steps == 0 or global_step == num_train_steps-1):
                precition,recall,f1 = evaluate(model, dev_loader, metric)
                train_score_record.append(f1)
                model.train()
                if f1 > max(train_score_record):
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
    
        
