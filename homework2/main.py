import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import random
from collections import Counter
import os
import matplotlib.pyplot as plt
import importlib

# custom import
import text_loader
import LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: " + str(device))

parameters = {
"vocab_size" : 1024,
"embedding_dim" : 16,
"n" : 9,
"input_size" : 16,
"hidden_size" : 64,
"learning_rate": 1e-3,
"batch_size": 512
}

importlib.reload(LSTM)
model = LSTM.LSTM(parameters["vocab_size"], parameters["input_size"], parameters["hidden_size"],parameters["embedding_dim"]).to(device)

importlib.reload(text_loader)
file = "./homework2/PKU_TXT/ChineseCorpus199801.txt"

if os.path.exists("dataset.pkl"):
    dataset = torch.load("dataset.pkl")
else:
    dataset = text_loader.TextLoader(file)
    dataset.generate("gbk", parameters["vocab_size"], parameters["n"], 0.8, parameters["batch_size"], "n_gram")
    dataset.save("dataset.pkl")


cumulative_epoch = None
# prepare trainer
if cumulative_epoch is None:
    cumulative_epoch = 0
x_record = []
y_record = []

trainer = LSTM.Trainer(model,parameters["learning_rate"],device=device)

epochs = 60
learning_rate = 1e-3
for t in range(epochs):
    print(f"Epoch {cumulative_epoch+1}\n-------------------------------")
    trainer.train(dataset.train_loader)
    correct, test_loss = trainer.test(dataset.test_loader)

    cumulative_epoch+=1
    x_record.append(cumulative_epoch)
    y_record.append((correct, test_loss, learning_rate))
    
    if cumulative_epoch % 5 == 0:
        if not os.path.exists('checkpoint'):
            os.makedirs('checkpoint')
        torch.save(trainer.model.state_dict(), './homework2/checkpoint/LSTM-'+str(cumulative_epoch)+'.pth')
print("Done!")
