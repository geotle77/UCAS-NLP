import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import random
from collections import Counter
import os
import matplotlib.pyplot as plt
import importlib
import argparse
import yaml
# custom import
import dataset
import NN


# 创建解析器
parser = argparse.ArgumentParser(description='the train of different model.')
# 添加参数
parser.add_argument('--model', type=str, default='lstm',
                    help='model type: lstm, rnn,fnn.')
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
args = parser.parse_args()

if args.device == 'cuda:0' and not torch.cuda.is_available():
    args.device = 'cpu'
device = args.device
print("Using device: " + str(device))
path = "./homework2/config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

importlib.reload(NN)
importlib.reload(dataset)

if args.model == "lstm":
    parameters = config["lstm"]
elif args.model == "rnn":
    parameters = config["rnn"]
elif args.model == "fnn":
    parameters = config["fnn"]


file = "./homework2/PKU_TXT/ChineseCorpus199801.txt"

# load dataset
if parameters["dataset"] == "n_gram":
    if os.path.exists("dataset_n_gram.pkl"):
        dataset = torch.load("dataset_n_gram.pkl")
    else:
        dataset = dataset.TextLoader(file)
        dataset.generate("gbk", parameters["vocab_size"], parameters["n"], 0.8, parameters["batch_size"], "n_gram")
        dataset.save("dataset_n_gram.pkl")

if os.path.exists("dataset_nn.pkl"):
    dataset = torch.load("dataset_nn.pkl")
else:
    dataset = dataset.TextLoader(file)
    dataset.generate("gbk", parameters["vocab_size"], parameters["n"], 0.8, parameters["batch_size"], "nn")
    dataset.save("dataset_nn.pkl")

if args.model == "lstm":
    model = NN.LSTM(parameters["vocab_size"], parameters["input_size"], parameters["hidden_size"],parameters["embedding_dim"]).to(device)
    trainer = NN.Trainer(model,parameters["learning_rate"],device=device)
elif args.model == "rnn":
    model = NN.RNN(parameters["vocab_size"], parameters["input_size"], parameters["hidden_size"],parameters["embedding_dim"]).to(device)
    trainer = NN.Trainer(model,parameters["learning_rate"],device=device)
elif args.model == "fnn":
    model = NN.FNN(parameters["vocab_size"], parameters["input_size"], parameters["hidden_size"],parameters["embedding_dim"]).to(device)
    trainer = NN.Trainer(model,parameters["learning_rate"],device=device)

cumulative_epoch = None
# prepare trainer
if cumulative_epoch is None:
    cumulative_epoch = 0

# load model
if not os.path.exists("checkpoint"):
    os.mkdir("checkpoint")

pth_list = os.listdir("checkpoint")
latest_pth = None
cumulative_epoch = None
for pth in pth_list:
    if pth.endswith(".pth") and pth.startswith(args.model):
        if latest_pth is None:
            latest_pth = pth
        else:
            id = int(pth.split("-")[-1].split(".")[0])
            latest_id = int(latest_pth.split("-")[-1].split(".")[0])
            if id > latest_id:
                latest_pth = pth
                cumulative_epoch = id

if latest_pth is not None:
    print("load model from checkpoint/" + latest_pth)
    model.load_state_dict(torch.load("checkpoint/" + latest_pth))
    model.eval()
x_record = []
y_record = []

epochs = config["train"]["epochs"]
learning_rate = config["train"]["learning_rate"]
for t in range(epochs):
    print(f"Epoch {cumulative_epoch+1}\n-------------------------------")
    trainer.train(dataset.train_loader)
    correct, test_loss = trainer.test(dataset.test_loader)

    cumulative_epoch+=1
    x_record.append(cumulative_epoch)
    y_record.append((correct, test_loss, learning_rate))
    
    if cumulative_epoch % 5 == 0:
        if not os.path.exists('./homework2/checkpoint'):
            os.makedirs('./homework2/checkpoint')
        torch.save(trainer.model.state_dict(), './homework2/checkpoint/'+args.model+'-'+str(cumulative_epoch)+'.pth')
print("Done!")
