import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import random
from collections import Counter
import os
import pickle
import importlib
import argparse
import yaml
import sys
# custom import
import dataset
import NN
import evaluate


# 创建解析器
parser = argparse.ArgumentParser(description='the train of different model.')
# 添加参数
parser.add_argument('--model', type=str, default='RNN',
                    help='model type: LSTM,RNN,FNN.')
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument('--is_train', type=bool, default=True)
parser.add_argument('--is_eval', type=bool, default=False)
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


parameters = config["lstm"]
if args.model == "RNN":
    parameters = config["rnn"]
elif args.model == "FNN":
    parameters = config["fnn"]

file = "./homework2/PKU_TXT/ChineseCorpus199801.txt"

# load dataset
if parameters["dataset"] == "n-gram":
    if os.path.exists("./homework2/dataset_n_gram.pkl"):
        dataset = torch.load("./homework2/dataset_n_gram.pkl")
    else:
        dataset = dataset.TextLoader(file)
        dataset.generate("gbk", parameters["vocab_size"], parameters["n"], 0.8, config["train"]["batch_size"], "n_gram")
        dataset.save("./homework2/dataset_n_gram.pkl")
if parameters["dataset"] == "nn":
    if os.path.exists("./homework2/dataset_nn.pkl"):
        dataset = torch.load("./homework2/dataset_nn.pkl")
    else:
        dataset = dataset.TextLoader(file)
        dataset.generate("gbk", parameters["vocab_size"], parameters["n"], 0.8, config["train"]["batch_size"], "nn")
        dataset.save("./homework2/dataset_nn.pkl")


model = NN.LSTM(parameters["vocab_size"], parameters["input_size"], parameters["hidden_size"],parameters["embedding_dim"]).to(device)
trainer = NN.Trainer(model,config["train"]["lr"],device=device)
if args.model == "RNN":
    model = NN.RNN(parameters["vocab_size"], parameters["input_size"], parameters["hidden_size"],parameters["embedding_dim"]).to(device)
    trainer = NN.Trainer(model,config["train"]["lr"],device=device)
elif args.model == "FNN":
    model = NN.FNN(parameters["vocab_size"], parameters["input_size"], parameters["hidden_size"],parameters["embedding_dim"]).to(device)
    trainer = NN.Trainer(model,config["train"]["lr"],device=device)

cumulative_epoch = None
# prepare trainer
if cumulative_epoch is None:
    cumulative_epoch = 0

# train
if args.is_train:
    x_record = []
    y_record = []
    print("start training the model: "+args.model)
    epochs = config["train"]["epochs"]
    learning_rate = config["train"]["lr"]
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
    data = {'x_record': x_record, 'y_record': y_record}
    with open('./homework2/train_info/loss_info_'+args.model, 'wb') as f:
        pickle.dump(data, f)

    

if args.is_eval:
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
    else :
        sys.exit("No trained model available in checkpoint, cannot proceed.")
    
    lookup_table = evaluate.get_lookup_table(trainer.model)
    print(lookup_table.shape)
    word = random.choice(list(dataset.top_words.keys()))
    # word = "５/m"
    print(word)
    evaluate.top_10_similar(evaluate.get_lookup_table(trainer.model), dataset.top_words[word],dataset)