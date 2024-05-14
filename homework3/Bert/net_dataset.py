import re
from collections import Counter
import numpy as np
import os
import torch
import pickle
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from functools import partial
from datasets import Dataset
import unicodedata
from torch.utils.data import BatchSampler, DataLoader

data_path = "./homework3/PKU_TXT/ChineseCorpus199801.txt"
file_path = "./homework3/data/"
# 定义一个函数来将词语转换为BMES格式
def word2bmes(word):
    if len(word) == 1:
        return ['S']
    elif len(word) == 2:
        return ['B', 'E']
    else:
        return ['B'] + ['M'] * (len(word) - 2) + ['E']

def replace_full_width_numbers(text):
    full_width_numbers = '０１２３４５６７８９'
    half_width_numbers = '0123456789'
    trans_table = str.maketrans(full_width_numbers, half_width_numbers)
    return text.translate(trans_table)

def split_numbers(text):
    return re.sub(r'\d', lambda x: ' ' + x.group() + ' ', text)

def load_data(file_path):
    dataset={}
    text = []
    labels = []
    with open(file_path, 'r', encoding='gbk') as f:
        lines = [line for line in f.readlines() if line.strip()] # 去掉空行
    pattern = re.compile(r'\d{8}-\d{2}-\d{3}-\d{3}/\w')
    special_format = re.compile(r'\[(.*?)\]nt')
    for line in lines:
        line = replace_full_width_numbers(line)  # 将全角数字替换为半角数字
        if special_format.search(line):
            line = special_format.sub(r'\1', line)
        result = pattern.sub('', line)
        words = result.strip().split()
        words = [re.sub(r'/\w*', '', word) for word in words]
        text.append(''.join(word for word in words))
        bmes = ' '.join([tag for word in words for tag in word2bmes(word)])
        labels.append(bmes)
    dataset['text'] = text
    dataset['labels'] = labels

    # 首先，将数据集划分为训练集和一个临时的测试集
    train_text, temp_text, train_labels, temp_labels = train_test_split(dataset['text'], dataset['labels'], test_size=0.2, random_state=42)

    # 然后，将临时的测试集划分为验证集和测试集
    val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, test_size=0.6, random_state=42)

    # 将训练集、验证集和测试集保存在一个字典中
    split_dataset = {
        'train': {'text': train_text, 'labels': train_labels},
        'dev': {'text': val_text, 'labels': val_labels},
        'test': {'text': test_text, 'labels': test_labels}
    }
    return split_dataset

def convert_examples_to_features(example,tokenlizer, label2id,  max_seq_len=512,is_infer=False):
    text = example['text'].strip().split(" ")
    text = [split_numbers(word) for word in text[0]]
    
    # tokens = tokenizer.tokenize(text)
    # print(tokens)
    # print(len(tokens))
    
    encoded_inputs = tokenlizer(text,max_length = max_seq_len,is_split_into_words=True,return_length=True)

    if not is_infer:
        label = [label2id[label] for label in example['labels'].split(" ")] [:max_seq_len-2]
        encoded_inputs['labels'] = [label2id["O"]]+label+[label2id["O"]]
        assert (len(encoded_inputs['input_ids']) == len(encoded_inputs['labels']))
    return encoded_inputs



if __name__ == "__main__":
    if os.path.exists(file_path+'nn_data.pkl'):
        with open(file_path+'nn_data.pkl', 'rb') as f:
            dataset = pickle.load(f)
    else:
        dataset = load_data(data_path)
        with open(file_path+'nn_data.pkl', 'wb') as f:
            pickle.dump(dataset, f)

    dataset['train'] = Dataset.from_dict(dataset['train'])
    dataset['dev'] = Dataset.from_dict(dataset['dev'])
    dataset['test'] = Dataset.from_dict(dataset['test'])

    model_name = "bert-base-chinese"
    label2id = {"O":0,"B":1,"M":2,"E":3,"S":4}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # example = {"text":"钱其琛访问德国","labels":"S B E B E B E"}
    # features = convert_examples_to_features(example,tokenizer,label2id)
    # print(features)

    max_seq_len = 512
    trans_fn = partial(convert_examples_to_features,tokenlizer=tokenizer,label2id=label2id,max_seq_len=max_seq_len,is_infer=False)
    columns = ["text","labels"]
    train_dataset = dataset['train'].map(trans_fn, batched=False, remove_columns=columns)
    dev_dataset =dataset['dev'].map(trans_fn, batched=False, remove_columns=columns)
    test_dataset = dataset['test'].map(trans_fn, batched=False, remove_columns=columns)
    
    train_dataset.save_to_disk(file_path+'train_dataset')
    dev_dataset.save_to_disk(file_path+'dev_dataset')
    test_dataset.save_to_disk(file_path+'test_dataset')
    

    print("train_dataset:",len(train_dataset))
    print("dev_dataset:",len(dev_dataset))
    print("test_dataset:",len(test_dataset))

    
