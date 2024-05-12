from collections import Counter
import numpy as np
import os
import torch
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab

data_path = "homework3/PKU_TXT/ChineseCorpus199801.txt"
file_path = "homework3/data/"

def generate_count(data_path, encoding):
    with open(data_path, 'r', encoding=encoding) as f:
           words = f.read().split()
    with open(data_path, 'r', encoding=encoding) as f:
        lines = f.readlines()
            
        word_count = Counter(words)

        with open(file_path+"word_count.txt", 'w',encoding=encoding) as f:
            for word, count in word_count.most_common():
                f.write(word + ' ' + str(count) + '\n')

def return_dict(word_count_file,encoding='utf-8'):
    word_dict = set()
    with open(word_count_file, 'r', encoding=encoding) as f:
        for line in f:
            word = line.split('/')[0]  # 提取词语
            if word.startswith('[') and len(word) > 1:  # 如果词语以"["开头，并且长度大于1
                word = word[1:]  # 去除左方括号
            word_dict.add(word)
    return word_dict

def generate_bmes_dataset(file_path, encoding):
    # 读取语料库文件
    with open(file_path+'word_count.txt', 'r', encoding=encoding) as f:
        lines = f.readlines()

    # 创建一个列表来存储BMES格式的训练集
    bmes_dataset = []

    # 遍历每一行
    for line in lines:
        # 使用'/'分割词和词性，取第一个元素即词
        word = line.split('/')[0]
        # 根据词的长度进行BMES标注
        if len(word) == 1:
            bmes_dataset.append(word + '/S')
        else:
            bmes_dataset.append(word[0] + '/B')
            for char in word[1:-1]:
                bmes_dataset.append(char + '/M')
            bmes_dataset.append(word[-1] + '/E')

    # 将BMES格式的训练集写入文件
    with open(file_path+'bmes_dataset.txt', 'w', encoding='utf-8') as f:
        for item in bmes_dataset:
            f.write(item + '\n')
        

if __name__ == "__main__":
    if not os.path.exists(file_path+"word_count.txt"):
        generate_count(data_path, 'gbk')
    if not os.path.exists(file_path+"bmes_dataset.txt"):
        generate_bmes_dataset(file_path, 'gbk')

    