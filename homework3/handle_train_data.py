from collections import Counter
import numpy as np
import os
import torch
# from torchtext.data.utils import get_tokenizer
# from collections import Counter
# from torchtext.vocab import Vocab
import re

data_path = "homework3/PKU_TXT/ChineseCorpus199801.txt"
file_path = "homework3/data/"

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
    if not os.path.exists(file_path+"bmes_dataset.txt"):
        generate_bmes_dataset(file_path, 'gbk')

    