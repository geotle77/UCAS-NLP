import os
import numpy as np
import re
import jieba
import time
import pickle


# jieba.enable_parallel()

token = "[0-9\s+\.\!\/_,$%^*()?;；：【】+\"\'\[\]\\]+|[+——！，;:。？《》、~@#￥%……&*（）“”.=-]+"
labels_index = {}
stop_words = set(open("../dict/stop_words.txt", "r", encoding="utf-8").read().split())

DATA_PATH = "../data/dataset"
BASE_PATH= "../data/CN_Corpus/SogouC.reduced/Reduced"
def preprocess(text):
    text1 = re.sub('&nbsp', ' ', text)
    str_no_punctuation = re.sub(token, ' ', text1)  # 去掉标点
    text_list = list(jieba.cut(str_no_punctuation))  # 分词列表
    text_list = [item for item in text_list if item != ' ']  # 去掉空格
    return ' '.join(text_list)

def load_data():
    X_data = {'train': [], 'test': []}
    y= {'train': [], 'test': []}
    for type_name in ['train', 'test']:
        corpus_dir = os.path.join(DATA_PATH, type_name)
        for label in os.listdir(corpus_dir):
            label_dir = os.path.join(corpus_dir, label)
            file_list = os.listdir(label_dir)
            print(f"loading {type_name} data, label: {label}, file num: {len(file_list)}")

            for file in file_list:
                file_path = os.path.join(label_dir, file)
                with open(file_path, "r", encoding="gb2312",errors='ignore') as f:
                    text_content = preprocess(f.read())
                X_data[type_name].append(text_content)
                y[type_name].append(label)

        print(f"loading {type_name} data done, data num: {len(X_data[type_name])}")

    return X_data['train'], y['train'], X_data['test'], y['test']

def preprocess_keras(text):
    text1 = re.sub('&nbsp', ' ', text)
    str_no_punctuation = re.sub(token, ' ', text1)  # 去掉标点
    text_list = list(jieba.cut(str_no_punctuation))  # 分词列表
    text_list = [item for item in text_list if item != ' ' and item not in stop_words]  # 去掉空格和停用词
    return ' '.join(text_list)

def load_raw_datasets():
    labels = []
    texts = []
    t1= time.time()
    for cate_index,label in enumerate(os.listdir(BASE_PATH)):
        label_dir = os.path.join(BASE_PATH, label)
        file_list = os.listdir(label_dir)
        labels_index[label] = cate_index  # 记录分类标签的整数标号
        print("label: {}, len: {}".format(label, len(file_list)))

        for fname in file_list:
            f = open(os.path.join(label_dir, fname), encoding='gb2312', errors='ignore')
            texts.append(preprocess_keras(f.read()))
            f.close()
            labels.append(labels_index[label])

    t2 = time.time()
    print("load data done, cost time: ", t2-t1)
    return texts, labels,labels_index

def load_pretrained():
    base_idr = "../data/sgns.sogou.word.bz2"
    import bz2
    word_vectors = {}  # 初始化字典来存储词和对应的词向量
    with bz2.open(base_idr, "rt", encoding='utf-8') as f:
        first_line = True
        for line in f:
            if first_line:  # 跳过第一行的基本信息
                first_line = False
                continue
            parts = line.strip().split()  # 分割每一行
            word = parts[0]  # 第一个元素是词
            vector = [float(x) for x in parts[1:]]  # 剩下的元素是词向量
            word_vectors[word] = vector  # 存储到字典中
    with open ("../data/word_vectors.pkl", "wb") as f:
        pickle.dump(word_vectors, f)
    return word_vectors  # 返回包含所有词向量的字典
    
    
if __name__ == '__main__':
    # X_train_data, y_train_data, X_test_data, y_test_data = load_data()
    # pickle.dump((X_train_data, y_train_data, X_test_data, y_test_data), open("./data/load_data.pkl", "wb"))
    load_pretrained()
