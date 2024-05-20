import re
import collections
from bi_MM import BiMM
data_path = './homework3/data/'
file_path = './homework3/PKU_TXT/ChineseCorpus199801.txt'


def get_vocab( file_path):
    vocab = collections.defaultdict(int) 
    with open(data_path+"word_count.txt", 'r', encoding='gbk') as f:
        lines = f.readlines()
        for line in lines:
            word, count = line.split()
            word = re.sub(r'/\w*', '', word)
            vocab[' '.join(list(word)) + '</w>'] = int(count)
    return vocab

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

def get_tokens(vocab):
    tokens = collections.defaultdict(int)
    for word, freq in vocab.items():
        word_tokens = word.split()
        for token in word_tokens:
            tokens[token] += freq
    return tokens

def wash_test(data_path):
    sentences = [[]]
    real_values = [[]]
    with open(data_path,'r',encoding='GBK')as f:
        lines = [line for line in f.readlines() if line.strip()] # 去掉空行
    pattern = re.compile(r'\d{8}-\d{2}-\d{3}-\d{3}/\w')
    special_format = re.compile(r'\[(.*?)\]nt')
    for line in lines:
        if special_format.search(line):
            line = special_format.sub(r'\1', line)
        line = pattern.sub('', line).strip()
        line = re.sub(r'/\w*', '', line)
        real_value = line
        delete = re.sub(r' ','',line)
        if delete != '':
            sentences.append(delete)
            real_values.append(real_value)
    sentences.pop(0)
    real_values.pop(0)
    return sentences,real_values


def bpe(vocab):
    # 使用BPE算法进行子词压缩
    num_merges = 100
    dict = set()
    for i in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
    for key in vocab.keys():
        key = re.sub(r'</\w>', '', key).replace(' ', '')
        dict.add(key)
    model = BiMM(dict)
    return model

        

def evaluate(segments, real_values):
    right_count = 0
    total_count = 0
    pred_count = 0
    dataset = zip(real_values, segments)
    for sample, segment in dataset:
        sample = sample.split()
        segment = segment.split()
        for word in sample:
            total_count += 1
        for word in segment:
            pred_count += 1
            if word in sample:
                right_count += 1
    accuracy = right_count / total_count
    recall = right_count / pred_count
    f1 = 2 * accuracy * recall / (accuracy + recall)
    return accuracy, recall, f1


if __name__ == '__main__':
    dict,char_dict = get_vocab(5000, data_path+"word_count.txt")

