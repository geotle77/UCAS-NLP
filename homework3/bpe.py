import re
data_path = './homework3/data/'
file_path = './homework3/PKU_TXT/ChineseCorpus199801.txt'


def get_vocab(vocab_size, file_path):
    with open(data_path+"word_count.txt", 'r', encoding='gbk') as f:
        lines = f.readlines()
        vocab = []
        freq = []
        for line in lines:
            if len(vocab) >= vocab_size:
                break
            word, count = line.split()
            vocab.append(word)
            freq.append(int(count))
    return vocab, freq

def wash_test(vocab,data_path):
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


def bpe(segments):
    # 使用BPE算法进行子词压缩
    squeeze = []
    for segment in segments:
        segment = segment.split()
        while True:
            # 重复生成所有相邻两位的组合并统计频率
            pairs = {}
            for i in range(len(segment) - 1):
                pair = segment[i] + segment[i + 1]
                if pair not in pairs:
                    pairs[pair] = 0
                pairs[pair] += 1
            # 找出频率最高的组合
            max_pair = max(pairs, key=pairs.get)
            # 频率最大值为1则跳出
            if pairs[max_pair] == 1:
                break
            # 将频率最高的组合合并
            new_segment = []
            i = 0
            while i < len(segment):
                if i < len(segment) - 1 and segment[i] + segment[i + 1] == max_pair:
                    new_segment.append(max_pair)
                    i += 2
                else:
                    new_segment.append(segment[i])
                    i += 1
            segment = new_segment
        segment = '/'.join(segment)
        squeeze.append(segment)
    return squeeze

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