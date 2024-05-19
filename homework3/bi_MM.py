from collections import Counter
import os

data_path = "homework3/PKU_TXT/ChineseCorpus199801.txt"
file_path = "homework3/data/word_count.txt"

def generate_count(data_path, encoding):
    with open(data_path, 'r', encoding=encoding) as f:
           words = f.read().split()
    with open(data_path, 'r', encoding=encoding) as f:
        lines = f.readlines()
            
        word_count = Counter(words)

        with open(file_path+"word_count.txt", 'w',encoding=encoding) as f:
            for word, count in word_count.most_common():
                f.write(word + ' ' + str(count) + '\n')

def return_dict(word_count_file,encoding='gbk'):
    word_dict = set()
    with open(word_count_file, 'r', encoding=encoding) as f:
        for line in f:
            word = line.split('/')[0]  # 提取词语
            if word.startswith('[') and len(word) > 1:  # 如果词语以"["开头，并且长度大于1
                word = word[1:]  # 去除左方括号
            word_dict.add(word)
    return word_dict


class BiMM:
    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.max_word_length = max(len(word) for word in dictionary)

    def forward_max_matching(self, sentence):
        words = []
        i = 0
        while i < len(sentence):
            max_len_word = sentence[i:i+1]
            for j in range(self.max_word_length):
                if i + j < len(sentence) and sentence[i:i+j+1] in self.dictionary:
                    max_len_word = sentence[i:i+j+1]
            words.append(max_len_word)
            i += len(max_len_word)
        return '/'.join(words)

    def backward_max_matching(self, sentence):
        words = []
        i = len(sentence)
        while i > 0:
            max_len_word = sentence[i-1:i]
            for j in range(self.max_word_length):
                if i - j > 0 and sentence[i-j-1:i] in self.dictionary:
                    max_len_word = sentence[i-j-1:i]
            words.insert(0, max_len_word)
            i -= len(max_len_word)
        return '/'.join(words)

    def bi_directional_max_matching(self, sentence):
        forward_words = self.forward_max_matching(sentence)
        backward_words = self.backward_max_matching(sentence)
        if len(forward_words.split('/')) < len(backward_words.split('/')):
            return forward_words
        else:
            return backward_words


if __name__ == "__main__":
    if not os.path.exists(file_path):
        generate_count(data_path, 'gbk')
    word_dict = return_dict(file_path, 'gbk')
    bimm = BiMM(word_dict)
    sentence = "小华为了考试早晨买了一杯小米粥喝，让黄飞鸿蒙题目中有几个苹果，但是郭麒麟刷牙选中华为的就是干净，速度快，每次只挤5g就够用。我喜欢在大城市生活流浪地球不爆炸我就不退缩，平时也看看《东吴京剧》、《大战狼人》、《鸿蒙至尊》等经典电视剧。\\我用中华为的就是便宜实惠，而且每次只用5g，我最喜欢的画家是达芬奇，尤其喜欢他的代表作佛罗伦萨画派蒙娜丽莎。\\秦始皇派蒙恬还原神舟十二对接并顺便一提瓦特改良了蒸汽机。"
    print("Input text:",sentence)
    print("Result:",bimm.bi_directional_max_matching(sentence))
