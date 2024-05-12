from handle_train_data import return_dict, generate_count
import os

data_path = "homework3/PKU_TXT/ChineseCorpus199801.txt"
file = "homework3/data/word_count.txt"
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
    if not os.path.exists(file):
        generate_count(data_path, 'gbk')
    word_dict = return_dict(file, 'gbk')
    bimm = BiMM(word_dict)
    sentence = "北京大学生喝进口红酒"

    print(bimm.bi_directional_max_matching(sentence))
