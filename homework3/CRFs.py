import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump,load   
import os

class CRFsModel:
    def __init__(self):
        self.crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        data = []
        labels = []

        for line in lines:
            char, label = line.strip().split('/')
            data.append(char)
            labels.append(label)

        self.X_train = data
        self.y_train = labels

    def train(self):
        self.crf.fit(self.X_train, self.y_train)
        
    def use_user_dict(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.user_dict = [_.strip() for _ in f.readlines()]

    def segment(self, text):
        # 将文本转换为模型可以接受的格式
        data = list(text)

        # 使用模型进行预测
        labels = self.crf.predict([data])[0]

        # 如果存在用户词典，就使用用户词典进行分词
        if self.user_dict:
            for word in self.user_dict:
                t = len(word)
                for i in range(len(text)-t + 1):
                    if text[i:i+t] == word:
                        labels[i] = "B"
                        for j in range(i+1, i+t):
                            labels[j] = "M"
                        # if i+t+1 < len(text):
                        #     labels[i+t+1] = "B"

        # 根据预测的标签进行分词
        words = []
        word = ''
        for char, label in zip(data, labels):
            if label in ['B', 'S']:
                if word:
                    words.append(word)
                word = char
            else:
                word += char
        if word:
            words.append(word)

        return words
    
    def save_model(self, file_path):
        dump(self.crf, file_path)
    def load_model(self, file_path):
        self.crf = load(file_path)

if __name__ == "__main__":
    crf = CRFsModel()
    crf.load_data("homework3/data/bmes_dataset.txt")
    if os.path.exists("./homework3/train_result/crf_model.joblib"):
        crf.load_model("./homework3/train_result/crf_model.joblib")
    else:
        crf.train()
        crf.save_model("./homework3/train_result/crf_model.joblib")
    crf.use_user_dict("homework3/data/user_data.txt")
    text = "我爱北京天安门"
    words = crf.segment(text)
    print(words)