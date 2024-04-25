from collections import Counter
import torch
import torch.utils.data as Data
import pickle
import numpy as np

class TextLoader:
    def __init__(self, file) :
        self.file = file

    def generate(self, encoding, vocab_size, n, ratio,batch_size=512,type="n_gram" ):
        
        with open(self.file, 'r', encoding=encoding) as f:
           words = f.read().split()
        with open(self.file, 'r', encoding=encoding) as f:
           lines = f.readlines()
            
        word_count = Counter(words)

        with open('word_count.txt', 'w') as f:
            for word, count in word_count.most_common():
                f.write(word + ' ' + str(count) + '\n')
        
        x = []
        y = []
        
        self.top_words = {word: i+1 for i, (word, _) in enumerate(word_count.most_common(vocab_size - 1))}

        if type=="nn":
            # n-gram
            for line in lines:
                words = line.split()
                if len(words) >= n:
                    for i in range(len(words)-n+1):
                        x.append([self.top_words.get(word, 0) for word in words[i:i+n-1]])
                        y.append([self.top_words.get(word, 0) for word in words[i+1:i+n]])
        
        if type=="n_gram":
            # n-gram
            for line in lines:
                words = line.split()
                if len(words) >= n:
                        for i in range(len(words)-n+1):
                            x.append([self.top_words.get(word, 0) for word in words[i:i+n-1]])
                            y.append(self.top_words.get(words[i+n-1], 0))

        print("x shape:", np.array(x).shape)
        print("y shape:", np.array(y).shape)
        
        dataset = Data.TensorDataset(torch.tensor(x), torch.tensor(y))
        train_size = int(ratio * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        self.train_loader = train_loader
        self.test_loader = test_loader

    def save(self, obj_file):
        torch.save(self, obj_file)