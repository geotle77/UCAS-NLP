import torch
import torch.nn as nn
import numpy as np

class LSTM(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.lstm = nn.LSTM(input_size, hidden_size,  batch_first=True)
        self.liner_relu = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.ReLU()
        )   
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.norm(x)
        x, _ = self.lstm(x)
        x = self.liner_relu(x)
        x = torch.matmul(x, self.embedding.weight.T)
        return x

class FNN(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, embedding_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        x = torch.matmul(x, self.embedding.weight.T)
        return x
    

class RNN(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim, elementwise_affine=False)
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.linear_relu = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.norm(x)
        x, _ = self.rnn(x)
        x = self.linear_relu(x)
        x = torch.matmul(x, self.embedding.weight.T)
        return x

class Trainer():
    def __init__(self, model, learning_rate=0.001,device='cpu',optimizer=None,loss_fn=None):
        self.device = device
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate) if optimizer is None else optimizer
        self.loss_fn = nn.CrossEntropyLoss() if loss_fn is None else loss_fn

    def train(self, data_loader):
        size = len(data_loader.dataset)
        model = self.model
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        model.train()
        for batch, (X, y) in enumerate(data_loader):
            X, y = X.to(self.device), y.to(self.device)
            # Compute prediction error
            pred = model(X)
            # print(pred.shape, y.shape)
            loss = loss_fn(pred.reshape(-1,pred.shape[-1]), y.flatten())
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch+1) % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    def test(self, data_loader):
        size = len(data_loader.dataset)
        model = self.model
        loss_fn = self.loss_fn
        num_batches = len(data_loader)
        
        size = len(data_loader.dataset)
        num_batches = len(data_loader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                test_loss += loss_fn(pred.reshape(-1,pred.shape[-1]), y.flatten()).item()
                correct += (pred.argmax(-1) == y).type(torch.float).sum().item()*pred.shape[0]/np.prod(pred.shape[:-1])
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return correct, test_loss

        
        


