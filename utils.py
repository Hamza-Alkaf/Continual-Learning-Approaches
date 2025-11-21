import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
import numpy as np
from avalanche.models import SimpleCNN

class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = SimpleCNN(num_classes=num_classes)
        self.model.features[0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    def forward(self, x):
        return self.model(x)

def train_one_epoch(approach, dataloader, optimizer, device):
    approach.train()
    total_loss = 0

    for images, labels, _ in dataloader:
        images, labels = images.to(device), labels.to(device) 
        optimizer.zero_grad() 
        
        loss = approach(images,labels) 
    
        optimizer.step()  
    
        total_loss += loss.item()
    
        
    return total_loss / len(dataloader) 

def train_stream(benchmark, approach, epochs, device='cpu'):
    optimizer = optim.Adam(approach.model.parameters(),lr=0.001)
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream
    approach.set_stream(train_stream, test_stream)
    n_experieneces=benchmark.n_experiences
    R=np.zeros((n_experieneces,n_experieneces))
    for train_exp in train_stream:
        train_loader = DataLoader(train_exp.dataset,batch_size=64,shuffle=True)
        
        for epoch in range(epochs):
            train_loss = train_one_epoch(approach=approach, dataloader=train_loader, optimizer=optimizer, device=device)
            
        approach.adapt()
        R = evaluate_stream(approach=approach, R=R, row=train_exp._current_experience, test_stream=benchmark.test_stream, 
                            device=device)
    return approach, R

def evaluate_stream(approach, R, row, test_stream, device='cpu'):
    approach.eval()
    for test_exp in test_stream:
        correct = 0
        col = test_exp.current_experience
        test_loader = DataLoader(test_exp.dataset,batch_size=64,shuffle=False)
        for X, y, _ in test_loader:
            X, y = X.to(device), y.to(device)
            preds = approach(X)
            correct += torch.sum(preds.argmax(dim=1)==y)
        R[row][col] = correct/len(test_loader.dataset)
    return R

