import torch
import torch.nn as nn
import copy
class Approach:
    
    def __init__(self, model, criterion=None):
        self.model = model.cuda()
        if criterion == None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion


    def final_loss(self, preds, labels):
        return self.criterion(preds, labels)

    def adapt(self, dataloaders: dict):
        pass
    def __call__(self, input, labels=None):
        if self.model.training:
            if labels is None:
                raise(RuntimeError("When the model is training pass the labels, otherwise set the model to evaluation mode"))
            preds = self.model(input)
            loss = self.final_loss(preds, labels)
            return loss
        return self.model(input)
    
    def train(self):
        self.model.train()
    def eval(self):
        self.model.eval()
    

    
class Naive(Approach):
    pass
        
        
    
