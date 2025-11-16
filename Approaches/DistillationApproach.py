import torch
import torch.nn as nn
from Approaches.Approach import Approach
from copy import deepcopy
import torch.nn.functional as F
class LwF(Approach):
    def __init__(self, model, criterion=None, device='cpu', lambda_distill=1):
        super().__init__(model, criterion, device)
        self.old_model = None
        self.lambda_distill = lambda_distill
        
    
    def adapt(self):
        self.old_model = deepcopy(self.model)

    def distillation_loss(self, preds, old_preds, T=2):
        old = F.softmax(old_preds / T, dim=1)
        new = F.log_softmax(preds / T, dim=1)
        return F.kl_div(new, old, reduce="mean") * (T**2)
    
    def final_loss(self, preds, labels, old_preds):
        task_loss = super().final_loss(preds, labels)
        distill_loss = self.distillation_loss(preds, old_preds)
        return task_loss + self.lambda_distill * distill_loss
    
    def __call__(self, input, labels=None):
        if self.model.training:
            if labels is None:
                raise(RuntimeError("When the model is training pass the labels, " \
                "otherwise set the model to evaluation mode"))
            
            preds = self.model(input)
            if self.old_model is not None:
                old_preds = self.old_model(input)
                loss = self.final_loss(preds, labels, old_preds)
            else :
                loss = super().final_loss(preds, labels)
            loss.backward()
            return loss
        
        return self.model(input)


        