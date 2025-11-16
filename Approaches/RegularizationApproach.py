import torch
import torch.nn as nn
from Approaches.Approach import Approach
from torch.utils.data import DataLoader
class RegularizationApproach(Approach):
    
    def __init__(self, model, criterion=None, device='cpu', lambda_reg=1, alpha=0.5):
        
        super().__init__(model, criterion, device)
        self.lambda_reg = lambda_reg
        self.alpha = alpha
        self.importance = {}
        self.star_vars = {}
        self.curr_task = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.importance[name] = torch.zeros_like(param.data)
                self.star_vars[name] = param.data.clone()
    
    def adapt(self):
        if self.curr_task >= len(self.streams["test"]):
            return
        curr_test_dataset = self.streams["test"][self.curr_task].dataset
        test_loader = DataLoader(curr_test_dataset, batch_size=64)
        self.estimate_importance(test_loader)
        self.update_star_vars()
        self.curr_task+=1
    
    def final_loss(self, preds, labels):
         return self.criterion(preds, labels) + self.penalty()
    
    def estimate_importance(self, dataloader, num_samples=None):
        pass
    
    def update_star_vars(self):
        """Update the optimal parameters after learning a task"""
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.star_vars[name] = param.data.clone()
    
    def penalty(self):
        """Compute MAS regularization penalty"""

        penalty_loss = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.importance:
                penalty_loss += (self.importance[name] * 
                               (param - self.star_vars[name]) ** 2).sum()
        return self.lambda_reg * penalty_loss





class MAS(RegularizationApproach):

    def __init__(self, model, criterion=None, device='cpu', lambda_reg=1, alpha=0.5):
        super().__init__(model=model, criterion=criterion, device=device, lambda_reg=lambda_reg, alpha=alpha)
    
    def estimate_importance(self, dataloader, num_samples=None):
        self.model.eval()
        
        temp_importance = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                temp_importance[name] = torch.zeros_like(param.data)
        
        sample_count = 0
        
        for _, (data, _, _) in enumerate(dataloader):
            if num_samples and sample_count >= num_samples:
                break
                
            data = data.to(self.device)
            
            self.model.zero_grad()

            output = self.model(data)

            output = ((torch.norm(output, p=2, dim=1))**2).mean()
           
            output.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    temp_importance[name] += param.grad.abs() / len(dataloader.dataset)
            
            sample_count += len(data)
           
        
        for name in temp_importance:
            if name in self.importance:
                self.importance[name] = (
                    self.alpha * temp_importance[name] + 
                    (1 - self.alpha) * self.importance[name]
                )
            else:
                self.importance[name] = temp_importance[name] 




class EWC(RegularizationApproach):
    """
    Elastic Weights Consolidation (EWC)
    """
    
    def __init__(self, model, criterion=None, device='cpu', lambda_reg=1, alpha=0.5):
        super().__init__(model=model, criterion=criterion, device=device, lambda_reg=lambda_reg, alpha=alpha)
        self.ce = nn.CrossEntropyLoss()
    
    def estimate_importance(self, dataloader, num_samples=None):
        self.model.eval()

        
        temp_importance = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                temp_importance[name] = torch.zeros_like(param.data)
        
        sample_count = 0
        
        for _, (data, labels, _) in enumerate(dataloader):
            if num_samples and sample_count >= num_samples:
                break
                
            data = data.to(self.device)
            labels = labels.to(self.device)
            
            output = self.model(data)

            self.model.zero_grad()
            
            loss = self.ce(output,labels)
            
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    temp_importance[name] += (param.grad.data.clone() / len(dataloader.dataset)).pow(2)
            
            sample_count += len(data)
           
        
        for name in temp_importance:
            if name in self.importance:
                self.importance[name] = (
                    self.alpha * temp_importance[name] + 
                    (1 - self.alpha) * self.importance[name]
                )
            else:
                self.importance[name] = temp_importance[name]     