import torch
import torch.nn as nn
import copy
from Approaches.Approach import Approach
import numpy as np
from torch.utils.data import DataLoader
class MemoryApproach(Approach):
    
    def __init__(self, model, criterion=None, n_samples=100, max_size=1000):
        
        super().__init__(model,criterion)
        self.memory = {}
        
        self.n_samples = n_samples
        self.max_size = max_size
        self.last_task = -1

    def sample_memory(self,dataloader):
        self.last_task+=1
        dataset = dataloader.dataset
        loader = DataLoader(dataset=dataset, batch_size=self.n_samples, shuffle=True)
        x, y, _ = next(iter(loader))

        self.memory[self.last_task] = [(img,label) for img,label in zip(x,y)]

        if self.memory_size() > 1000:
            self.handle_memory_overflow()

        
    def adapt(self, dataloaders: dict):
        train_loader = dataloaders["train"]
        self.sample_memory(train_loader)
    
    def handle_memory_overflow(self):
        n_tasks = len(list(self.memory.keys()))
        n_samples = self.max_size/n_tasks

        for key in list(self.memory.keys()):
            self.memory[key] = self.memory[key][:n_samples]

    def get_batch(self,batch_size):
        unraveled_memory=[]
        for task in self.memory.keys():
            for (x,y) in self.memory[task]:
                unraveled_memory.append((x,y))
        n_samples = min(batch_size,self.memory_size())
        loader = DataLoader(unraveled_memory,shuffle=True,batch_size=n_samples)

        imgs, labels = next(iter(loader))

        return imgs,labels
        


    def memory_size(self):
        n_keys = len(list(self.memory.keys()))
        n_values = len(list(self.memory.values())[self.last_task])
        return n_keys * n_values

            




class ER(MemoryApproach):
    def __init__(self, model, criterion=nn.CrossEntropyLoss(), n_samples=100, max_size=1000, sampling_freq=0.05):
        super().__init__(model,criterion,n_samples,max_size)
        self.sampling_freq = sampling_freq
        
        
    def __call__(self, input, labels=None):
        
        if labels is not None:
            take_from_memory = np.random.random() < self.sampling_freq
            if take_from_memory and self.memory != {}:
                
                batch_size = input.size(0)
                memory_batch = self.get_batch(batch_size=batch_size)
                x, y = memory_batch[0].cuda(), memory_batch[1].cuda()
                input = torch.cat((input,x),dim=0)
                labels = torch.cat((labels,y),dim=0)

            return super().__call__(input, labels)

        return super().__call__(input)

        
        