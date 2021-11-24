import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self):
        pass
    
    def __getitem__(self, index):
        print(index)
        return torch.randn(1)
    
    def __len__(self):
        return 10


# indices = torch.arange(10)
# sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
dataset = MyDataset()
# loader = DataLoader(dataset, sampler=sampler)

for epoch in range(2):
    print(epoch)
    idx = torch.multinomial(torch.ones(10) / 10, 3)
    sampler = torch.utils.data.sampler.SubsetRandomSampler(idx)
    loader = DataLoader(dataset, sampler=sampler)
    for _ in loader:
        pass
