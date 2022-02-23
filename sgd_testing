from torch import nn, optim
from  torch.utils.data import Dataset, DataLoader
model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=1e-3)
steps = 303
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 30, T_mult=2, 
                                                           eta_min=1e-5, last_epoch=-1, verbose=False)
#scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
lr=[]
for epoch in range(50):
    for idx in range(steps):
        scheduler.step()
        #print(scheduler.get_lr())
        lr.append(scheduler.get_last_lr())
