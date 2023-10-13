from modules.dataset import phosc_dataset
import torch
from modules.models import PHOSCnet

model = PHOSCnet()

x = torch.randn(5, 50, 250, 3).view(-1, 3, 50, 250)

y = model(x)
print(x)

#print(y['phos'])
#print(y['phoc'])