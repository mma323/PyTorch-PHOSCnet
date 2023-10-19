import torch
import torch.nn as nn
from torch.nn import functional as F


class PHOSCLoss(nn.Module):
    #phos_w = 4.5 phoc_w = 1
    def __init__(self, phos_w=4.5, phoc_w=1):
        super().__init__()

        self.phos_w = phos_w
        self.phoc_w = phoc_w

    def forward(self, y: dict, targets: torch.Tensor):
        phos_loss = self.phos_w * nn.MSELoss()(y['phos'], targets[:, :165])
        
        phoc_loss = self.phoc_w * nn.CrossEntropyLoss()(y['phoc'], targets[:, 165:])

        loss = (phos_loss + phoc_loss)

        return loss
