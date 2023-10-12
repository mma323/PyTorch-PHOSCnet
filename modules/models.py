import torch

import torch.nn as nn

from modules.pyramidpooling import TemporalPyramidPooling

from timm.models.registry import register_model

__all__ = [
    'PHOSCnet_temporalpooling'
]


class PHOSCnet(nn.Module):
    def __init__(self):
       super(PHOSCnet, self).__init__()

       self.conv = nn.Sequential(
           nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1),
           nn.ReLU(),
           nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
           nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
           nn.ReLU(),
           nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
           nn.ReLU(),
           nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
           nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
           nn.ReLU(),
           nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
           nn.ReLU(),
           nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
           nn.ReLU(),
           nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
           nn.ReLU(),
           nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
           nn.ReLU(),
           nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
           nn.ReLU(),
           nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1),
           nn.ReLU(),
           nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
           nn.ReLU(),
           nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1),
           nn.ReLU(),
           nn.Flatten()
        )

       self.temporal_pool = TemporalPyramidPooling([1, 2, 4])

       in_features_from_conv = 4096
        
       self.phos = nn.Sequential(
           nn.Linear(in_features=in_features_from_conv, out_features=4096),
           nn.ReLU(),
           nn.Dropout(p=0.5),
           nn.Linear(in_features=4096, out_features=4096),
           nn.ReLU(),
           nn.Dropout(p=0.5),
           nn.Linear(in_features=4096, out_features=165),
           nn.ReLU()
        )

       self.phoc = nn.Sequential(
              nn.Linear(in_features=in_features_from_conv, out_features=4096),
              nn.ReLU(),
              nn.Dropout(p=0.5),
              nn.Linear(in_features=4096, out_features=4096),
              nn.ReLU(),
              nn.Dropout(p=0.5),
              nn.Linear(in_features=4096, out_features=604),
              nn.Sigmoid()
        )


    def forward(self, x: torch.Tensor) -> dict:
        x = self.conv(x)
        x = self.temporal_pool(x)

        return {'phos': self.phos(x), 'phoc': self.phoc(x)}


@register_model
def PHOSCnet_temporalpooling(**kwargs):
    return PHOSCnet()


if __name__ == '__main__':
    model = PHOSCnet()

    x = torch.randn(5, 50, 250, 3).view(-1, 3, 50, 250)

    y = model(x)

    print(y['phos'].shape)
    print(y['phoc'].shape)
