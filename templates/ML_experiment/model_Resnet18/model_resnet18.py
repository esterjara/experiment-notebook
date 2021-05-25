import torch.nn as nn
from torchvision import models
import enb
from enb import ml


class Resnet18(ml.Model):
    def __init__(self, num_classes):
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        super().__init__(param_dict=dict(model=model))

