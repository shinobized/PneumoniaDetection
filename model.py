import torchvision.models as models
import torch.nn as nn
import numpy as np


class MyAlexNet(nn.Module):
    def __init__(self, params):
        super(MyAlexNet, self).__init__()
        self.params = params
        self.pre_layers = models.alexnet(pretrained=True)

        # delete last layer
        self.pre_layers.classifier = self.pre_layers.classifier[:-1]

        for param in self.pre_layers.parameters():
                param.requires_grad = False
        self.my_layer = nn.Linear(4096, params.num_classes)

    def forward(self, x):
        self.pre_output = self.pre_layers(x)
        output = self.my_layer(self.pre_output)
        return output

loss_fn = nn.CrossEntropyLoss()

def metric(output, y):
    y_pred = np.argmax(output, axis=1)
    return np.mean(y == y_pred)

    