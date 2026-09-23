import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    
    def __init__(self, input_size, l1, output_size):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(input_size, l1, bias=True, dtype=torch.float32)
        self.layer2 = nn.Linear(l1, output_size, bias=True, dtype=torch.float32)

    def forward(self, x):
        y = self.layer1(x)
        y_hat = torch.relu(y)
        scores = self.layer2(y_hat)
        # prob = torch.softmax(scores, dim=1)
        return scores
