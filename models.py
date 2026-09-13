import torch
import torch.nn as nn
import xgboost as xgb


class SimpleNet(nn.Module):
    
    def __init__(self, input_size, l1, output_size):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(input_size, l1, bias=False, dtype=torch.float16)
        self.layer2 = nn.Linear(l1, output_size, bias=False, dtype=torch.float16)

    def forward(self, x):
        y = self.layer1(x)
        y_hat = torch.relu(y)
        scores = self.layer2(y_hat)
        # prob = torch.softmax(scores)
        return scores
