import torch.nn as nn
from models.graph_auto_encoder import graph_auto_encoder
from models.graph_neural_network import *

class dummy_model(nn.Module):
    def __init__(self, config):
      super(dummy_model, self).__init__()
      self.flatten = nn.Flatten()
      self.linear_relu_stack = nn.Sequential(
          nn.Linear(128, 128),
      )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

