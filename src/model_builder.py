import torch
from torch import nn

class SimpleMLP(nn.Module):
    def __init__(self, input_shape, hidden_units1, hidden_units2, output_shape):
        super().__init__()
        self.hidden_layer_1 = nn.Sequential(
            nn.Linear(input_shape, hidden_units1),
            nn.ReLU()
        )
        self.hidden_layer_2 = nn.Sequential(
            nn.Linear(hidden_units1, hidden_units2),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(hidden_units2, output_shape)
    
    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.output_layer(x)
        return x
