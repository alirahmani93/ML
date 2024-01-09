import torch
from torch import nn
from torchviz import make_dot

model = nn.Sequential()
model.add_module('W0', nn.Linear(8, 16))  # nodes in layer
model.add_module('tanh', nn.Tanh())  # Activation Function
model.add_module('W1', nn.Linear(16, 1))  # 1 is mean Binary classification
model.add_module('Softmax', nn.Softmax())



x = torch.randn(1, 8)
y = model(x)

make_dot(y.mean(), params=dict(model.named_parameters()))
