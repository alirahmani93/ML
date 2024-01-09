import torch

from AI import get_device

device = get_device()

x = torch.tensor([[1, 2], [3, 4]], device=device, dtype=torch.int8, requires_grad=True)

print(torch.empty((2, 3)),device=device, )
