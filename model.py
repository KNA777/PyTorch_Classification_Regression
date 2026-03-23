import torch

from torch import nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

tensor = torch.rand([16, 784], dtype=torch.float32)

out = model(tensor)

print(out.shape)

print(model.state_dict()) # Возвращает ссылку на состояние модели
