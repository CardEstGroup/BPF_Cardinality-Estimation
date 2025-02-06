import torch
import torch.nn as nn
import numpy
import math


output = [
    [1, 2],
    [3, 4],
    [5, 6]
]
target = [
    [3, 3],
    [3, 9],
    [5, 6]
]

output_Tensor = torch.FloatTensor(output)
target_Tensor = torch.FloatTensor(target)

print(output_Tensor.div(target_Tensor))

# BCEWithLogitsLoss = sigmoid + BCELoss
loss = nn.MSELoss(reduction='sum')
# loss = nn.MSELoss(reduction='mean')
print(loss(output_Tensor, target_Tensor))


