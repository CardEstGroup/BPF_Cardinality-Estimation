import torch
import torch.nn as nn
import numpy
import math


output = [
    [0.3992, 0.2232, 0.6435],
    [0.3800, 0.3044, 0.3241],
    [0.6281, 0.4689, 0.3834]
]
target = [
    [0, 1, 1],
    [0, 0, 1],
    [1, 0, 1]
]

output_Tensor = torch.FloatTensor(output)
target_Tensor = torch.FloatTensor(target)



# BCEWithLogitsLoss = sigmoid + BCELoss
loss = nn.BCEWithLogitsLoss()
print(loss(output_Tensor, target_Tensor))


# BCELoss
output = torch.sigmoid(output_Tensor)
loss = nn.BCELoss()
print(loss(output, target_Tensor))


# BCEWithLogitsLoss的手动版
output = torch.sigmoid(output_Tensor)
loss = 0.0
for i in range(output_Tensor.shape[0]):
    for j in range(output_Tensor.shape[1]):
        x = output[i][j]
        y = target_Tensor[i][j]
        loss = loss + y * torch.log(x) + (1-y) * torch.log(1-x)
loss /= (-1 * output_Tensor.numel())
print(format(loss,'.4f'))

# print(numpy.log(math.e))
# output_Tensor = torch.pow(output_Tensor, math.e)
# print(output_Tensor)
# output_Tensor = torch.log(output_Tensor)
# print(output_Tensor)
# exit(0)