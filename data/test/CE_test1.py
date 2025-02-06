import torch
import torch.nn as nn
import numpy
import math
from torch.nn import functional as F


output = [
    [0.1, 0.2, 0.3],
    [1.4, 0.4, 0.2],
    # [0.6272, -0.1120, 0.3048]
]
target = [
    [1, 2]
    # [1, 2, 0]
]



output_Tensor = torch.FloatTensor(output)
target_Tensor = torch.LongTensor(target).view(-1) # torch.Size([1, 2]) → torch.Size([2])
# print(target_Tensor.shape)


# CrossEntropyLoss
# return nll_loss(log_softmax(input, 1), target, weight, None, ignore_index, None, reduction)
loss = nn.CrossEntropyLoss()
print(loss(output_Tensor, target_Tensor)) #tensor(2.3185)



# CrossEntropyLoss的手动版1
# softmax
output = torch.softmax(output_Tensor, dim=1) # dim=1 按行
# print(output)

# log + softmax
output = torch.log(output)
# print(output)

# NLLLoss 把label对应值取出，去掉负号，求均值
loss = 0.0
for i in range(target_Tensor.numel()):
    loss += output[i][target_Tensor[i].item()]
loss /= (-1 * target_Tensor.numel())
print(format(loss,'.4f'))




# CrossEntropyLoss的手动版2
# log + softmax
output = torch.log_softmax(output_Tensor, dim=1) # dim=1 按行
# print(output)

# NLLLoss 把label对应值取出，去掉负号，求均值
loss = 0.0
for i in range(target_Tensor.numel()):
    loss += output[i][target_Tensor[i].item()]
loss /= (-1 * target_Tensor.numel())
print(format(loss,'.4f'))




# CrossEntropyLoss的手动版3
# log + softmax
output = torch.log_softmax(output_Tensor, dim=1) # dim=1 按行
# print(output)

# NLLLoss 把label对应值取出，去掉负号，求均值
loss = F.nll_loss(output, target_Tensor)
print(format(loss,'.4f'))