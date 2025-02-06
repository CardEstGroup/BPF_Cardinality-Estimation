import torch
from torch.nn import functional as F
from func.myloss import my_CrossEntropyLoss_novar


output = [
    [0.1, 0.2, 0.3],
    [1.4, 0.4, 0.2],
    # [0.6272, -0.1120, 0.3048]
]
target = [
    [1, 2]
    # [1, 2, 0]
]



output_Tensor = torch.FloatTensor(output) # torch.Size([2, 3])
# print(output_Tensor)
# print(output_Tensor.shape)

target_Tensor = torch.LongTensor(target).view(-1) # torch.Size([1, 2]) → torch.Size([2])
# print(target_Tensor)
# print(target_Tensor.shape) # torch.Size([2])

# 归一化
output_sum_Tensor = torch.sum(output_Tensor, dim = 1).unsqueeze(dim=1) # torch.Size([2])  → torch.Size([2, 1])
# print(output_sum_Tensor)
# print(output_sum_Tensor.shape)

# output_sum_Tensor = torch.pow(output_sum_Tensor, 2)
# print(output_sum_Tensor)
# print(output_sum_Tensor.shape)
# exit(0)

output_Tensor = torch.div(output_Tensor, output_sum_Tensor)  # torch.Size([2, 3])
# print(output_Tensor)
# print(output_Tensor.shape) # torch.Size([2, 3])
# exit(0)


# CrossEntropyLoss的手动版1
# ln
log_output_Tensor = torch.log(output_Tensor)
# print(log_output_Tensor)
# print(log_output_Tensor.shape)
# NLLLoss 把label对应值取出，去掉负号，求均值
loss = 0.0
for i in range(target_Tensor.numel()):
    # print(i)
    # print(log_output_Tensor[i][target_Tensor[i].item()])
    loss += log_output_Tensor[i][target_Tensor[i].item()]
loss /= (-1 * target_Tensor.numel())
print(format(loss,'.4f'))


# CrossEntropyLoss的手动版1.1
# ln
log_output_Tensor = torch.log(output_Tensor)
# print(log_output_Tensor)
# print(log_output_Tensor.shape)
# NLLLoss 把label对应值取出，去掉负号，求均值
target_onehot = torch.FloatTensor(2, 3)
target_onehot.zero_()
# print(target_Tensor)
# print(target_Tensor.shape)
target_onehot.scatter_(1, target_Tensor.unsqueeze(1), 1)
# print(target_onehot)
# exit(0)
log_output_onehot = log_output_Tensor.mul(target_onehot)
# print(log_output_onehot)
loss = (-1 * log_output_onehot.sum() / target_Tensor.numel())
print(format(loss,'.4f'))




# CrossEntropyLoss的手动版2
# ln
log_output_Tensor = torch.log(output_Tensor)
# NLLLoss 把label对应值取出，去掉负号，求均值
loss = F.nll_loss(log_output_Tensor, target_Tensor)
print(log_output_Tensor)
print(target_Tensor)
print(format(loss,'.4f'))


# CrossEntropyLoss的手动版3
# ln + NLLLoss 把label对应值取出，去掉负号，求均值
lossfn = my_CrossEntropyLoss_novar(size_average='mean')
loss = lossfn((output_Tensor, None), target_Tensor) # nll_loss
print(format(loss,'.4f'))


