from typing import Optional
import torch
from torch.nn import _reduction as _Reduction
from torch.nn.modules.loss import _WeightedLoss
from torch.overrides import (has_torch_function_variadic, handle_torch_function)
Tensor = torch.Tensor


class my_CE_EM(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction', 'para']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', para: dict = {}) -> None:
        super(my_CE_EM, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.para = para

    def forward(self, input: Tensor, target: Tensor, w: Tensor, prior: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        return my_cross_entropy_EM(input, target, w, prior, weight=self.weight, ignore_index=self.ignore_index, reduction=self.reduction, para=self.para)

def my_cross_entropy_EM(
    input: Tensor,
    target: Tensor,
    w: Tensor,
    prior: Tensor,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    para: dict = {},
):

    if has_torch_function_variadic(input, target, para):
        return handle_torch_function(
            my_cross_entropy_EM,
            (input, target, w, prior, para),
            input,
            target,
            w,
            prior,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
            para=para,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    a_m, a_s = input
    # print('a_m:')
    # print(a_m)
    # print('target:')
    # print(target)
    # exit(0)

    # CE = F.nll_loss(torch.log(a_m), target, w, None, ignore_index, None, reduction)
    # return CE

    log_a_m = 10*torch.log(a_m)
    # log_a_m = torch.log(a_m)
    # print('loss log_a_m:')
    # print(log_a_m)
    # exit(0)

    # print('prior:')
    # print(prior)
    # if prior != None: # 贝叶斯估计
    #     target += prior
    # print('target:')
    # print(target)
    # exit(0)

    negative_log_a_m_onehot = log_a_m.mul(target) * (-1)
    # print('loss negative_log_a_m_onehot:')
    # print(negative_log_a_m_onehot)
    # exit(0)

    loss = negative_log_a_m_onehot.sum()
    # print('loss:')
    # print(loss)      
    # exit(0)

    return loss






class my_CrossEntropyLoss_KL_MSE(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', q_s: float = 0.01) -> None:
        super(my_CrossEntropyLoss_KL_MSE, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.q_s = q_s

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        assert self.weight is None or isinstance(self.weight, Tensor)
        return my_cross_entropy_KL_MSE(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction, q_s=self.q_s)

def my_cross_entropy_KL_MSE(
    input: Tensor,
    target: Tensor,
    q_s: float,
    weight: Optional[Tensor] = None,
    size_average: Optional[bool] = None,
    ignore_index: int = -100,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
):

    if has_torch_function_variadic(input, target, q_s):
        return handle_torch_function(
            my_cross_entropy_KL_MSE,
            (input, target, q_s),
            input,
            target,
            q_s,
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction,
        )
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    a_m, a_s = input

    # # KL(N(μ,σ^2)||N(p,ε^2))（前向KL） 行
    # a_s_div_q_s = a_s.div(q_s)
    # return (torch.log(a_s_div_q_s) * (-1) + a_s_div_q_s + torch.pow(a_m - target, 2).div(q_s)).sum()

    # # KL(N(p,ε^2)||N(μ,σ^2)) （后向KL） 行
    # q_s_div_a_s = torch.pow(a_s, -1) * q_s
    # return (torch.log(q_s_div_a_s) * (-1) + q_s_div_a_s + torch.pow(a_m - target, 2).div(a_s)).sum()


    # print(target)
    # print(a_m)
    # 交叉熵 P*log(1/Q) = -P*log/Q
    # print(torch.log(a_m))
    # print(torch.log(a_m).mul(target))
    # print(torch.log(a_m).mul(target) * -1)
    # loss = torch.log(a_m).mul(target) * -1
    # exit(0)

    # 均方误差
    # print('a_m - target')
    # print(a_m - target)
    loss = torch.pow(a_m - target, 2)
    # print(loss)
    # exit(0)

    # KL散度 # P*log(P/Q)
    # print(target.div(a_m)) # P/Q
    # print(torch.log(target.div(a_m))) # log(P/Q)
    # print(torch.log(target.div(a_m)).mul(target)) # P*log(P/Q)
    # print(torch.sum(torch.log(target.div(a_m)).mul(target)))  # P*log(P/Q)
    # loss = torch.log(target.div(a_m)).mul(target)
    # KL散度 # Q*log(Q/P)
    # loss = torch.log(a_m.div(target)).mul(a_m)

    N = 1
    if reduction == "mean":
        N = target.shape[0]
    loss = loss.sum() / N

    return loss