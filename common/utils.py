import torch
import math
from scipy.stats import truncnorm
from scipy.stats import norm
import numpy as np
import torch.distributions.multivariate_normal

def for_each(f, l):
    for x in l:
        f(x)

def hard_target_update(main, target):
    target.load_state_dict(main.state_dict())


def soft_target_update(main, target, tau=0.005):
    for main_param, target_param in zip(main.parameters(), target.parameters()):
        target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)


def layer_norm(layer, std=1.0, bias_const=0.0):
    if isinstance(layer, torch.nn.Sequential):
        for l in layer:
            if isinstance(l, torch.nn.Linear):
                torch.nn.init.orthogonal_(l.weight, std)
                torch.nn.init.constant_(l.bias, bias_const)
    else:
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)


# path: save path   args: argparse config
def write_config(path, args):
    path = path + '/algo_config'
    with  open(path, 'w+') as f:
        f.write(str(vars(args)))
        f.close()


def norm_cdf(x, mean=0, std=1):
    # Computes standard normal cumulative distribution function
    x = (x - mean) / std
    return (1. + torch.erf(x / math.sqrt(2.))) / 2.


def norm_pdf(x, mean=0, std=1):
    # Computes standard normal dense distribution function
    x = (x - mean) / std
    return 1 / (math.sqrt(2 * math.pi) * std) * torch.exp(-x * x / 2)


def normal_logproba(x, mean, logstd):
    std = torch.exp(logstd)
    std_sq = std.pow(2)
    logproba = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
    return logproba.sum(1)


def normal_entropy(logstd):
    return 0.5 + 0.5 * math.log(2 * math.pi) + logstd


def truncated_normal_Z(mean, logstd, a=-1, b=1, std=None):
    if std == None:
        std = torch.exp(logstd)
    F_a = norm_cdf(a, mean, std)
    F_b = norm_cdf(b, mean, std)
    return F_b - F_a


# Tensor: mean and std 1dim [a,b,c,...]
def truncated_normal(mean, logstd, a=-1, b=1):
    std = torch.exp(logstd)
    a, b = (a * torch.ones(1)).expand_as(mean), (b * torch.ones(1)).expand_as(mean)
    a, b = (a - mean) / std, (b - mean) / std
    tensor = torch.Tensor([truncnorm.rvs(a_, b_, loc=m, scale=s, size=mean.shape) for a_, b_, m, s in
                           zip(a, b, mean, std)])
    return tensor.reshape(mean.shape)


def truncated_normal_logprob(x, mean, logstd, a=-1, b=1):
    std = torch.exp(logstd)
    std_sq = std.pow(2)
    logproba_norm = - 0.5 * math.log(2 * math.pi) - logstd - (x - mean).pow(2) / (2 * std_sq)
    # 将norm的概率密度的log限制在定区域内
    logproba = (logproba_norm - torch.log(truncated_normal_Z(mean, logstd, a, b, std))).clamp(-5, 5)
    # logproba = torch.log(norm_pdf(x, mean, std) / truncated_normal_Z(mean, logstd, a, b, std))
    return logproba.sum(1)


def truncated_normal_entropy(mean, logstd, a=-1, b=1):
    std = torch.exp(logstd)
    Z = truncated_normal_Z(mean, logstd, a, b, std)
    a_ = (a - mean) / std
    b_ = (b - mean) / std
    t = (a_ * norm_pdf(a_) - b_ * norm_pdf(b_)) / (2 * Z)
    # t中nan值换为0
    t = torch.where(torch.isnan(t), torch.full_like(t, 0), t)
    # Z中为0的log取一个下界值为-25
    return math.log(math.sqrt(2 * math.pi * math.e)) + logstd + torch.log(Z).clamp_min(-25) + t


if __name__ == '__main__':
    np.random.seed(10)
    torch.manual_seed(10)
    mean = 0
    std = 0.4
    a = 0.1
    print(norm.cdf(-a, mean, std), norm.cdf(a, mean, std), norm.cdf(a, mean, std) - norm.cdf(-a, mean, std))
    print(norm.pdf(0,mean,std))
    # print(torch.exp(torch.Tensor([-5])))

    # # norm cdf方法正确
    # print(norm_cdf(torch.Tensor([1.96])))
    # print(norm.cdf(0.76, 1, 3), norm_cdf(torch.Tensor([0.76]), 1, 3))
    # # norm pdf方法正确
    # print(norm.pdf(0.76, 1, 2), norm_pdf(torch.Tensor([0.76]), 1, 2))
    # #验证truncnorm.rvs的a, b需要提前变换到标准正态分布上
    # a = truncnorm.rvs(-1, 1, size=100)
    # print(a)
    # # 验证a,位于高斯分布很边缘的情况
    # m = 2.2679
    # v = 0.0605
    # a = truncnorm.rvs((-1 - m) / v, (1 - m) / v, m, v, size=100)
    # print(norm.pdf(0.999, m, v))
    # print(norm_cdf(torch.Tensor([1]), mean=2.2679, std=0.0605))
    # # 验证torch.clamp方法
    # print(torch.clamp(torch.Tensor([float('inf')]), -1, 1))
    # # 验证norm的pdf的值域
    # print(norm.pdf(0, 0, 0.0001))
    # # 验证truncated_normal_logprob正确性
    #
    # m = torch.Tensor([2, 2])
    # v = torch.Tensor([2, 2])
    # a = torch.Tensor([-1, -1])
    # b = torch.Tensor([1, 1])
    # a_ = (a - m) / v
    # b_ = (b - m) / v
    # c = truncnorm.rvs(a_, b_, loc=m, scale=v, size=2)
    # print('c', c)
    # print(truncnorm.logpdf(c, a_, b_, loc=m, scale=v),
    #       truncated_normal_logprob(torch.Tensor(c), m, torch.log(torch.Tensor([v])), a, b))
    # # 验证 Z正确性
    # # z = norm.cdf(b) - norm.cdf(a)
    # z = norm.cdf(b, m, v) - norm.cdf(a, m, v)
    # print('Z:', norm.cdf(b, m, v) - norm.cdf(a, m, v), truncated_normal_Z(m, torch.log(torch.Tensor([v])), a, b))
    # # 验证truncated_normal_entropy正确性
    # print(truncnorm.entropy(a_, b_, loc=m, scale=v))
    # # print(math.log(math.sqrt(2 * math.pi * math.e)) + math.log(v) + math.log(z) + (
    # #         a_ * norm.pdf(a_) - b_ * norm.pdf(b_)) / (2 * z))
    # print(truncated_normal_entropy(m, torch.log(torch.Tensor([v])), a, b))
    #
    #
    # # 验证 truncated_normal_Z、truncated_normal_logprob、truncated_normal_entropy无异常值
    # a = torch.randn(100, 1).T[0]
    # mean = torch.randn(100, 1).T[0]
    # logstd = torch.randn(100, 1).T[0]
    # temp = truncated_normal(mean, logstd)
    # # print(temp)
    # z = truncated_normal_Z(mean, logstd)
    # print(z)
    # temp_logprob = truncated_normal_logprob(torch.Tensor(temp), mean, logstd)
    # print(temp_logprob)
    # temp_entropy = truncated_normal_entropy(mean, logstd)
    # print(temp_entropy)
