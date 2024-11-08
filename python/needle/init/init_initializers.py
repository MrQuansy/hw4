import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    limit = gain * math.sqrt(6.0 / (fan_in + fan_out))
    weights = rand(fan_in, fan_out, low=-limit, high=limit)
    return weights
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    weights = randn(fan_in, fan_out, mean=0, std=std)
    return weights
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    bound = math.sqrt(6 / fan_in)
    if shape is None:
        return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    else:
        return rand(*shape, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    std = math.sqrt(2 / fan_in)
    return randn(fan_in, fan_out, mean=0, std=std)
    ### END YOUR SOLUTION
