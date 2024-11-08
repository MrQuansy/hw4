"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union, Iterable

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs

        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * self.scalar * (a ** (self.scalar - 1))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs

        grad_a = out_grad / b
        grad_b = out_grad * (-a / (b * b))

        return grad_a, grad_b
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axes = list(range(len(a.shape)))
        if self.axes is None:
            axes[-1], axes[-2] = axes[-2], axes[-1]
            return a.transpose(axes)
        axes[self.axes[0]] = self.axes[1]
        axes[self.axes[1]] = self.axes[0]
        return a.transpose(axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Permute(TensorOp):
    def __init__(self, axes):
        self.axes = axes

    def compute(self, a):
        return a.permute(self.axes)

    def gradient(self, out_grad, node):
        index = [0] * len(self.axes)
        for i in range(len(self.axes)):
            index[self.axes[i]] = i
        return permute(out_grad, tuple(index))


def permute(a, axes):
    return Permute(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        ret = summation(out_grad, tuple(range(len(out_grad.shape) - len(input_shape))))
        for i in range(len(input_shape)):
            if input_shape[-1 - i] == 1 and self.shape[-1 - i] != 1:
                ret = summation(ret, (len(input_shape) - 1 - i,))
        return reshape(ret, input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        if isinstance(axes, int):
            self.axes = (axes,)

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.sum(axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        if self.axes is None:
            new_shape = (1,) * len(input_shape)
            return broadcast_to(reshape(out_grad, new_shape), input_shape)

        new_shape = list(input_shape)
        for axis in self.axes:
            new_shape[axis] = 1
        return broadcast_to(reshape(out_grad, new_shape), input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs

        a_shape = a.shape
        b_shape = b.shape

        # examples:
        # 1. if a(5,4), b(6,6,4,3)
        # 2. if a(6,6,5,4), b(4,3)
        # 3. if a(6,6,5,4), b(6,6,4,3)

        grad_a = matmul(out_grad, transpose(b))
        if len(b_shape) > 2 and len(a_shape) == 2:
            sum_axes = tuple(range(len(b_shape) - 2))
            grad_a = summation(grad_a, axes=sum_axes)

        grad_b = matmul(transpose(a), out_grad)
        if len(a_shape) > 2 and len(b_shape) == 2:
            sum_axes = tuple(range(len(a_shape) - 2))
            grad_b = summation(grad_b, axes=sum_axes)

        return grad_a, grad_b
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad / a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * exp(a)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a * (a > 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return Tensor((a.cached_data > 0).astype(array_api.float32)) * out_grad
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * ((-(node**2)) + 1)  # out_grad * (1 - tanh^2(x))
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if len(args) > 0:
            shape = args[0].shape
            for arg in args:
                assert arg.shape == shape, "The shape of all tensors should be the same"
            ret_shape = list(shape)
            ret_shape.insert(self.axis, len(args))
            ret = array_api.empty(ret_shape, device=args[0].device)
            for i, arg in enumerate(args):
                slices = [slice(None)] * len(ret_shape)
                slices[self.axis] = i
                ret[tuple(slices)] = arg
            return ret
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        ret = []
        ret_shape = list(A.shape)
        ret_shape.pop(self.axis)
        for i in range(A.shape[self.axis]):
            slices = [slice(None)] * len(A.shape)
            slices[self.axis] = i
            ret.append((A[tuple(slices)]).compact().reshape(ret_shape))
        return tuple(ret)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0:
            return a
        out_shape = list(a.shape)
        for i in self.axes:
            out_shape[i] *= self.dilation + 1
        out = array_api.full(out_shape, 0, device=a.device)
        slices = [slice(None)] * len(a.shape)
        for dim in self.axes:
            slices[dim] = slice(None, None, self.dilation + 1)
        out[tuple(slices)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0:
            return a
        out_shape = list(a.shape)
        for i in self.axes:
            out_shape[i] //= self.dilation + 1
        out = array_api.empty(out_shape, device=a.device)
        slices = [slice(None)] * len(a.shape)
        for dim in self.axes:
            slices[dim] = slice(None, None, self.dilation + 1)
        out = a[tuple(slices)]
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        batch_size, in_height, in_width, in_channel = A.shape
        bs, hs, ws, cs = A.strides
        kernel_height, kernel_width, in_channel, out_channel = B.shape

        pad_A = A.pad(
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0))
        ).compact()
        B = B.compact()
        # img2col
        batch_size, in_height, in_width, in_channel = pad_A.shape
        bs, hs, ws, cs = pad_A.strides
        receiptive_field_shape = (
            batch_size,
            (in_height - kernel_height) // self.stride + 1,
            (in_width - kernel_width) // self.stride + 1,
            kernel_height,
            kernel_width,
            in_channel,
        )
        receiptive_field_strides = (bs, hs * self.stride, ws * self.stride, hs, ws, cs)
        receiptive_field = pad_A.as_strided(
            receiptive_field_shape, receiptive_field_strides
        ).compact()
        reveiptive_vector = receiptive_field.reshape(
            (
                receiptive_field.size // (kernel_height * kernel_width * in_channel),
                kernel_height * kernel_width * in_channel,
            )
        ).compact()

        kernel_vector = B.reshape(
            (kernel_height * kernel_width * in_channel, out_channel)
        ).compact()
        out = reveiptive_vector @ kernel_vector
        return out.reshape(
            (
                batch_size,
                (in_height - kernel_height) // self.stride + 1,
                (in_width - kernel_width) // self.stride + 1,
                out_channel,
            )
        ).compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs
        s, _, _, _ = W.shape

        # X_grad
        W_flipped = flip(W, (0, 1))
        W_flipped_permuted = transpose(W_flipped, (2, 3))
        outgrad_dilated = dilate(out_grad, (1, 2), self.stride - 1)
        X_grad = conv(outgrad_dilated, W_flipped_permuted, padding=s - 1 - self.padding)

        # W_grad
        outgrad_dilated_permuted = permute(outgrad_dilated, (1, 2, 0, 3))
        X_permuted = permute(X, (3, 1, 2, 0))
        W_grad = conv(X_permuted, outgrad_dilated_permuted, padding=self.padding)
        W_grad = permute(W_grad, (1, 2, 0, 3))
        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
