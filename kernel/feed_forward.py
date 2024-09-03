from bmm import bmm
from silu import silu
import torch
from torch import nn
import triton
import triton.language as tl


def inner_product_triton(x, y, out, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    in1 = tl.load(x + offset, mask)
    in2 = tl.load(y + offset, mask)
    out = in1 * in2
    tl.store(out + offset, out, mask)


def inner_product(x, y, out):
    BLOCK_SIZE = 128
    size = x.numel()
    BLOCK_NUM = triton.cdiv(size, BLOCK_SIZE)
    inner_product_triton[BLOCK_NUM,](x, y, out, size, BLOCK_SIZE)


class LayerWeight():
    def __init__(self, dim0, dim1, weight):
        self.dim0 = dim0
        self.dim1 = dim1
        self.weight = weight.t()


def feed_forward(x, w1: LayerWeight, w2: LayerWeight, w3: LayerWeight, out):
    w1_out = bmm(x, w1.weight)
    silu(w1_out)
    w3_out = bmm(x, w3.weight)
    inner_product(w1_out, w3_out, w1_out)
    output = bmm(w1_out, w2.weight)
    return output


if __name__ == '__main__':
    w1 = nn.Linear(dim, hidden_dim, bias=False)
    w2 = nn.Linear(hidden_dim, dim, bias=False)
    w3 = nn.Linear(dim, hidden_dim, bias=False)