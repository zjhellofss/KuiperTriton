import torch

import triton
import triton.language as tl


def transpose12_inner(input):
    tl.trans(input, (1, 2))


if __name__ == '__main__':
    pass
