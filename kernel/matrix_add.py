import torch

import triton
import triton.language as tl


@triton.jit
def add_triton(in_ptr1, in_ptr2, out_ptr, size, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < size
    in1 = tl.load(in_ptr1 + offset, mask)
    in2 = tl.load(in_ptr2 + offset, mask)
    out = in1 + in2
    tl.store(out_ptr + offset, out, mask)


def add(in1, in2, out):
    assert in1.numel() == in2.numel()
    assert in1.numel() == out.numel()
    size = in1.numel()
    block_size = 128
    block_num = triton.cdiv(size, block_size)
    add_triton[block_num,](in1, in2, out, size, block_size)


if __name__ == '__main__':
    input1 = torch.randn(1, 3, 224, 224).cuda()
    input2 = torch.randn(1, 3, 224, 224).cuda()
    output2 = torch.randn(1, 3, 224, 224).cuda()

    output1 = input1 + input2
    add(input1, input2, output2)
    print(torch.sum(abs(output1.view(-1) - output2.view(-1))))
