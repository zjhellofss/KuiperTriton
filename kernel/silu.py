import torch

import triton
import triton.language as tl


@triton.jit
def silu_kernel(input_ptr,
                output_ptr,
                n_elements,
                BLOCK_SIZE: tl.constexpr,
                ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    output = x * tl.sigmoid(x)
    tl.store(output_ptr + offsets, output, mask=mask)


def silu(x: torch.Tensor):
    output = torch.empty_like(x)
    assert torch.cuda.is_available()
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    silu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


def silu_inplace(x: torch.Tensor):
    assert torch.cuda.is_available()
    assert x.is_cuda
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    silu_kernel[grid](x, x, n_elements, BLOCK_SIZE=1024)