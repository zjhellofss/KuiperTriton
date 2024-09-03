import torch

import triton
import triton.language as tl
import torch
from torch import nn


@triton.jit
def rmsnorm_triton(x_ptr, rms_w_ptr, output_ptr,
                   stride_x_batch, stride_x_m, stride_x_k,
                   stride_rms_w, stride_out_batch, stride_out_m, stride_out_k,
                   head_size, eps, BLOCK_N_SIZE: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    offset_m = pid_b * stride_x_batch + pid_m * stride_x_m
    block_n = tl.arange(0, BLOCK_N_SIZE)
    var = tl.zeros((BLOCK_N_SIZE,), tl.float32)

    for block_idx in range(0, head_size, BLOCK_N_SIZE):
        offset_n = block_idx + block_n
        x_ptr_mask = offset_n < head_size
        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=x_ptr_mask, other=0.0)
        var += x * x

    var = tl.sum(var, axis=0) / head_size
    rstd = 1 / tl.sqrt(var + eps)

    for block_idx in range(0, head_size, BLOCK_N_SIZE):
        offset_n = block_idx + block_n
        x_ptr_mask = offset_n < head_size
        rms_w = tl.load(rms_w_ptr + offset_n * stride_rms_w, mask=x_ptr_mask)

        x = tl.load(x_ptr + offset_m + offset_n * stride_x_k, mask=x_ptr_mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        out = x_hat * rms_w
        out_off = pid_b * stride_out_batch + pid_m * stride_out_m + offset_n * stride_out_k
        tl.store(output_ptr + out_off, out, mask=x_ptr_mask)


def rmsnorm(input: torch.Tensor, weight: torch.Tensor, output: torch.Tensor):
    assert torch.cuda.is_available()
    assert input.is_cuda
    assert output.is_cuda
    batch_size = input.size(0)
    seq_len = input.size(1)
    head_size = input.size(2)

    stride_x_batch = input.stride(0)
    stride_x_m = input.stride(1)
    stride_x_k = input.stride(2)

    stride_rms_w = weight.stride(0)

    stride_out_batch = output.stride(0)
    stride_out_m = output.stride(1)
    stride_out_k = output.stride(2)

    eps = 1e-6
    BLOCK_N_SIZE = 128

    def grid(meta): return batch_size, seq_len

    rmsnorm_triton[grid](input, weight, output, stride_x_batch, stride_x_m, stride_x_k, stride_rms_w, stride_out_batch,
                         stride_out_m, stride_out_k, head_size, eps, BLOCK_N_SIZE)
