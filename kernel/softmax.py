import torch

import triton
import triton.language as tl


@triton.jit
def softmax_triton(output_ptr, stride_output_row, input_ptr, stride_input_row,
                   num_cols, block_size: tl.constexpr):
    row_index = tl.program_id(0)
    row_start_ptr = input_ptr + row_index * stride_input_row
    col_offsets = tl.arange(0, block_size)
    input_pointers = row_start_ptr + col_offsets

    row_mask = col_offsets < num_cols
    row = tl.load(input_pointers, mask=row_mask, other=float("-inf"))

    row_max = tl.max(row, axis=0)
    safe_row = row - row_max
    numerator = tl.exp(safe_row)

    denominator = tl.sum(numerator, axis=0)
    sm_out = numerator / denominator

    output_row_ptr = output_ptr + row_index * stride_output_row
    output_pointers = output_row_ptr + col_offsets
    tl.store(output_pointers, sm_out, mask=row_mask)


def softmax(input: torch.Tensor, out: torch.Tensor):
    input_shape = input.shape
    input = input.view(-1, input.shape[-1])
    rows, cols = input.shape
    block_size = triton.next_power_of_2(cols)

    grid = (rows,)
    out = out.view(-1, out.shape[-1])
    softmax_triton[grid](out, out.stride(0), input, input.stride(0), cols, block_size=block_size)

    out.view(input_shape)


if __name__ == '__main__':
    bs = 2
    heads = 8
    seq_len = 8
    head_dim = 128
    input = torch.randn(bs, heads, seq_len, head_dim).cuda()
    output1 = torch.randn(bs, heads, seq_len, head_dim).cuda()
    softmax(input, output1)

    output2 = torch.softmax(input, dim=-1)
    print(torch.sum(abs(output2.view(-1) - output1.view(-1))))
    print(torch.max(abs(output2.view(-1) - output1.view(-1))))
