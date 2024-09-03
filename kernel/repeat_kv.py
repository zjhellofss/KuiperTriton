import torch
import triton
import triton.language as tl


@triton.jit
def repeat_kernel_triton(input, output, repeat, head_dim, block_size: tl.constexpr):
    tid = tl.program_id(0)
    input_ptr = input + tid * head_dim
    output_ptr = output + tid * repeat * head_dim

    block_n = tl.arange(0, block_size)
    for block_idx in range(0, head_dim, block_size):
        offset = block_idx + block_n
        mask = offset < head_dim
        input_ = tl.load(input_ptr + offset, mask)
        for r in range(repeat):
            output_ptr_repeat = output_ptr + r * head_dim + offset
            tl.store(output_ptr_repeat, input_, mask)


def repeat_kv(input, output, repeat):
    bs, seq_len, kv_heads, head_dim = input.shape
    head_dim_blocks = bs * seq_len * kv_heads

    block_size = 32
    repeat_kernel_triton[head_dim_blocks,](input, output, repeat, head_dim, block_size)


if __name__ == '__main__':
    bs = 12
    seq_len = 5
    kv_heads = 16
    repeat = 4
    head_dim = 1024

    input = torch.randn((bs, seq_len, kv_heads, head_dim)).cuda()
    output1 = torch.randn((bs, seq_len, kv_heads * repeat, head_dim)).cuda()
    import time

    # repeat_kv(input, output1, repeat=repeat)

    for i in range(5):
        t1 = time.time()
        repeat_kv(input, output1, repeat=repeat)
        t2 = time.time()

        t3 = time.time()
        output2 = input[:, :, :, None, :].expand(bs, seq_len, kv_heads, repeat, head_dim).contiguous()
        t4 = time.time()

        print('triton time:{}'.format(t2 - t1))
        print('torch time:{}'.format(t4 - t3))
        print('-' * 32)
