import torch

import triton
import triton.language as tl


@triton.jit
def attn_qk_generate_triton(q_ptr, q_bs_stride, q_h_stride,
                            k_ptr, k_bs_stride, k_h_stride, k_seq_stride,
                            o_ptr, o_bs_stride, o_h_stride,
                            bs, heads, cache_len, head_dim,
                            BLOCK_SIZE: tl.constexpr):
    """
    在生成阶段时，seq_len = 1
    :param q_ptr: q张量，维度是(bs, heads, 1, head_dim)
    :param q_bs_stride q张量在第0维上每个元素之间的距离
    :param q_h_stride q张量在第1维上每个元素之间的距离
    :param k_ptr: k张量，维度是(bs, heads, 1 + cache_len, head_dim)
    :param o_ptr: 输出张量，维度是(bs, heads, 1 + cache_len)
    :param o_bs_stride: 输出张量在第0维上每个元素的距离
    :param k_bs_stride k张量在第0维上每个元素之间的距离
    :param k_h_stride k张量在第1维上每个元素之间的距离
    :param BLOCK_SIZE: 向量化处理元素的个数
    """

    b_id = tl.program_id(0)
    h_id = tl.program_id(1)
    seq_id = tl.program_id(2)
    if b_id >= bs:
        return

    if h_id >= heads:
        return

    if seq_id >= cache_len + 1:
        return

    q_head_ptr = q_ptr + b_id * q_bs_stride + h_id * q_h_stride
    k_head_ptr = k_ptr + b_id * k_bs_stride + h_id * k_h_stride + seq_id * k_seq_stride
    o_head_ptr = o_ptr + b_id * o_bs_stride + h_id * o_h_stride

    block_n = tl.arange(0, BLOCK_SIZE)
    part_sum = tl.zeros((BLOCK_SIZE,), tl.float32)
    for block_idx in range(0, head_dim, BLOCK_SIZE):
        offset = block_idx + block_n
        mask = offset < head_dim

        key = tl.load(k_head_ptr + offset, mask=mask, other=0.0)
        query = tl.load(q_head_ptr + offset, mask=mask, other=0.0)

        part_sum += key * query
    sum = tl.sum(part_sum)
    tl.store(o_head_ptr + seq_id, sum)


def attn_qk_generate(query: torch.Tensor, key: torch.Tensor, output: torch.Tensor):
    q_bs_stride, q_h_stride = query.stride(0), query.stride(1)
    o_bs_stride, o_h_stride = output.stride(0), output.stride(1)
    k_bs_stride, k_h_stride, k_seq_stride = key.stride(0), key.stride(1), key.stride(2)
    BLOCK_SIZE = 128

    bs, heads, total_len, head_dim = key.shape
    cache_len = total_len - 1
    assert cache_len > 0

    def grid(meta): return bs, heads, cache_len + 1

    attn_qk_generate_triton[grid](query, q_bs_stride, q_h_stride,
                                  key, k_bs_stride, k_h_stride, k_seq_stride,
                                  output, o_bs_stride, o_h_stride,
                                  bs, heads, cache_len, head_dim, BLOCK_SIZE)


if __name__ == '__main__':
    bs = 2
    heads = 8
    seq_len = 1
    cache_len = 4
    head_dim = 16
    q = torch.randn((bs, heads, seq_len, head_dim)).cuda()
    k = torch.randn(bs, heads, seq_len + cache_len, head_dim).cuda()
    o1 = torch.randn(bs, heads, seq_len + cache_len).cuda()
    attn_qk_generate(q, k, o1)
    o2 = q @ (k.transpose(2, 3))

    print(torch.sum(abs(o1 - o2.view(bs, heads, seq_len + cache_len))))
    print(torch.max(abs(o1 - o2.view(bs, heads, seq_len + cache_len))))

