import torch
import triton
import triton.language as tl

@triton.jit
def bmm_kernel(
        A, B, O,
        M, N, K,
        TILE_M: tl.constexpr, TILE_N: tl.constexpr, TILE_K: tl.constexpr,
):
    pid_b = tl.program_id(2)
    A += pid_b * M * K
    B += pid_b * K * N
    O += pid_b * M * N

    pidx = tl.program_id(0)
    pidy = tl.program_id(1)

    pid_m, pid_n = pidx, pidy

    offs_m = pid_m * TILE_M + tl.arange(0, TILE_M)
    offs_n = pid_n * TILE_N + tl.arange(0, TILE_N)
    offs_k = tl.arange(0, TILE_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    a_ptrs = A + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = B + offs_k[:, None] * N + offs_n[None, :]
    o_ptrs = O + offs_m[:, None] * N + offs_n[None, :]

    num_iters = tl.cdiv(K, TILE_K)
    o = tl.zeros((TILE_M, TILE_N), dtype=tl.float32)
    for _ in range(num_iters):
        mask_k = offs_k < K

        mask_a = mask_m[:, None] & mask_k[None, :]
        mask_b = mask_k[:, None] & mask_n[None, :]

        a = tl.load(a_ptrs, mask_a)
        b = tl.load(b_ptrs, mask_b)

        offs_k += TILE_K
        a_ptrs += TILE_K
        b_ptrs += TILE_K * N

        o += tl.dot(a, b, allow_tf32=False)

    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(o_ptrs, o, mask_c)


def bmm(A, B):
    batch, M, K = A.shape
    _, _, N = B.shape
    A = A.contiguous()
    B = B.contiguous()
    C = torch.empty((batch, M, K), dtype=A.dtype, device=A.device)

    TILE_M, TILE_N, TILE_K = 32, 32, 32
    grid_fn = lambda meta: (
        triton.cdiv(M, TILE_M),
        triton.cdiv(N, TILE_N),
        batch,
    )
    with torch.cuda.device(A.device):
        bmm_kernel[grid_fn](A, B, C, M, N, K, TILE_M, TILE_N, TILE_K)


