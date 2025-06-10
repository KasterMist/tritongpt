import torch
import triton
import triton.language as tl
from typing import Tuple


@triton.jit
def flash_attn_fwd(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    m_ptr,
    l_ptr,
    qkv_stride_b,
    qkv_stride_h,
    qkv_stride_sq,
    qkv_stride_hd,
    ml_stride_b,
    ml_stride_h,
    BLOCK_HD: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    head_scale,
    num_head,
    context_sq,
):
    """Flash Attention with causal masking."""

    q_chunk_pid = tl.program_id(axis=0)  # parallelize across sq chunks
    bh_pid = tl.program_id(axis=1)  # parallelize across batch x heads

    off_bs = (bh_pid // num_head,)
    off_h = (bh_pid % num_head,)

    bh_offset = off_bs.to(tl.int64) * qkv_stride_b + off_h.to(tl.int64) * qkv_stride_h

    q_block_ptr = tl.make_block_ptr(
        q_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(q_chunk_pid * BLOCK_SQ, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        k_ptr + bh_offset,
        shape=(BLOCK_HD, context_sq),
        block_shape=(BLOCK_HD, BLOCK_SQ),
        strides=(qkv_stride_hd, qkv_stride_sq),
        order=(0, 1),
        offsets=(0, 0),
    )

    v_block_ptr = tl.make_block_ptr(
        v_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(0, 0),
    )

    out = tl.zeros([BLOCK_SQ, BLOCK_HD], dtype=tl.float32)
    m_i = tl.full([BLOCK_SQ], value=float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_SQ], dtype=tl.float32)

    q = tl.load(q_block_ptr, boundary_check=(0, 1))

    # scale by 1/ln(2), 2^x much faster than e^x
    ln2_inv: tl.constexpr = 1.44269504

    head_scale *= ln2_inv
    head_scale = head_scale.to(tl.float16)

    q *= head_scale
    max_range = context_sq
    max_range = q_chunk_pid * BLOCK_SQ + 1

    offs_k = tl.arange(0, BLOCK_SQ)
    offs_q = tl.arange(0, BLOCK_SQ)

    for chunk in range(0, max_range - 1, BLOCK_SQ):
        k = tl.load(
            k_block_ptr,
        )
        v = tl.load(
            v_block_ptr,
        )

        s_ij = tl.dot(q, k, allow_tf32=False)  # [BLOCK_SQ, BLOCK_SK]

        m_ij = tl.max(s_ij, axis=1)  # [BLOCK_SQ, ]
        p_ij = tl.math.exp2(s_ij - m_ij[:, None])  # [BLOCK_SQ, BLOCK_SK]
        l_ij = tl.sum(p_ij, axis=1)  # [BLOCK_SQ, ]

        m_i_new = tl.maximum(m_i, m_ij)

        running_correction = tl.math.exp2(m_i - m_i_new)
        new_correction = tl.math.exp2(m_ij - m_i_new)

        l_i_new = (running_correction * l_i) + (new_correction * l_ij)

        out = (l_i * running_correction)[:, None] * out

        out += new_correction[:, None] * tl.dot(
            p_ij.to(tl.float16), v, allow_tf32=False
        )

        out /= (l_i_new)[:, None]

        m_i = m_i_new
        l_i = l_i_new

        k_block_ptr = tl.advance(k_block_ptr, offsets=(0, BLOCK_SQ))
        v_block_ptr = tl.advance(v_block_ptr, offsets=(BLOCK_SQ, 0))

    # final block - we reuse code here to remove conditionals from for loop
    k = tl.load(
        k_block_ptr,
    )
    v = tl.load(
        v_block_ptr,
    )
    s_ij = tl.dot(q, k, allow_tf32=False)  # [BLOCK_SQ, BLOCK_SK]
    offs = max_range - 1
    s_ij = tl.where(
        q_chunk_pid * BLOCK_SQ + offs_k[:, None] >= (offs + offs_q[None, :]),
        s_ij,
        float("-inf"),
    )

    m_ij = tl.max(s_ij, axis=1)  # [BLOCK_SQ, ]
    p_ij = tl.math.exp2(s_ij - m_ij[:, None])  # [BLOCK_SQ, BLOCK_SK]
    l_ij = tl.sum(p_ij, axis=1)  # [BLOCK_SQ, ]
    m_i_new = tl.maximum(m_i, m_ij)
    running_correction = tl.math.exp2(m_i - m_i_new)
    new_correction = tl.math.exp2(m_ij - m_i_new)
    l_i_new = (running_correction * l_i) + (new_correction * l_ij)
    out = (l_i * running_correction)[:, None] * out
    out += new_correction[:, None] * tl.dot(p_ij.to(tl.float16), v, allow_tf32=False)
    out /= (l_i_new)[:, None]
    m_i = m_i_new
    l_i = l_i_new

    out_block_ptr = tl.make_block_ptr(
        o_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(q_chunk_pid * BLOCK_SQ, 0),
    )

    tl.store(
        out_block_ptr,
        value=out.to(tl.float16),
    )

    bh_offset = off_bs.to(tl.int64) * ml_stride_b + off_h.to(tl.int64) * ml_stride_h

    # store m and l which are used in backward pass to recreate softmax activation
    m_ptr_start = m_ptr + (bh_offset) + (q_chunk_pid * BLOCK_SQ)
    l_ptr_start = l_ptr + (bh_offset) + (q_chunk_pid * BLOCK_SQ)

    tl.store(m_ptr_start + offs_q, m_i)
    tl.store(l_ptr_start + offs_q, l_i)


def flash_wrapper_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Function wrapping causal Flash Attention kernel."""
    batch, nh, sq, hd = q.shape

    BLOCK_HD = triton.next_power_of_2(hd)
    BLOCK_SQ = 64 if BLOCK_HD < 128 else 32
    num_warps = 4 if BLOCK_HD <= 128 else 8

    assert (
        sq % BLOCK_SQ == 0
    ), f"Number of elements in sequence must be a multiple of {BLOCK_SQ}"

    out = torch.empty_like(q)

    def grid(META):
        return (triton.cdiv(sq, META["BLOCK_SQ"]), batch * nh)

    m = torch.empty((batch, nh, sq), device=q.device, dtype=torch.float16)
    l = torch.empty_like(m)

    head_scale = 1.0 / (q.shape[-1] ** 0.5)

    flash_attn_fwd[grid](
        q,
        k,
        v,
        out,
        m,
        l,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        m.stride(0),
        m.stride(1),
        BLOCK_HD=BLOCK_HD,
        BLOCK_SQ=BLOCK_SQ,
        num_warps=num_warps,
        num_stages=2,
        head_scale=head_scale,
        context_sq=sq,
        num_head=nh,
    )

    return out, m, l


@triton.jit
def flash_attn_bwd(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    m_ptr,
    l_ptr,
    dO_ptr,
    dV_ptr,
    dK_ptr,
    dQ_ptr,
    qkv_stride_b,
    qkv_stride_h,
    qkv_stride_sq,
    qkv_stride_hd,
    ml_stride_b,
    ml_stride_h,
    head_scale,
    BLOCK_HD: tl.constexpr,
    BLOCK_SQ: tl.constexpr,
    context_sq,
    num_head,
):
    """Flash Attention backward pass with causal masking"""

    kv_chunk_pid = tl.program_id(axis=0)  # parallelize across kv chunks
    bh_pid = tl.program_id(axis=1)  # parallelize across batch x heads

    off_bs = (bh_pid // num_head,)
    off_h = (bh_pid % num_head,)

    bh_offset = off_bs.to(tl.int64) * qkv_stride_b + off_h.to(tl.int64) * qkv_stride_h

    q_block_ptr = tl.make_block_ptr(
        q_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(0, 0),
    )

    dout_block_ptr = tl.make_block_ptr(
        dO_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(0, 0),
    )

    out_block_ptr = tl.make_block_ptr(
        o_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(0, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        k_ptr + bh_offset,
        shape=(BLOCK_HD, context_sq),
        block_shape=(BLOCK_HD, BLOCK_SQ),
        strides=(qkv_stride_hd, qkv_stride_sq),
        order=(0, 1),
        offsets=(0, kv_chunk_pid * BLOCK_SQ),
    )

    v_block_ptr = tl.make_block_ptr(
        v_ptr + bh_offset,
        shape=(BLOCK_HD, context_sq),
        block_shape=(BLOCK_HD, BLOCK_SQ),
        strides=(qkv_stride_hd, qkv_stride_sq),
        order=(0, 1),
        offsets=(0, kv_chunk_pid * BLOCK_SQ),
    )

    dV = tl.zeros([BLOCK_SQ, BLOCK_HD], dtype=tl.float32)
    dK = tl.zeros([BLOCK_SQ, BLOCK_HD], dtype=tl.float32)

    k_trans = tl.load(k_block_ptr, boundary_check=(1,))
    v = tl.load(v_block_ptr, boundary_check=(1,))

    # constants so tl.math.exp2 can be used instead of tl.exp
    ln2_inv: tl.constexpr = 1.44269504
    ln2: tl.constexpr = 0.6931471824645996

    head_scale *= ln2_inv
    head_scale = head_scale.to(tl.float16)
    max_range = context_sq
    min_range = kv_chunk_pid * BLOCK_SQ

    offs_k = tl.arange(0, BLOCK_SQ)
    offs_q = (kv_chunk_pid * BLOCK_SQ) + tl.arange(0, BLOCK_SQ)

    ml_bh_offset = off_bs.to(tl.int64) * ml_stride_b + off_h.to(tl.int64) * ml_stride_h

    m_ptr_start = m_ptr + ml_bh_offset
    l_ptr_start = l_ptr + ml_bh_offset

    # loop is split into pre/post masking to remove conditional use
    for q_chunk in range(0, min_range + 1, BLOCK_SQ):
        q = tl.load(
            q_block_ptr,
        )
        dout = tl.load(
            dout_block_ptr,
        )
        out = tl.load(
            out_block_ptr,
        )
        q *= head_scale

        m_i = tl.load(m_ptr_start + q_chunk + offs_k, mask=offs_k < context_sq)[:, None]
        l_i = tl.load(l_ptr_start + q_chunk + offs_k, mask=offs_k < context_sq)[:, None]
        s_ij = tl.dot(q, k_trans, allow_tf32=False)

        s_ij = tl.where(
            (q_chunk + offs_k[:, None]) >= (offs_q[None, :]),
            s_ij,
            float("-inf"),
        )

        P_ij = (1.0 / l_i) * tl.math.exp2(s_ij - m_i)

        dV += tl.dot(tl.trans(P_ij.to(tl.float16)), dout, allow_tf32=False)

        dP_ij = tl.dot(dout, v, allow_tf32=False)
        D_i = tl.sum(dout * out, axis=1)[:, None]
        dS_ij = P_ij * (dP_ij - D_i)

        dK += tl.dot(tl.trans(dS_ij.to(tl.float16)), q, allow_tf32=False)

        q_block_ptr = tl.advance(q_block_ptr, offsets=(BLOCK_SQ, 0))
        out_block_ptr = tl.advance(out_block_ptr, offsets=(BLOCK_SQ, 0))
        dout_block_ptr = tl.advance(dout_block_ptr, offsets=(BLOCK_SQ, 0))

    min_range_offset = min_range + BLOCK_SQ

    for q_chunk in range(min_range_offset, max_range, BLOCK_SQ):
        q = tl.load(
            q_block_ptr,
        )
        dout = tl.load(
            dout_block_ptr,
        )
        out = tl.load(
            out_block_ptr,
        )
        q *= head_scale

        m_i = tl.load(m_ptr_start + q_chunk + offs_k, mask=offs_k < context_sq)[:, None]
        l_i = tl.load(l_ptr_start + q_chunk + offs_k, mask=offs_k < context_sq)[:, None]
        s_ij = tl.dot(q, k_trans, allow_tf32=False)

        P_ij = (1.0 / l_i) * tl.math.exp2(s_ij - m_i)

        dV += tl.dot(tl.trans(P_ij.to(tl.float16)), dout, allow_tf32=False)

        dP_ij = tl.dot(dout, v, allow_tf32=False)
        D_i = tl.sum(dout * out, axis=1)[:, None]
        dS_ij = P_ij * (dP_ij - D_i)

        dK += tl.dot(tl.trans(dS_ij.to(tl.float16)), q, allow_tf32=False)

        q_block_ptr = tl.advance(q_block_ptr, offsets=(BLOCK_SQ, 0))
        out_block_ptr = tl.advance(out_block_ptr, offsets=(BLOCK_SQ, 0))
        dout_block_ptr = tl.advance(dout_block_ptr, offsets=(BLOCK_SQ, 0))

    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(kv_chunk_pid * BLOCK_SQ, 0),
    )

    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(kv_chunk_pid * BLOCK_SQ, 0),
    )

    tl.store(
        dV_block_ptr,
        value=dV.to(tl.float16),
    )
    tl.store(
        dK_block_ptr,
        value=(ln2 * dK).to(tl.float16),
    )

    # ----------
    # compute dQ
    # ----------

    # reset block pointers
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(kv_chunk_pid * BLOCK_SQ, 0),
    )

    q_block_ptr = tl.make_block_ptr(
        q_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(kv_chunk_pid * BLOCK_SQ, 0),
    )

    dout_block_ptr = tl.make_block_ptr(
        dO_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(kv_chunk_pid * BLOCK_SQ, 0),
    )
    out_block_ptr = tl.make_block_ptr(
        o_ptr + bh_offset,
        shape=(context_sq, BLOCK_HD),
        block_shape=(BLOCK_SQ, BLOCK_HD),
        strides=(qkv_stride_sq, qkv_stride_hd),
        order=(1, 0),
        offsets=(kv_chunk_pid * BLOCK_SQ, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        k_ptr + bh_offset,
        shape=(BLOCK_HD, context_sq),
        block_shape=(BLOCK_HD, BLOCK_SQ),
        strides=(qkv_stride_hd, qkv_stride_sq),
        order=(0, 1),
        offsets=(0, 0),
    )

    v_block_ptr = tl.make_block_ptr(
        v_ptr + bh_offset,
        shape=(BLOCK_HD, context_sq),
        block_shape=(BLOCK_HD, BLOCK_SQ),
        strides=(qkv_stride_hd, qkv_stride_sq),
        order=(0, 1),
        offsets=(0, 0),
    )

    q = tl.load(
        q_block_ptr,
    )
    q *= head_scale

    dQ = tl.zeros([BLOCK_SQ, BLOCK_HD], dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_SQ)

    max_range = kv_chunk_pid * BLOCK_SQ + 1
    final = max_range - 1

    m_ptr_start = m_ptr + (ml_bh_offset) + (kv_chunk_pid * BLOCK_SQ)
    l_ptr_start = l_ptr + (ml_bh_offset) + (kv_chunk_pid * BLOCK_SQ)

    m_i = tl.load(m_ptr_start + offs_k, mask=offs_k < context_sq)[:, None]
    l_i = tl.load(l_ptr_start + offs_k, mask=offs_k < context_sq)[:, None]

    dout = tl.load(
        dout_block_ptr,
    )
    out = tl.load(
        out_block_ptr,
    )

    for q_chunk in range(0, final, BLOCK_SQ):
        v_trans = tl.load(
            v_block_ptr,
        )
        k_trans = tl.load(
            k_block_ptr,
        )

        s_ij = tl.dot(q, k_trans, allow_tf32=False)

        P_ij = (1.0 / l_i) * tl.math.exp2(s_ij - m_i)

        dP_ij = tl.dot(dout, v_trans, allow_tf32=False)
        D_i = tl.sum(dout * out, axis=1)[:, None]

        dS_ij = P_ij * (dP_ij - D_i)

        dQ += tl.dot(dS_ij.to(tl.float16), tl.trans(k_trans), allow_tf32=False)

        v_block_ptr = tl.advance(v_block_ptr, offsets=(0, BLOCK_SQ))
        k_block_ptr = tl.advance(k_block_ptr, offsets=(0, BLOCK_SQ))

    v_trans = tl.load(v_block_ptr)
    k_trans = tl.load(k_block_ptr)

    s_ij = tl.dot(q, k_trans, allow_tf32=False)

    # causal masking on final block
    s_ij = tl.where(
        kv_chunk_pid * BLOCK_SQ + offs_k[:, None] >= (final + offs_k[None, :]),
        s_ij,
        float("-inf"),
    )

    P_ij = (1.0 / l_i) * tl.math.exp2(s_ij - m_i)

    dP_ij = tl.dot(dout, v_trans, allow_tf32=False)
    D_i = tl.sum(dout * out, axis=1)[:, None]

    dS_ij = P_ij * (dP_ij - D_i)

    dQ += tl.dot(dS_ij.to(tl.float16), tl.trans(k_trans), allow_tf32=False)

    tl.store(
        dQ_block_ptr,
        (ln2 * head_scale * dQ).to(tl.float16),
    )


def flash_wrapper_bwd(
    grad_output: torch.Tensor,
    out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function calling Flash Attention backward pass."""
    batch, nh, sq, hd = q.shape

    BLOCK_HD = triton.next_power_of_2(hd)
    BLOCK_SQ = 64 if BLOCK_HD < 128 else 32

    num_warps = 4 if BLOCK_HD <= 128 else 8

    assert hd in [32, 64, 128], "Only head_dims of [32,64,128] are supported."
    assert (
        sq % BLOCK_SQ == 0
    ), f"Number of elements in sequence must be a multiple of {BLOCK_SQ}"

    dQ = torch.zeros_like(q)
    dK = torch.empty_like(k)
    dV = torch.empty_like(v)

    def grid(META):
        return (triton.cdiv(sq, META["BLOCK_SQ"]), batch * nh)

    head_scale = (1.0) / (q.shape[-1] ** 0.5)

    flash_attn_bwd[grid](
        q,
        k,
        v,
        out,
        m,
        l,
        grad_output,
        dV,
        dK,
        dQ,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        m.stride(0),
        m.stride(1),
        head_scale=head_scale,
        BLOCK_HD=BLOCK_HD,
        BLOCK_SQ=BLOCK_SQ,
        context_sq=sq,
        num_warps=num_warps,
        num_stages=2,
        num_head=nh,
    )

    return dQ, dK, dV





@triton.jit
def flash_attn_fwd_kernel(
    q_ptr, k_ptr, v_ptr,            # q, k, v matrix pointer
    o_ptr,                          # output matrix pointer
    m_ptr, l_ptr,                   # store intermediate values (max, normalizer) for backward
    qkv_stride_batch, qkv_stride_head, qkv_stride_seq, qkv_stride_head_dim,
    ml_stride_batch, ml_stride_head,
    BLOCK_HEAD_DIM: tl.constexpr,   # block size for head dim
    BLOCK_SEQ: tl.constexpr,        # block size for seq
    head_scale,
    num_head,
    context_seq                     # seq length
):
    """
    flash attention with causal masking
    """
    batch_head_pid = tl.program_id(axis=0)      # parallelize through batch and heads
    sequence_chunk_pid = tl.program_id(axis=1)  # parallelize through sequence chunks

    offset_batch = batch_head_pid // num_head
    offset_head = batch_head_pid % num_head
    batch_head_offset = offset_batch.to(tl.int64) * qkv_stride_batch + offset_head.to(tl.int64) * qkv_stride_head

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + batch_head_offset,
        shape=(context_seq, BLOCK_HEAD_DIM),
        strides=(qkv_stride_seq, qkv_stride_head_dim),
        offsets=(sequence_chunk_pid * BLOCK_SEQ, 0),
        block_shape=(BLOCK_SEQ, BLOCK_HEAD_DIM),
        order=(1, 0)
    )

    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + batch_head_offset,
        shape=(BLOCK_HEAD_DIM, context_seq),
        strides=(qkv_stride_head_dim, qkv_stride_seq),
        offsets=(0, 0),
        block_shape=(BLOCK_HEAD_DIM, BLOCK_SEQ),
        order=(0, 1)
    )

    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + batch_head_offset,
        shape=(context_seq, BLOCK_HEAD_DIM),
        strides=(qkv_stride_seq, qkv_stride_head_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SEQ, BLOCK_HEAD_DIM),
        order=(0, 1)
    )

    out = tl.zeros([BLOCK_SEQ, BLOCK_HEAD_DIM], dtype=tl.float32)
    m_i = tl.full([BLOCK_SEQ], float("-inf"), dtype=tl.float32)   # max softmax num for current chunk
    l_i = tl.zeros([BLOCK_SEQ], dtype=tl.float32)                               # normalization factor

    q = tl.load(q_block_ptr, boundary_check=(0, 1))

    # scale by 1/ln(2),  e**(x) = 2 ** (x * log_2(e)), use exp2(x) -> 2**x to replace exp(x) to increase the speed
    ln2_inv: tl.constexpr = 1.44269504      # which is log_2(e)
    head_scale *= ln2_inv
    head_scale = head_scale.to(tl.float16)

    q *= head_scale
    # start_n = sequence_chunk_pid * BLOCK_SEQ
    # end_n = (sequence_chunk_pid + 1) * BLOCK_SEQ
    min_range = context_seq
    max_range = sequence_chunk_pid * BLOCK_SEQ + 1

    offsets_k = tl.arange(0, BLOCK_SEQ)
    offsets_q = tl.arange(0, BLOCK_SEQ)

    for chunk in range(0, max_range - 1, BLOCK_SEQ):
        k = tl.load(k_block_ptr)
        v = tl.load(v_block_ptr)

        s_ij = tl.dot(q, k, allow_tf32=False)       # [BLOCK_SEQ, BLOCK_SEQ]

        m_ij = tl.max(s_ij, axis=1)     # [BLOCK_SEQ, ]
        p_ij = tl.math.exp2(s_ij - m_ij[:, None])   #[BLOCK_SEQ, BLOCK_SEQ] -> numeritor of softmax
        l_ij = tl.sum(p_ij, axis=1)     # [BLOCK_SEQ, ] -> denominator of softmax

        m_i_new = tl.maximum(m_i, m_ij)
        running_correction = tl.math.exp2(m_i - m_i_new)    # 
        new_correction = tl.math.exp2(m_ij - m_i_new)

        l_i_new = (running_correction * l_i) + (new_correction * l_ij)

        out = (l_i * running_correction)[:, None] * tl.dot(p_ij.to(tl.float16), v, allow_tf32=False)

        out /= (l_i_new)[:, None]

        m_i = m_i_new
        l_i = l_i_new

        k_block_ptr = tl.advance(k_block_ptr, offsets=(0, BLOCK_SEQ))
        v_block_ptr = tl.advance(v_block_ptr, offsets=(BLOCK_SEQ, 0))
    
    # final block - we reuse code here to remove conditionals from for loop
    k = tl.load(k_block_ptr)
    v = tl.load(v_block_ptr)
    s_ij = tl.dot(q, k, allow_tf32=False)
    offs = max_range - 1
    s_ij = tl.where(sequence_chunk_pid * BLOCK_SEQ + offsets_k[:, None] >= (offs + offsets_q),
                    s_ij,
                    float("-inf"))
    m_ij = tl.max(s_ij, axis=1)  # [BLOCK_SEQ, ]
    p_ij = tl.math.exp2(s_ij - m_ij[:, None])  # [BLOCK_SEQ, BLOCK_SEQ]
    l_ij = tl.sum(p_ij, axis=1)  # [BLOCK_SEQ, ]
    m_i_new = tl.maximum(m_i, m_ij)
    running_correction = tl.math.exp2(m_i - m_i_new)
    new_correction = tl.math.exp2(m_ij - m_i_new)
    l_i_new = (running_correction * l_i) + (new_correction * l_ij)
    out = (l_i * running_correction)[:, None] * out
    out += new_correction[:, None] * tl.dot(p_ij.to(tl.float16), v, allow_tf32=False)
    out /= (l_i_new)[:, None]
    m_i = m_i_new
    l_i = l_i_new

    out_block_ptr = tl.make_block_ptr(
        o_ptr + batch_head_offset,
        shape=(context_seq, BLOCK_HEAD_DIM),
        block_shape=(BLOCK_SEQ, BLOCK_HEAD_DIM),
        strides=(qkv_stride_seq, qkv_stride_head),
        order=(1, 0),
        offsets=(sequence_chunk_pid * BLOCK_SEQ, 0),
    )

    tl.store(
        out_block_ptr,
        value=out.to(tl.float16),
    )

    batch_head_offset = offset_batch.to(tl.int64) * ml_stride_batch + offset_head.to(tl.int64) * ml_stride_head

    # store m and l which are used in backward pass to recreate softmax activation
    m_ptr_start = m_ptr + (batch_head_offset) + (sequence_chunk_pid * BLOCK_SEQ)
    l_ptr_start = l_ptr + (batch_head_offset) + (sequence_chunk_pid * BLOCK_SEQ)

    tl.store(m_ptr_start + offsets_q, m_i)
    tl.store(l_ptr_start + offsets_q, l_i)


def flash_attn_fwd_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    batch, num_head, seq, head_dim = q.shape

    BLOCK_HEAD_DIM = triton.next_power_of_2(head_dim)
    BLOCK_SEQ = 64 if BLOCK_HEAD_DIM < 128 else 32
    num_warps = 4 if BLOCK_HEAD_DIM <= 128 else 8

    assert(seq % BLOCK_SEQ == 0), f"Number of elements in sequence must be a multiple of {BLOCK_SEQ}"

    out = torch.empty_like(q)

    grid = lambda META: (batch * num_head, triton.cdiv(seq, META["BLOCK_SEQ"]))

    m = torch.empty((batch, num_head, seq), device=q.device, dtype=torch.float16)
    l = torch.empty_like(m)

    head_scale = 1.0 / (q.shape[-1] ** 0.5)

    flash_attn_fwd_kernel[grid](
        q, k, v, out,
        m, l,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        m.stride(0), m.stride(1),
        BLOCK_HEAD_DIM=BLOCK_HEAD_DIM,
        BLOCK_SEQ=BLOCK_SEQ,
        num_warps=num_warps,
        num_stages=2,
        head_scale=head_scale,
        context_seq=seq,
        num_head=num_head
    )

    return out, m, l


@triton.jit
def flash_attn_bwd_kernel(
    q_ptr, k_ptr, v_ptr,
    o_ptr, m_ptr, l_ptr,
    dO_ptr, dV_ptr, dK_ptr, dQ_ptr,
    qkv_stride_batch, qkv_stride_head, qkv_stride_seq, qkv_stride_head_dim,
    ml_stride_batch, ml_stride_head,
    head_scale,
    BLOCK_HEAD_DIM: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    context_seq,
    num_head,
):
    """Flash Attention backward pass with causal masking"""
    batch_head_pid = tl.program_id(axis=0)  # parallelize across batch x heads
    kv_chunk_pid = tl.program_id(axis=1)    # parallelize across kv chunks

    offset_batch = batch_head_pid // num_head
    offset_head = batch_head_pid % num_head

    offset_batch_head = offset_batch.to(tl.int64) * qkv_stride_batch + offset_head.to(tl.int64) * qkv_stride_head

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + offset_batch_head,
        shape=(context_seq, BLOCK_HEAD_DIM),
        strides=(qkv_stride_seq, qkv_stride_head_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SEQ, BLOCK_HEAD_DIM),
        order=(1, 0),
    )

    dout_block_ptr = tl.make_block_ptr(
        base=dO_ptr + offset_batch_head,
        shape=(context_seq, BLOCK_HEAD_DIM),
        strides=(qkv_stride_seq, qkv_stride_head_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SEQ, BLOCK_HEAD_DIM),
        order=(1, 0),
    )

    out_block_ptr = tl.make_block_ptr(
        base=o_ptr + offset_batch_head,
        shape=(context_seq, BLOCK_HEAD_DIM),
        strides=(qkv_stride_seq, qkv_stride_head_dim),
        offsets=(0, 0),
        block_shape=(BLOCK_SEQ, BLOCK_HEAD_DIM),
        order=(1, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + offset_batch_head,
        shape=(BLOCK_HEAD_DIM, context_seq),
        strides=(qkv_stride_seq, qkv_stride_head_dim),
        offsets=(0, kv_chunk_pid * BLOCK_SEQ),
        block_shape=(BLOCK_SEQ, BLOCK_HEAD_DIM),
        order=(0, 1), 
    )

    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + offset_batch_head,
        shape=(BLOCK_HEAD_DIM, context_seq),
        strides=(qkv_stride_seq, qkv_stride_head_dim),
        offsets=(0, kv_chunk_pid * BLOCK_SEQ),
        block_shape=(BLOCK_SEQ, BLOCK_HEAD_DIM),
        order=(0, 1), 
    ) 

    dV = tl.zeros([BLOCK_SEQ, BLOCK_HEAD_DIM], dtype=tl.float32)
    dK = tl.zeros([BLOCK_SEQ, BLOCK_HEAD_DIM], dtype=tl.float32)

    k_trans = tl.load(k_block_ptr, boundary_check=(1,))
    v = tl.load(v_block_ptr, boundary_check=(1,))

    ln2_inv: tl.constexpr = 1.44269504
    ln2: tl.constexpr = 0.6931471824645996

    head_scale *= ln2_inv
    head_scale = head_scale.to(tl.float16)
    max_range = context_seq
    min_range = kv_chunk_pid * BLOCK_SEQ

    offsets_k = tl.arange(0, BLOCK_SEQ)
    offsets_q = kv_chunk_pid * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)

    ml_offset_batch_head = offset_batch.to(tl.int64) * ml_stride_batch + offset_head.to(tl.int64) * ml_stride_head

    m_ptr_start = m_ptr + ml_offset_batch_head
    l_ptr_start = l_ptr + ml_offset_batch_head

    # loop is split into pre/post masking to remove conditional use
    for q_chunk in range(0, min_range + 1, BLOCK_SEQ):
        q = tl.load(q_block_ptr)
        dout = tl.load(dout_block_ptr)
        out = tl.load(out_block_ptr)
        q *= head_scale

        m_i = tl.load(m_ptr_start + q_chunk + offsets_k, mask=offsets_k < context_seq)[:, None]
        l_i = tl.load(l_ptr_start + q_chunk + offsets_k, mask=offsets_k < context_seq)[:, None]
        s_ij = tl.dot(q, k_trans, allow_tf32=False)

        s_ij = tl.where(
            (q_chunk + offsets_k[:, None]) >= (offsets_q[None, :]),
            s_ij,
            float("-inf")
        )

        P_ij = (1.0 / l_i) * tl.math.exp2(s_ij - m_i)

        dV += tl.dot(tl.trans(P_ij.to(tl.float16)), dout, allow_tf32=False)

        dP_ij = tl.dot(dout, v, allow_tf32=False)
        D_i = tl.sum(dout * out, axis=1)[:, None]
        dS_ij = P_ij * (dP_ij - D_i)

        dK += tl.dot(tl.trans(dS_ij.to(tl.float16)), q, allow_tf32=False)

        q_block_ptr = tl.advance(q_block_ptr, offsets=(BLOCK_SEQ, 0))
        out_block_ptr = tl.advance(out_block_ptr, offsets=(BLOCK_SEQ, 0))
        dout_block_ptr = tl.advance(dout_block_ptr, offsets=(BLOCK_SEQ, 0))

    min_range_offset = min_range + BLOCK_SEQ

    for q_chunk in range(min_range_offset, max_range, BLOCK_SEQ):
        q = tl.load(
            q_block_ptr,
        )
        dout = tl.load(
            dout_block_ptr,
        )
        out = tl.load(
            out_block_ptr,
        )
        q *= head_scale

        m_i = tl.load(m_ptr_start + q_chunk + offsets_k, mask=offsets_k < context_seq)[:, None]
        l_i = tl.load(l_ptr_start + q_chunk + offsets_k, mask=offsets_k < context_seq)[:, None]
        s_ij = tl.dot(q, k_trans, allow_tf32=False)

        P_ij = (1.0 / l_i) * tl.math.exp2(s_ij - m_i)

        dV += tl.dot(tl.trans(P_ij.to(tl.float16)), dout, allow_tf32=False)

        dP_ij = tl.dot(dout, v, allow_tf32=False)
        D_i = tl.sum(dout * out, axis=1)[:, None]
        dS_ij = P_ij * (dP_ij - D_i)

        dK += tl.dot(tl.trans(dS_ij.to(tl.float16)), q, allow_tf32=False)

        q_block_ptr = tl.advance(q_block_ptr, offsets=(BLOCK_SEQ, 0))
        out_block_ptr = tl.advance(out_block_ptr, offsets=(BLOCK_SEQ, 0))
        dout_block_ptr = tl.advance(dout_block_ptr, offsets=(BLOCK_SEQ, 0))
    
    dV_block_ptr = tl.make_block_ptr(
        base=dV_ptr + offset_batch_head,
        shape=(context_seq, BLOCK_HEAD_DIM),
        strides=(qkv_stride_seq, qkv_stride_head),
        offsets=(kv_chunk_pid * BLOCK_SEQ, 0),
        block_shape=(BLOCK_SEQ, BLOCK_HEAD_DIM),
        order=(1, 0)
    )

    dK_block_ptr = tl.make_block_ptr(
        base=dK_ptr + offset_batch_head,
        shape=(context_seq, BLOCK_HEAD_DIM),
        strides=(qkv_stride_seq, qkv_stride_head),
        offsets=(kv_chunk_pid * BLOCK_SEQ, 0),
        block_shape=(BLOCK_SEQ, BLOCK_HEAD_DIM),
        order=(1, 0)
    )

    tl.store(dV_block_ptr, dV.to(tl.float16))
    tl.store(dK_block_ptr, (ln2 * dK).to(tl.float16))

    """
    compute dQ
    """

    # reset block pointers
    dQ_block_ptr = tl.make_block_ptr(
        base=dQ_ptr + offset_batch_head,
        shape=(context_seq, BLOCK_HEAD_DIM),
        strides=(qkv_stride_seq, qkv_stride_head_dim),
        offsets=(kv_chunk_pid * BLOCK_SEQ, 0),
        block_shape=(BLOCK_SEQ, BLOCK_HEAD_DIM),
        order=(1, 0)
    )

    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + offset_batch_head,
        shape=(context_seq, BLOCK_HEAD_DIM),
        strides=(qkv_stride_seq, qkv_stride_head_dim),
        offsets=(kv_chunk_pid * BLOCK_SEQ, 0),
        block_shape=(BLOCK_SEQ, BLOCK_HEAD_DIM),
        order=(1, 0),
    )

    dout_block_ptr = tl.make_block_ptr(
        base=dO_ptr + offset_batch_head,
        shape=(context_seq, BLOCK_HEAD_DIM),
        strides=(qkv_stride_seq, qkv_stride_head_dim),
        offsets=(kv_chunk_pid * BLOCK_SEQ, 0),
        block_shape=(BLOCK_SEQ, BLOCK_HEAD_DIM),
        order=(1, 0),
    )

    out_block_ptr = tl.make_block_ptr(
        base=o_ptr + offset_batch_head,
        shape=(context_seq, BLOCK_HEAD_DIM),
        strides=(qkv_stride_seq, qkv_stride_head_dim),
        offsets=(kv_chunk_pid * BLOCK_SEQ, 0),
        block_shape=(BLOCK_SEQ, BLOCK_HEAD_DIM),
        order=(1, 0),
    )

    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + offset_batch_head,
        shape=(BLOCK_HEAD_DIM, context_seq),
        strides=(qkv_stride_head_dim, qkv_stride_seq),
        offsets=(0, 0),
        block_shape=(BLOCK_SEQ, BLOCK_HEAD_DIM),
        order=(0, 1), 
    )

    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + offset_batch_head,
        shape=(BLOCK_HEAD_DIM, context_seq),
        strides=(qkv_stride_head_dim, qkv_stride_seq),
        offsets=(0, 0),
        block_shape=(BLOCK_SEQ, BLOCK_HEAD_DIM),
        order=(0, 1), 
    )

    q = tl.load(q_block_ptr)
    q *= head_scale

    dQ = tl.zeros([BLOCK_SEQ, BLOCK_HEAD_DIM], dtype=tl.float32)
    offsets_k = tl.arange(0, BLOCK_SEQ) 
    max_range = kv_chunk_pid * BLOCK_SEQ + 1
    final = max_range - 1

    m_ptr_start = m_ptr + (ml_offset_batch_head) + (kv_chunk_pid * BLOCK_SEQ)
    l_ptr_start = l_ptr + (ml_offset_batch_head) + (kv_chunk_pid * BLOCK_SEQ)

    m_i = tl.load(m_ptr_start + offsets_k, mask=offsets_k < context_seq)[:, None]
    l_i = tl.load(l_ptr_start + offsets_k, mask=offsets_k < context_seq)[:, None]

    dout = tl.load(
        dout_block_ptr,
    )
    out = tl.load(
        out_block_ptr,
    )


    for q_chunk in range(0, final, BLOCK_SEQ):
        v_trans = tl.load(
            v_block_ptr,
        )
        k_trans = tl.load(
            k_block_ptr,
        )

        s_ij = tl.dot(q, k_trans, allow_tf32=False)

        P_ij = (1.0 / l_i) * tl.math.exp2(s_ij - m_i)

        dP_ij = tl.dot(dout, v_trans, allow_tf32=False)
        D_i = tl.sum(dout * out, axis=1)[:, None]

        dS_ij = P_ij * (dP_ij - D_i)

        dQ += tl.dot(dS_ij.to(tl.float16), tl.trans(k_trans), allow_tf32=False)

        v_block_ptr = tl.advance(v_block_ptr, offsets=(0, BLOCK_SEQ))
        k_block_ptr = tl.advance(k_block_ptr, offsets=(0, BLOCK_SEQ))

    v_trans = tl.load(v_block_ptr)
    k_trans = tl.load(k_block_ptr)

    s_ij = tl.dot(q, k_trans, allow_tf32=False)

    # causal masking on the final block
    s_ij = tl.where(
        kv_chunk_pid * BLOCK_SEQ + offsets_k[:, None] >= (final + offsets_k[None, :]),
        s_ij,
        float("-inf"),
    )

    P_ij = (1.0 / l_i) * tl.math.exp2(s_ij - m_i)

    dP_ij = tl.dot(dout, v_trans, allow_tf32=False)
    D_i = tl.sum(dout * out, axis=1)[:, None]

    dS_ij = P_ij * (dP_ij - D_i)

    dQ += tl.dot(dS_ij.to(tl.float16), tl.trans(k_trans), allow_tf32=False)

    tl.store(
        dQ_block_ptr,
        (ln2 * head_scale * dQ).to(tl.float16),
    )


def flash_attn_bwd_func(
    grad_output: torch.Tensor,
    out: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Function calling Flash Attention backward pass."""
    batch, nh, sq, hd = q.shape

    BLOCK_HD = triton.next_power_of_2(hd)
    BLOCK_SQ = 64 if BLOCK_HD < 128 else 32

    num_warps = 4 if BLOCK_HD <= 128 else 8

    assert hd in [32, 64, 128], "Only head_dims of [32,64,128] are supported."
    assert (
        sq % BLOCK_SQ == 0
    ), f"Number of elements in sequence must be a multiple of {BLOCK_SQ}"

    dQ = torch.zeros_like(q)
    dK = torch.empty_like(k)
    dV = torch.empty_like(v)

    def grid(META):
        return (batch * nh, triton.cdiv(sq, META["BLOCK_SQ"]))

    head_scale = (1.0) / (q.shape[-1] ** 0.5)

    flash_attn_bwd_kernel[grid](
        q,
        k,
        v,
        out,
        m,
        l,
        grad_output,
        dV,
        dK,
        dQ,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        m.stride(0),
        m.stride(1),
        head_scale=head_scale,
        BLOCK_HD=BLOCK_HD,
        BLOCK_SQ=BLOCK_SQ,
        context_sq=sq,
        num_warps=num_warps,
        num_stages=2,
        num_head=nh,
    )

    return dQ, dK, dV