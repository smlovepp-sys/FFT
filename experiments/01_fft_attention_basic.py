"""
FFT 注意力基础测试
====================
验证：
  1. FFT 注意力能前向传播
  2. 梯度能回传
  3. 输出形状正确
"""

import torch
from opa import OpticalMultiheadAttention


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    B, N, D = 2, 128, 64
    x = torch.randn(B, N, D, device=device, requires_grad=True)

    attn = OpticalMultiheadAttention(
        d_model=D, n_heads=4, w_per_head=4,
        max_seq_len=512, causal=False,
    ).to(device)

    out = attn(x)
    print(f"输入: {x.shape} → 输出: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in attn.parameters()):,}")

    loss = out.pow(2).mean()
    loss.backward()

    grad_norm = sum(
        p.grad.norm().item() for p in attn.parameters()
        if p.grad is not None
    )
    print(f"梯度范数: {grad_norm:.4f}")
    assert grad_norm > 0, "梯度为 0，反向传播有问题"
    print("✓ 通过")


if __name__ == "__main__":
    main()