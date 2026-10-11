"""
Phase-Difference Entanglement Codec
====================================
用相位差编码承载标量信息，抗共模噪声。

编码：
    φ_i = 2π · u · w_i,  i = 1..M
    其中 w = [2^0, 2^1, ..., 2^(M-1)]（指数权重）

解码（中心化最小二乘）：
    û = Σ_i (w_i - w̄)(φ_i - φ̄) / (2π Σ_i (w_i - w̄)²)

等价于差分最小二乘：
    û = Σ_{i<j} (w_i - w_j)(φ_i - φ_j) / (2π Σ_{i<j} (w_i - w_j)²)

核心性质：
    共模噪声（所有模式共享的偏移）被完全消除。
    差分噪声会影响解码，但可以通过增加模式数来平均。

作者: [苏]
License: MIT
"""

import numpy as np
import torch


__all__ = ["PhaseDiffCodec"]


class PhaseDiffCodec:
    """
    相位差缠绕编解码器。

    参数:
        n_modes:        模式数 M
        snr_db:         信噪比（dB）
        common_ratio:   共模噪声占比，0..1
        device:         设备
    """
    def __init__(
        self,
        n_modes: int = 8,
        snr_db: float = 30.0,
        common_ratio: float = 0.8,
        device: str = "cuda",
    ):
        self.n_modes = n_modes
        self.snr_db = snr_db
        self.common_ratio = common_ratio
        self.device = device

        # 指数权重 w_i = 2^(i-1)
        self.w = torch.tensor(
            [2 ** i for i in range(n_modes)],
            dtype=torch.float32, device=device)
        self.w_bar = self.w.mean()
        self.w_centered = self.w - self.w_bar
        self.denom = (self.w_centered ** 2).sum().item()

    @torch.no_grad()
    def encode(self, u: torch.Tensor) -> torch.Tensor:
        """
        编码：标量 u → 相位 φ。

        u: (N,) 或 (N, 1)，范围 [0, 1]
        返回: (N, M)
        """
        u = u.to(self.device)
        if u.dim() == 2:
            u = u.squeeze(-1)
        phi = u.unsqueeze(-1) * 2 * np.pi * self.w.unsqueeze(0)
        return phi

    @torch.no_grad()
    def add_noise(self, phi: torch.Tensor) -> torch.Tensor:
        """
        加噪声：共模 + 差分。
        """
        snr_lin = 10 ** (self.snr_db / 10)
        S = (2 * np.pi) ** 2 / 12

        sigma_c = (S * self.common_ratio / snr_lin) ** 0.5
        sigma_d = (S * (1 - self.common_ratio) / snr_lin) ** 0.5

        noise_c = torch.randn(
            phi.shape[0], 1, device=phi.device) * sigma_c
        noise_d = torch.randn_like(phi) * sigma_d

        return phi + noise_c + noise_d

    @torch.no_grad()
    def decode(self, phi_noisy: torch.Tensor) -> torch.Tensor:
        """
        解码：中心化最小二乘。

        phi_noisy: (N, M)
        返回: (N,) 估计的 u
        """
        phi_bar = phi_noisy.mean(dim=-1, keepdim=True)
        phi_centered = phi_noisy - phi_bar

        numerator = (phi_centered * self.w_centered.unsqueeze(0)).sum(dim=-1)
        u_hat = numerator / (2 * np.pi * self.denom)
        return u_hat

    @torch.no_grad()
    def roundtrip(self, tensor: torch.Tensor,
                  add_noise: bool = True) -> torch.Tensor:
        """
        完整往返：张量 → 归一化 → 编码 → (加噪) → 解码 → 反归一化。
        """
        orig_shape = tensor.shape
        orig_dtype = tensor.dtype

        x = tensor.float().flatten()
        v_min, v_max = x.min(), x.max()
        if (v_max - v_min).item() < 1e-9:
            return tensor.clone()

        u = (x - v_min) / (v_max - v_min)

        phi = self.encode(u)
        if add_noise:
            phi = self.add_noise(phi)
        u_hat = self.decode(phi)

        x_recon = u_hat * (v_max - v_min) + v_min
        return x_recon.view(orig_shape).to(orig_dtype)


# ============================================================
# 测试
# ============================================================
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 模拟一批数值
    torch.manual_seed(42)
    values = torch.randn(50000, device=device) * 50

    codec = PhaseDiffCodec(n_modes=8, snr_db=30.0,
                           common_ratio=0.8, device=device)

    recon = codec.roundtrip(values)
    mse = (values - recon).pow(2).mean().item()
    max_err = (values - recon).abs().max().item()

    print(f"8 模式, 30 dB, 共模 0.8")
    print(f"  MSE: {mse:.4e}")
    print(f"  Max err: {max_err:.4e}")

    # 无噪声
    recon_clean = codec.roundtrip(values, add_noise=False)
    mse_clean = (values - recon_clean).pow(2).mean().item()
    print(f"无噪声 MSE: {mse_clean:.4e}")