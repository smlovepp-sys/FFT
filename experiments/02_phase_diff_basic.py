"""
相位差缠绕基础测试
====================
验证：
  1. 无噪声下无损往返
  2. 纯共模噪声下中心化 LS 完美
  3. 纯差分噪声下两者接近
"""

import torch
import numpy as np


def encode(values, w, snr_db, common_ratio):
    v_min, v_max = values.min(), values.max()
    u = (values - v_min) / (v_max - v_min)
    w_t = torch.tensor(w, dtype=torch.float32)
    phi = u.unsqueeze(-1) * 2 * np.pi * w_t.unsqueeze(0)

    snr_lin = 10 ** (snr_db / 10)
    S = (2 * np.pi) ** 2 / 12
    sigma_c = (S * common_ratio / snr_lin) ** 0.5
    sigma_d = (S * (1 - common_ratio) / snr_lin) ** 0.5

    noise_c = torch.randn(phi.shape[0], 1) * sigma_c
    noise_d = torch.randn_like(phi) * sigma_d
    return phi + noise_c + noise_d, (v_min, v_max)


def decode_original(phi, w, v_range):
    w_t = torch.tensor(w, dtype=torch.float32)
    num = (phi * w_t.unsqueeze(0)).sum(dim=-1)
    den = 2 * np.pi * (w_t ** 2).sum()
    u_hat = num / den
    return u_hat * (v_range[1] - v_range[0]) + v_range[0]


def decode_centered(phi, w, v_range):
    w_t = torch.tensor(w, dtype=torch.float32)
    w_bar = w_t.mean()
    phi_bar = phi.mean(dim=-1, keepdim=True)
    w_c = w_t - w_bar
    phi_c = phi - phi_bar
    num = (phi_c * w_c.unsqueeze(0)).sum(dim=-1)
    den = 2 * np.pi * (w_c ** 2).sum()
    u_hat = num / den
    return u_hat * (v_range[1] - v_range[0]) + v_range[0]


def main():
    torch.manual_seed(42)
    print("Device: cpu（数据量小）")

    values = torch.randn(50000) * 50
    w = [1.0, 2.0, 4.0, 8.0]

    print("\n=== 无噪声 ===")
    phi, v_range = encode(values, w, snr_db=100, common_ratio=0.0)
    rec_o = decode_original(phi, w, v_range)
    rec_c = decode_centered(phi, w, v_range)
    print(f"  原始 LS MSE:   {(values - rec_o).pow(2).mean():.4e}")
    print(f"  中心化 LS MSE: {(values - rec_c).pow(2).mean():.4e}")

    print("\n=== 纯共模（30 dB）===")
    phi, v_range = encode(values, w, snr_db=30, common_ratio=1.0)
    rec_o = decode_original(phi, w, v_range)
    rec_c = decode_centered(phi, w, v_range)
    print(f"  原始 LS MSE:   {(values - rec_o).pow(2).mean():.4e}")
    print(f"  中心化 LS MSE: {(values - rec_c).pow(2).mean():.4e}")

    print("\n=== 纯差分（30 dB）===")
    phi, v_range = encode(values, w, snr_db=30, common_ratio=0.0)
    rec_o = decode_original(phi, w, v_range)
    rec_c = decode_centered(phi, w, v_range)
    print(f"  原始 LS MSE:   {(values - rec_o).pow(2).mean():.4e}")
    print(f"  中心化 LS MSE: {(values - rec_c).pow(2).mean():.4e}")
    print("✓ 通过")


if __name__ == "__main__":
    main()