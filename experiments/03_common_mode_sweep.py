"""
共模比例扫描
==============
找中心化 LS 反超原始 LS 的交叉点。
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
    return num / den * (v_range[1] - v_range[0]) + v_range[0]


def decode_centered(phi, w, v_range):
    w_t = torch.tensor(w, dtype=torch.float32)
    w_c = w_t - w_t.mean()
    phi_c = phi - phi.mean(dim=-1, keepdim=True)
    num = (phi_c * w_c.unsqueeze(0)).sum(dim=-1)
    den = 2 * np.pi * (w_c ** 2).sum()
    return num / den * (v_range[1] - v_range[0]) + v_range[0]


def main():
    torch.manual_seed(42)
    values = torch.randn(50000) * 50
    w = [1.0, 2.0, 3.0, 4.0]

    print(f"{'共模比例':>10} | {'原始 LS':>14} | {'中心化 LS':>14} | 赢家")
    print("-" * 60)

    for cr in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
        mse_o_list, mse_c_list = [], []
        for seed in range(5):
            torch.manual_seed(seed)
            phi, vr = encode(values, w, 30, cr)
            mse_o_list.append(
                (values - decode_original(phi, w, vr)).pow(2).mean().item())
            mse_c_list.append(
                (values - decode_centered(phi, w, vr)).pow(2).mean().item())
        mo, mc = np.mean(mse_o_list), np.mean(mse_c_list)
        winner = "原始" if mo < mc else "中心化"
        print(f"{cr:>10.1f} | {mo:>14.4e} | {mc:>14.4e} | {winner}")


if __name__ == "__main__":
    main()