"""
权重分布对比
==============
测不同 w 分布下的 Fisher 信息和实际 MSE。
"""

import torch
import numpy as np


def make_distributions(M=8):
    return {
        'linear': torch.arange(1, M + 1, dtype=torch.float32),
        'uniform': torch.ones(M),
        'quadratic': torch.arange(1, M + 1, dtype=torch.float32) ** 2,
        'exponential': 2.0 ** torch.arange(0, M, dtype=torch.float32),
        'log': torch.log(torch.arange(1, M + 1, dtype=torch.float32) + 1),
        'sqrt': torch.sqrt(torch.arange(1, M + 1, dtype=torch.float32)),
    }


def encode_decode(values, w, snr_db, common_ratio):
    v_min, v_max = values.min(), values.max()
    u = (values - v_min) / (v_max - v_min)
    w_t = torch.tensor(w, dtype=torch.float32)
    phi = u.unsqueeze(-1) * 2 * np.pi * w_t.unsqueeze(0)
    snr_lin = 10 ** (snr_db / 10)
    S = (2 * np.pi) ** 2 / 12
    sigma_c = (S * common_ratio / snr_lin) ** 0.5
    sigma_d = (S * (1 - common_ratio) / snr_lin) ** 0.5
    phi = phi + torch.randn(phi.shape[0], 1) * sigma_c + torch.randn_like(phi) * sigma_d

    w_c = w_t - w_t.mean()
    phi_c = phi - phi.mean(dim=-1, keepdim=True)
    num = (phi_c * w_c.unsqueeze(0)).sum(dim=-1)
    den = 2 * np.pi * (w_c ** 2).sum()
    u_hat = num / den
    rec = u_hat * (v_max - v_min) + v_min
    return (values - rec).pow(2).mean().item()


def main():
    torch.manual_seed(42)
    values = torch.randn(50000) * 50
    M = 8

    print(f"{'分布':<15} | {'F_orig':>10} | {'F_cent':>10} | {'比值':>8} | "
          f"{'30dB+80%共模 MSE':>18}")
    print("-" * 75)

    for name, w in make_distributions(M).items():
        F_orig = (w ** 2).sum().item()
        F_cent = ((w - w.mean()) ** 2).sum().item()
        ratio = F_orig / max(F_cent, 1e-10)

        # 归一化到 Σw²=M
        w_norm = w / (w ** 2).sum().sqrt() * np.sqrt(M)

        mses = []
        for seed in range(5):
            torch.manual_seed(seed)
            mses.append(encode_decode(values, w_norm, 30, 0.8))
        mse = np.mean(mses)
        print(f"{name:<15} | {F_orig:>10.2f} | {F_cent:>10.2f} | "
              f"{ratio:>8.2f} | {mse:>18.4e}")


if __name__ == "__main__":
    main()