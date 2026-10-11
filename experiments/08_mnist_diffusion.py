"""
MNIST 扩散：标准注意力 vs 光团注意力
=======================================
- 数据集：MNIST（28×28 灰度）
- 模型：TinyUNet，d_model=64，2 层注意力
- 步数：20000
- 对比：loss、显存、参数、生成图

预计：标准 ~8 分钟，光团 ~10 分钟
显存：都 < 1 GB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math, time, os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image


def P(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ============================================================
# 标准注意力
# ============================================================
class StandardAttention(nn.Module):
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = self.d_head ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out(out)


# ============================================================
# 光团注意力
# ============================================================
class LightGroupAttention(nn.Module):
    def __init__(self, d_model, n_modes=8, max_seq_len=1024):
        super().__init__()
        self.n_modes = n_modes
        self.encoder = nn.Linear(d_model, n_modes * 2)
        self.prop_phase = nn.Parameter(
            torch.randn(n_modes, max_seq_len) * 0.1)
        self.decoder = nn.Linear(n_modes * 2, d_model)

    def forward(self, x):
        B, N, D = x.shape
        M = self.n_modes

        optical = self.encoder(x)
        E_real = optical[..., :M]
        E_imag = optical[..., M:]
        E = torch.complex(E_real, E_imag)

        E_freq = torch.fft.fft(E.transpose(1, 2), dim=-1)
        phase = self.prop_phase[:, :N].unsqueeze(0)
        E_freq = E_freq * torch.exp(1j * phase)
        E = torch.fft.ifft(E_freq, dim=-1).transpose(1, 2)

        intensity = E.abs() ** 2
        intensity = F.softmax(intensity, dim=-1)

        weighted = E * intensity
        optical_out = torch.cat(
            [weighted.real, weighted.imag], dim=-1)
        return self.decoder(optical_out)


# ============================================================
# 注意力块
# ============================================================
class AttnBlock(nn.Module):
    def __init__(self, d_model, attn_type='standard', n_modes=8,
                 max_seq_len=1024):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        if attn_type == 'standard':
            self.attn = StandardAttention(d_model)
        else:
            self.attn = LightGroupAttention(d_model, n_modes, max_seq_len)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ============================================================
# TinyUNet（MNIST：1 通道，28×28）
# ============================================================
class TinyUNet(nn.Module):
    def __init__(self, in_ch=1, img_size=28, d_model=64,
                 attn_type='standard', n_modes=8):
        super().__init__()
        self.d_model = d_model
        self.img_size = img_size

        self.in_conv = nn.Conv2d(in_ch, d_model, 3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

        # 28 → 14 → 7
        self.enc28 = nn.Conv2d(d_model, d_model, 3, padding=1)
        self.down1 = nn.Conv2d(d_model, d_model, 4, stride=2, padding=1)
        self.enc14 = AttnBlock(d_model, attn_type, n_modes, 14 * 14)
        self.down2 = nn.Conv2d(d_model, d_model, 4, stride=2, padding=1)
        self.enc7 = AttnBlock(d_model, attn_type, n_modes, 7 * 7)

        self.mid = AttnBlock(d_model, attn_type, n_modes, 7 * 7)

        self.up1 = nn.ConvTranspose2d(d_model, d_model, 4, stride=2, padding=1)
        self.dec14 = AttnBlock(d_model, attn_type, n_modes, 14 * 14)
        self.up2 = nn.ConvTranspose2d(d_model, d_model, 4, stride=2, padding=1)
        self.dec28 = nn.Conv2d(d_model, d_model, 3, padding=1)

        self.norm_out = nn.GroupNorm(8, d_model)
        self.out_conv = nn.Conv2d(d_model, in_ch, 3, padding=1)

    def time_embed(self, t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) *
                          torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def _block_forward(self, block, h, cond):
        B, C, H, W = h.shape
        x_seq = h.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_seq = x_seq + cond.unsqueeze(1)
        x_seq = block(x_seq)
        return x_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)

    def forward(self, x, t):
        t_emb = self.time_mlp(self.time_embed(t, self.d_model))

        h = self.in_conv(x)
        h = h + t_emb[:, :, None, None]

        h = self.enc28(h)
        h = self.down1(h)
        h = self._block_forward(self.enc14, h, t_emb)
        skip14 = h
        h = self.down2(h)
        h = self._block_forward(self.enc7, h, t_emb)
        h = self._block_forward(self.mid, h, t_emb)

        h = self.up1(h)
        h = h + skip14
        h = self._block_forward(self.dec14, h, t_emb)
        h = self.up2(h)
        h = self.dec28(h)

        h = self.norm_out(h)
        return self.out_conv(h)


# ============================================================
# Diffusion
# ============================================================
class Diffusion:
    def __init__(self, T=1000, device='cuda'):
        self.T = T
        self.device = device
        betas = torch.linspace(1e-4, 0.02, T, device=device)
        alphas = 1 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod).sqrt()

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_a * x0 + sqrt_1ma * noise, noise

    @torch.no_grad()
    def sample(self, model, n_samples=16, n_steps=50, img_size=28):
        model.eval()
        x = torch.randn(n_samples, 1, img_size, img_size, device=self.device)
        step_size = self.T // n_steps
        timesteps = list(range(0, self.T, step_size))[::-1]
        for i, t in enumerate(timesteps):
            t_batch = torch.full((n_samples,), t, dtype=torch.long,
                                 device=self.device)
            t_norm = t_batch.float() / self.T
            pred_noise = model(x, t_norm)
            alpha = self.alphas_cumprod[t]
            if i + 1 < len(timesteps):
                alpha_prev = self.alphas_cumprod[timesteps[i + 1]]
            else:
                alpha_prev = torch.tensor(1.0, device=self.device)
            x0_pred = (x - (1 - alpha).sqrt() * pred_noise) / alpha.sqrt()
            x0_pred = x0_pred.clamp(-1, 1)
            x = alpha_prev.sqrt() * x0_pred + (1 - alpha_prev).sqrt() * pred_noise
        model.train()
        return x


# ============================================================
# 训练 + 生成
# ============================================================
def train_and_generate(name, attn_type, n_modes, steps, batch,
                       data_root, device, seed):
    torch.manual_seed(seed)

    P(f"\n{'=' * 70}")
    P(f"配置: {name} | attn={attn_type} | n_modes={n_modes} | seed={seed}")
    P(f"{'=' * 70}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_ds = datasets.MNIST(
        root=data_root, train=True, download=False, transform=transform)
    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True,
        num_workers=0, drop_last=True)

    model = TinyUNet(in_ch=1, img_size=28, d_model=64,
                     attn_type=attn_type, n_modes=n_modes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    P(f"参数量: {n_params:,}")

    diffusion = Diffusion(T=1000, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    warmup = 2000
    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    P(f"{'step':>7} | {'loss':>10} | {'peak_mem':>10} | {'time':>8}")
    P("-" * 50)

    model.train()
    t0 = time.time()
    losses = []
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    step = 0
    while step < steps:
        for x0, _ in train_loader:
            if step >= steps:
                break
            x0 = x0.to(device)
            B = x0.size(0)
            t = torch.randint(0, diffusion.T, (B,), device=device)
            x_t, noise = diffusion.q_sample(x0, t)
            t_norm = t.float() / diffusion.T
            pred_noise = model(x_t, t_norm)
            loss = F.mse_loss(pred_noise, noise)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1

            if step % 5000 == 0:
                avg = sum(losses[-5000:]) / 5000
                peak = torch.cuda.max_memory_allocated() / 1024**2
                P(f"{step:>7} | {avg:>10.4f} | {peak:>10.1f} | "
                  f"{time.time() - t0:>8.1f}")

    final_loss = sum(losses[-5000:]) / 5000
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    total_time = time.time() - t0

    P(f">>> {name} seed={seed}: loss={final_loss:.4f} | "
      f"peak_mem={peak_mem:.1f}MB | time={total_time:.1f}s")

    # 生成（只在 seed=42 时保存图）
    if seed == 42:
        P(f"  生成样本...")
        samples = diffusion.sample(model, n_samples=16, n_steps=50)
        samples = (samples.clamp(-1, 1) + 1) / 2

        os.makedirs('output', exist_ok=True)
        grid = samples.view(4, 4, 1, 28, 28).permute(0, 2, 1, 3, 4)
        grid = grid.reshape(4 * 28, 4 * 28)
        img = (grid.cpu().numpy() * 255).astype(np.uint8)
        save_path = f'output/mnist_{name}.png'
        Image.fromarray(img).save(save_path)
        P(f"  样本保存: {save_path}")

    del model, train_ds, train_loader
    torch.cuda.empty_cache()

    return final_loss, peak_mem, total_time, n_params


# ============================================================
# 主程序
# ============================================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    P(f"Device: {device}")
    if torch.cuda.is_available():
        P(f"GPU: {torch.cuda.get_device_name(0)}")
        P(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    DATA_ROOT = './data'   # MNIST 数据路径
    STEPS = 20000
    BATCH = 64
    SEEDS = [42]   # 跑通后可改为 [42, 123, 777]

    configs = [
        ("standard", "standard", 0),
        ("light_M8", "light", 8),
    ]

    results = {}
    for seed in SEEDS:
        for name, attn_type, n_modes in configs:
            loss, mem, t, params = train_and_generate(
                name, attn_type, n_modes, STEPS, BATCH,
                DATA_ROOT, device, seed)
            results[f"{name}_seed{seed}"] = {
                'name': name, 'seed': seed,
                'loss': loss, 'peak_mem': mem,
                'time': t, 'params': params,
            }

    P("\n" + "=" * 80)
    P("汇总")
    P("=" * 80)
    P(f"{'配置':<15} | {'seed':>5} | {'loss':>10} | "
      f"{'peak_mem(MB)':>14} | {'time(s)':>10} | {'params':>10}")
    P("-" * 80)
    for key, r in results.items():
        P(f"{r['name']:<15} | {r['seed']:>5} | {r['loss']:>10.4f} | "
          f"{r['peak_mem']:>14.1f} | {r['time']:>10.1f} | "
          f"{r['params']:>10,}")

    P("\n" + "=" * 80)
    P("对比")
    P("=" * 80)
    for seed in SEEDS:
        std = results[f"standard_seed{seed}"]
        light = results[f"light_M8_seed{seed}"]
        P(f"\nSeed = {seed}:")
        P(f"  loss:       标准={std['loss']:.4f}, 光团={light['loss']:.4f}")
        P(f"  显存:       标准={std['peak_mem']:.1f}MB, "
          f"光团={light['peak_mem']:.1f}MB")
        P(f"  时间:       标准={std['time']:.1f}s, 光团={light['time']:.1f}s")
        P(f"  参数:       标准={std['params']:,}, 光团={light['params']:,}")
        P(f"  参数比例:   {light['params'] / std['params']:.2f}x")
        P(f"  显存比例:   {light['peak_mem'] / std['peak_mem']:.2f}x")


if __name__ == "__main__":
    main()