"""
相位差缠绕走 SDXL 完整流程
============================
从轨迹读 noise_pred，用相位差缠绕承载，走完 20 步去噪。
"""

import os, json
import numpy as np
from PIL import Image


def P(*args, **kwargs):
    print(*args, **kwargs, flush=True)


TRAJ_DIR = "./trajectory/seed_42_clear_v3"
OUTPUT_DIR = "./output"
DEVICE = "cuda"
N_MODES = 8
SNR_DB = 30.0
COMMON_RATIO = 0.8


def main():
    import torch
    from diffusers import StableDiffusionXLPipeline

    MODEL_PATH = r"D:\sd-webui-forge-neo\models\Stable-diffusion\njsmixRe_titanium.safetensors"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipe = StableDiffusionXLPipeline.from_single_file(
        MODEL_PATH, torch_dtype=torch.float16, use_safetensors=True)
    pipe.vae.to(torch.float32)
    pipe.vae.config.force_upcast = True
    pipe.enable_model_cpu_offload()

    with open(f"{TRAJ_DIR}/meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    STEPS = meta['steps']
    CFG = meta['cfg']

    pipe.scheduler.set_timesteps(STEPS)
    all_timesteps = pipe.scheduler.timesteps

    latents = torch.load(f"{TRAJ_DIR}/latents/step_init.pt",
                         map_location=DEVICE).to(torch.float16)

    # 指数权重
    w = torch.tensor([2 ** i for i in range(N_MODES)],
                     dtype=torch.float32, device=DEVICE)
    w_bar = w.mean()
    w_centered = w - w_bar
    denom = (w_centered ** 2).sum().item()

    with torch.no_grad():
        for i, t in enumerate(all_timesteps):
            noise_pred_raw = torch.load(
                f"{TRAJ_DIR}/noise_preds/step_{i:02d}_raw.pt",
                map_location=DEVICE).to(torch.float16)

            # 相位差缠绕往返
            x = noise_pred_raw.float().flatten()
            v_min, v_max = x.min(), x.max()
            u = (x - v_min) / (v_max - v_min)

            phi = u.unsqueeze(-1) * 2 * np.pi * w.unsqueeze(0)
            snr_lin = 10 ** (SNR_DB / 10)
            S = (2 * np.pi) ** 2 / 12
            sigma_c = (S * COMMON_RATIO / snr_lin) ** 0.5
            sigma_d = (S * (1 - COMMON_RATIO) / snr_lin) ** 0.5
            phi = phi + torch.randn(phi.shape[0], 1, device=DEVICE) * sigma_c \
                      + torch.randn_like(phi) * sigma_d

            phi_c = phi - phi.mean(dim=-1, keepdim=True)
            num = (phi_c * w_centered.unsqueeze(0)).sum(dim=-1)
            u_hat = num / (2 * np.pi * denom)
            x_rec = u_hat * (v_max - v_min) + v_min
            noise_pred_light = x_rec.view(noise_pred_raw.shape).to(torch.float16)

            # CFG
            n_uncond, n_cond = noise_pred_light.chunk(2)
            noise_pred_cfg = n_uncond + CFG * (n_cond - n_uncond)

            latents = pipe.scheduler.step(
                noise_pred_cfg, t, latents).prev_sample

            if (i + 1) % 5 == 0:
                P(f"  step {i}")

    with torch.no_grad():
        latents_f32 = latents.to(torch.float32) / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(latents_f32).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.float().permute(0, 2, 3, 1).cpu().numpy()[0]
    out_path = f"{OUTPUT_DIR}/phase_diff_8mode_30db.png"
    Image.fromarray((image * 255).astype("uint8")).save(out_path)
    P(f"保存: {out_path}")


if __name__ == "__main__":
    main()