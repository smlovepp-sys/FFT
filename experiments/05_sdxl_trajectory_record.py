"""
SDXL 生成轨迹记录
==================
跑一次 SDXL，把完整中间过程存到本地。

注意：需要 SDXL 模型。没有的话跳过。
"""

import os, gc, time, json
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_PATH = r"D:\sd-webui-forge-neo\models\Stable-diffusion\njsmixRe_titanium.safetensors"
SAVE_DIR = "./trajectory/seed_42_clear_v3"

import torch
from PIL import Image


def P(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def main():
    SEED = 42
    WIDTH, HEIGHT = 1024, 1024
    STEPS = 20
    CFG = 5.5
    PROMPT = "1girl, sitting, red dress, silver hair, purple eyes, beautiful, elegant"
    NEG_PROMPT = ""

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "latents"), exist_ok=True)
    os.makedirs(os.path.join(SAVE_DIR, "noise_preds"), exist_ok=True)

    from diffusers import StableDiffusionXLPipeline

    gc.collect(); torch.cuda.empty_cache()

    pipe = StableDiffusionXLPipeline.from_single_file(
        MODEL_PATH, torch_dtype=torch.float16, use_safetensors=True)
    pipe.vae.to(torch.float32)
    pipe.vae.config.force_upcast = True
    pipe.enable_model_cpu_offload()
    try:
        pipe.vae.enable_slicing()
    except Exception:
        pass

    DEVICE = "cuda"

    # 预热
    with torch.no_grad():
        dummy = torch.randn(1, 4, HEIGHT // 8, WIDTH // 8,
                            device=DEVICE, dtype=torch.float32)
        _ = pipe.vae.decode(dummy).sample
        del dummy
    pipe.scheduler.set_timesteps(STEPS)

    with torch.no_grad():
        prompt_embeds, negative_embeds, pooled, negative_pooled = \
            pipe.encode_prompt(PROMPT, device=DEVICE,
                               num_images_per_prompt=1,
                               do_classifier_free_guidance=True,
                               negative_prompt=NEG_PROMPT)

    torch.save(prompt_embeds.cpu(), f"{SAVE_DIR}/prompt_embeds.pt")
    torch.save(negative_embeds.cpu(), f"{SAVE_DIR}/negative_embeds.pt")
    torch.save(pooled.cpu(), f"{SAVE_DIR}/pooled.pt")
    torch.save(negative_pooled.cpu(), f"{SAVE_DIR}/negative_pooled.pt")

    add_time_ids = pipe._get_add_time_ids(
        original_size=(HEIGHT, WIDTH),
        crops_coords_top_left=(0, 0),
        target_size=(HEIGHT, WIDTH),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=pooled.shape[-1],
    ).to(DEVICE)
    add_time_ids_cond = torch.cat([add_time_ids] * 2)
    torch.save(add_time_ids_cond.cpu(), f"{SAVE_DIR}/add_time_ids.pt")

    lat_h, lat_w = HEIGHT // 8, WIDTH // 8
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    latents = torch.randn(1, 4, lat_h, lat_w,
                          generator=generator, device=DEVICE,
                          dtype=torch.float16)
    latents = latents * pipe.scheduler.init_noise_sigma
    torch.save(latents.cpu(), f"{SAVE_DIR}/latents/step_init.pt")

    all_timesteps = pipe.scheduler.timesteps
    emb_C = torch.cat([negative_embeds, prompt_embeds])
    pooled_C = torch.cat([negative_pooled, pooled])

    with torch.no_grad():
        for i, t in enumerate(all_timesteps):
            latent_input = torch.cat([latents] * 2)
            latent_input = pipe.scheduler.scale_model_input(latent_input, t)

            noise_pred = pipe.unet(
                latent_input, t,
                encoder_hidden_states=emb_C,
                added_cond_kwargs={"text_embeds": pooled_C,
                                   "time_ids": add_time_ids_cond},
            ).sample

            n_uncond, n_cond = noise_pred.chunk(2)
            noise_pred_cfg = n_uncond + CFG * (n_cond - n_uncond)

            torch.save(latent_input.cpu(),
                       f"{SAVE_DIR}/latents/step_{i:02d}_input.pt")
            torch.save(noise_pred.cpu(),
                       f"{SAVE_DIR}/noise_preds/step_{i:02d}_raw.pt")
            torch.save(noise_pred_cfg.cpu(),
                       f"{SAVE_DIR}/noise_preds/step_{i:02d}_cfg.pt")

            latents = pipe.scheduler.step(noise_pred_cfg, t, latents).prev_sample
            torch.save(latents.cpu(),
                       f"{SAVE_DIR}/latents/step_{i:02d}_after.pt")

            P(f"  step {i:2d}: t={t.item():.0f}, std={latents.float().std():.4f}")

    torch.save(latents.cpu(), f"{SAVE_DIR}/final_latent.pt")

    with torch.no_grad():
        latents_f32 = latents.to(torch.float32) / pipe.vae.config.scaling_factor
        image = pipe.vae.decode(latents_f32).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.float().permute(0, 2, 3, 1).cpu().numpy()[0]
    Image.fromarray((image * 255).astype("uint8")).save(
        f"{SAVE_DIR}/final_image.png")

    meta = {
        'seed': SEED, 'width': WIDTH, 'height': HEIGHT,
        'steps': STEPS, 'cfg': CFG,
        'prompt': PROMPT, 'negative_prompt': NEG_PROMPT,
    }
    with open(f"{SAVE_DIR}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    P(f"完成。数据在 {SAVE_DIR}/")


if __name__ == "__main__":
    main()