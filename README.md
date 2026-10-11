# FFT- Optical Propagation Attention

This repository investigates two independent but combinable research directions:

1. **FFT Attention**: Replaces standard QK^T computation with FFT-based frequency-domain operations. In the current SDXL validation, an observed speedup of roughly 25% was recorded.
2. **Phase-Difference Encoding**: Encodes scalar values in relative phase differences across multiple modes, providing natural immunity to common-mode noise. In the current SDXL validation, output differences were visually imperceptible.

Both approaches have been tested in real SDXL image generation pipelines.

## Quick Start

### Installation

```bash
git clone https://github.com/smlovepp-sys/FFT-.git
cd FFT-
```

### Basic Usage

```python
from opa import OpticalMultiheadAttention

attn = OpticalMultiheadAttention(
    d_model=768,
    n_heads=12,
    w_per_head=4,
    max_seq_len=4096,
    causal=True,
)

output = attn(x)  # x: (B, N, d_model)
```

Note: Unlike standard attention, OpticalMultiheadAttention takes only a single input `x`. Query, Key, and Value are derived from the same optical field.

## Why Phase-Difference Encoding Matters

In real optical systems, shared noise affects all modes simultaneously: thermal drift, frequency fluctuations, and gain variations are common-mode effects.

Standard independent-phase encoding fails under such noise. Phase-difference encoding removes the shared component naturally.

The centered/differential decoding form is:

$$\hat{u} = \frac{\sum_i (w_i - \bar{w})(\phi_i - \bar{\phi})}{2\pi \sum_i (w_i - \bar{w})^2}$$

This cancels any shared phase offset while preserving signal information.

See [THEORY.md](THEORY.md) for the full derivation and validation notes.

## Key Findings

### 1) FFT Configuration Learnability

| Configuration | Avg R² | Fine-tuning |
|---|---:|---|
| FFT phase=0.01 | 0.9117 | Easy |
| FFT phase=0.1 | 0.9114 | Easy |
| FFT phase=0.5 | 0.9060 | Easy |
| Phase align α=0.5 | 0.6292 | Requires extensive training |
| Phase align α=1.0 | -0.0418 | Difficult |

**Conclusion**: FFT-based configurations are far easier to learn and tune than phase-alignment alternatives.

### 2) Common-Mode Noise Sweep

The advantage of phase-difference encoding appears precisely when common-mode noise dominates.

| Common-mode ratio | Original LS MSE | Centered LS MSE | Winner |
|---:|---:|---:|---|
| 0.0 | 1.94e-01 | 1.16e+00 | Original LS |
| 0.5 | 4.20e-01 | 5.80e-01 | Original LS |
| 0.8 | 5.56e-01 | 2.32e-01 | Centered LS (2.4×) |
| 1.0 | 6.47e-01 | 1.09e-10 | Centered LS (1e9×) |

**Crossover point**: common-mode ratio ≈ 0.55–0.60. Above this threshold, phase-difference encoding wins decisively.

### 3) Optimal Weight Distribution

The weight vector w determines the trade-off between differential efficiency and common-mode immunity.

| Distribution | F_orig/F_cent | 30 dB + 80% common MSE |
|---|---:|---:|
| uniform | ∞ (degenerate) | nan |
| linear [1..8] | 4.86 | 0.71 |
| quadratic | 2.46 | 0.36 |
| **exponential** | **1.59** | **0.23** |

**Optimal weight**: w_i = 2^(i-1)

The exponential distribution simultaneously maximizes differential efficiency (Fisher information in pure noise) and common-mode immunity. This is the empirically optimal choice across all tested noise mixtures.

### 4) Effect of Mode Count

More modes reduce encoding error but increase physical complexity.

| Modes | 30 dB + 80% common MSE |
|---:|---:|
| 4 | 0.232 |
| 8 | 0.028 |
| 12 | 0.008 |

**Recommendation**: 8 modes provides a practical balance between precision and physical device complexity.

### 5) Real SDXL Pipeline Validation

Phase-difference encoding was applied to every noise prediction in a 20-step SDXL denoising loop using 8 exponential-weight modes at SNR=30 dB and common-mode ratio=0.8.

**Encoding Performance**:
- max reconstruction error: 1.95e-03
- mean MSE: 8.82e-08

**Output Quality**:
- max pixel difference vs baseline: 57 / 255
- mean pixel difference: 0.91 / 255 (≈ 0.0036%)

The maximum difference is concentrated in a very small number of edge pixels; the average difference is extremely low. Visually, the generated image is effectively indistinguishable from the baseline.

## Mathematical Framework

### Fisher Information Analysis

Under Gaussian noise with variance σ²:

- Original LS: F_orig = Σ w_i² / σ²
- Centered LS: F_cent = Σ (w_i - w_bar)² / σ²

For linear weights [1, 2, ..., M], the ratio F_orig / F_cent ranges from 4.55 (M=12) to 10.00 (M=2).

This is not a bug; it reflects a fundamental trade-off: original LS preserves more information under pure differential noise, while centered LS rejects common-mode noise completely.

**See [THEORY.md](THEORY.md) for the full derivation and validation notes.**

## Research Status

Both FFT attention and phase-difference encoding are active research directions, not finalized claims.

The current evidence strongly suggests:

1. FFT attention is practical for sequence modeling and offers speed benefits.
2. Phase-difference encoding is a physically principled approach to shared-noise rejection.
3. Both can coexist in the same system and are compatible with modern diffusion models.

## Repository Structure

- `README.md` — overview and quick results
- `THEORY.md` — mathematical derivation, Fisher information, and detailed validation
- `opa.py` — FFT attention implementation
- `experiments/` — synthetic and pipeline validation scripts
- `output/` — generated images and benchmark results

## Citation

```bibtex
@software{fft_optical_attention,
  title={FFT-based Optical Propagation Attention},
  author={smlovepp-sys},
  year={2026},
  url={https://github.com/smlovepp-sys/FFT-}
}
```

## License

License TBD.

## Contributing

Contributions are welcome for:

- Attention implementation improvements
- Weight optimization strategies
- Noise model validation
- Real optical system deployment

## Contact

Please open an issue or discussion on GitHub for questions or suggestions.
