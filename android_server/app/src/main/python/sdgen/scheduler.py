"""Euler discrete scheduler as configured for stabilityai/sd-turbo (trailing timestep spacing).

scheduler_config: scaled_linear betas 0.00085..0.012, 1000 train steps, epsilon prediction,
timestep_spacing=trailing. Use 1-4 steps and no classifier-free guidance.
"""
import numpy as np

NUM_TRAIN_STEPS = 1000


def alphas_cumprod(n=NUM_TRAIN_STEPS, beta_start=0.00085, beta_end=0.012):
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, n, dtype=np.float64) ** 2
    return np.cumprod(1.0 - betas)


def euler_trailing(steps, n=NUM_TRAIN_STEPS):
    """Returns (timesteps[steps], sigmas[steps + 1], init_noise_sigma)."""
    if steps < 1:
        raise ValueError("steps must be >= 1")
    ac = alphas_cumprod(n)
    all_sigmas = ((1.0 - ac) / ac) ** 0.5
    ts = np.arange(n, 0, -n / steps).round().astype(np.float64)[:steps] - 1.0
    sigmas = np.append(np.interp(ts, np.arange(n), all_sigmas), 0.0)
    # diffusers' EulerDiscreteScheduler.init_noise_sigma is max(sigmas) for "linspace"/"trailing" spacing
    # (and sqrt(max^2 + 1) only for "leading").
    init_sigma = float(sigmas.max())
    return ts, sigmas, init_sigma
