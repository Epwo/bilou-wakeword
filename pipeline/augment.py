"""Étape 2b (optionnelle) — augmentation audio simple en numpy.

Remplace torch-audiomentations (qui casse sur Colab récent) par des
opérations numpy basiques : mix avec du bruit de fond à un SNR aléatoire,
convolution avec une réponse impulsionnelle de pièce (réverbération).

Objectif : rendre le modèle robuste au bruit ambiant sans dépendances
fragiles. Suffisant pour un wake word de bureau.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-9))


def mix_noise(clip: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Mixe `clip` avec `noise` au rapport signal/bruit `snr_db`."""
    if len(noise) < len(clip):
        # boucle le bruit pour couvrir le clip
        reps = int(np.ceil(len(clip) / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[:len(clip)]

    sig_rms, noise_rms = _rms(clip), _rms(noise)
    target_noise_rms = sig_rms / (10 ** (snr_db / 20))
    noise = noise * (target_noise_rms / (noise_rms + 1e-9))
    out = clip + noise
    # évite la saturation
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


def apply_reverb(clip: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Convolue le clip avec une réponse impulsionnelle (réverbération)."""
    out = np.convolve(clip, rir)[:len(clip)]
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 1.0:
        out = out / peak
    return out.astype(np.float32)


def augment_batch(
    clips: np.ndarray,
    noise_clips: list[np.ndarray] | None = None,
    rirs: list[np.ndarray] | None = None,
    snr_range: tuple[float, float] = (5.0, 20.0),
    p_noise: float = 0.7,
    p_reverb: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Applique aléatoirement bruit + réverbération à chaque clip.

    Retourne un tableau de même forme que `clips`.
    """
    rng = np.random.default_rng(seed)
    out = np.empty_like(clips)
    for i, clip in enumerate(clips):
        c = clip.copy()
        if rirs and rng.random() < p_reverb:
            c = apply_reverb(c, rirs[rng.integers(len(rirs))])
        if noise_clips and rng.random() < p_noise:
            snr = rng.uniform(*snr_range)
            c = mix_noise(c, noise_clips[rng.integers(len(noise_clips))], snr)
        out[i] = c
    return out
