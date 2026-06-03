"""Étape 2 — convertit l'audio en embeddings openWakeWord.

Robustesse : embeddings calculés via les modèles ONNX d'openWakeWord
(melspectrogram + embedding), sous onnxruntime — AUCUNE dépendance torchaudio.

`AudioFeatures.embed_clips(x)` retourne (N_clips, frames, 96). Le classifieur
attend des fenêtres de 16 frames × 96. On découpe donc chaque clip en
quelques fenêtres glissantes de 16 frames (vers la fin du clip, là où le mot
est prononcé).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


N_FRAMES = 16
EMB_DIM = 96


class FeatureExtractor:
    """Enveloppe autour d'openwakeword.utils.AudioFeatures (ONNX uniquement)."""

    def __init__(self):
        from openwakeword.utils import AudioFeatures
        # Selon la version pip, l'argument `inference_framework` peut ne pas
        # exister → on appelle sans argument (les défauts utilisent ONNX).
        self._af = AudioFeatures()

    def embed_clips(self, clips: np.ndarray,
                    windows_per_clip: int = 4) -> np.ndarray:
        """clips : (n_clips, n_samples) float32 16 kHz, tous même longueur.

        Retourne (M, 16, 96) : pour chaque clip, on garde les
        `windows_per_clip` dernières fenêtres glissantes de 16 frames
        (le mot est vers la fin du clip).
        """
        # openWakeWord exige de l'audio 16-bit PCM (int16), pas du float32.
        if clips.dtype != np.int16:
            clips = np.clip(clips, -1.0, 1.0)
            clips = (clips * 32767.0).astype(np.int16)

        emb = self._af.embed_clips(clips, batch_size=64)  # (n, frames, 96)
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 2:
            # certaines versions renvoient (frames, 96) pour un seul clip
            emb = emb[None]

        windows = []
        n, frames, dim = emb.shape
        for i in range(n):
            e = emb[i]                       # (frames, 96)
            if frames < N_FRAMES:
                pad = np.zeros((N_FRAMES - frames, dim), dtype=e.dtype)
                e = np.concatenate([pad, e])
                f = N_FRAMES
            else:
                f = frames
            # `windows_per_clip` dernières positions de fenêtre
            last_start = f - N_FRAMES
            first_start = max(0, last_start - windows_per_clip + 1)
            for s in range(first_start, last_start + 1):
                windows.append(e[s:s + N_FRAMES])

        if not windows:
            return np.zeros((0, N_FRAMES, EMB_DIM), dtype=np.float32)
        return np.stack(windows).astype(np.float32)


def load_wavs_as_array(wav_dir: str | Path, target_len: int = 48000) -> np.ndarray:
    """Charge tous les WAV en (n, target_len) float32, 16 kHz mono.

    target_len = 3 s à 16 kHz : assez long pour produire ≥ 16 frames
    d'embedding (≈ 24 frames), même pour les clips lents.
    """
    import soundfile as sf

    wav_dir = Path(wav_dir)
    clips = []
    for wav in sorted(wav_dir.glob("*.wav")):
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        clips.append(audio)
    if not clips:
        return np.zeros((0, target_len), dtype=np.float32)
    return np.stack(clips).astype(np.float32)
