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

    def embed_clips(self, clips: np.ndarray) -> np.ndarray:
        """clips : (n_clips, n_samples) float32 16 kHz, déjà centrés sur la
        parole (≈ 2 s chacun → ≈ 16 frames d'embedding).

        Retourne (M, 16, 96) : pour chaque clip, on garde TOUTES les fenêtres
        de 16 frames disponibles (généralement 1, parfois 2-3 si le clip est
        un peu plus long). Comme les clips sont centrés sur le mot, ces
        fenêtres contiennent bien le mot — pas du silence.
        """
        # openWakeWord exige de l'audio 16-bit PCM (int16), pas du float32.
        if clips.dtype != np.int16:
            clips = np.clip(clips, -1.0, 1.0)
            clips = (clips * 32767.0).astype(np.int16)

        emb = self._af.embed_clips(clips, batch_size=64)  # (n, frames, 96)
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 2:
            emb = emb[None]

        windows = []
        n, frames, dim = emb.shape
        for i in range(n):
            e = emb[i]
            if frames < N_FRAMES:
                pad = np.zeros((N_FRAMES - frames, dim), dtype=e.dtype)
                e = np.concatenate([pad, e])
                f = N_FRAMES
            else:
                f = frames
            for s in range(0, f - N_FRAMES + 1):     # toutes les fenêtres
                windows.append(e[s:s + N_FRAMES])

        if not windows:
            return np.zeros((0, N_FRAMES, EMB_DIM), dtype=np.float32)
        return np.stack(windows).astype(np.float32)


def extract_speech_window(audio: np.ndarray, win: int = 32000) -> np.ndarray:
    """Extrait la fenêtre de `win` samples (≈ 2 s) la plus énergique du clip
    = celle qui contient le mot prononcé.

    Crucial : les clips piper ont le mot au début/milieu puis du silence.
    Sans ce centrage, on risque d'entraîner le modèle sur du silence.
    """
    n = len(audio)
    if n <= win:
        pad = win - n
        return np.pad(audio, (pad // 2, pad - pad // 2)).astype(np.float32)
    # énergie de chaque fenêtre glissante via somme cumulée (rapide)
    energy = audio.astype(np.float64) ** 2
    cumsum = np.concatenate([[0.0], np.cumsum(energy)])
    win_energy = cumsum[win:] - cumsum[:-win]
    start = int(np.argmax(win_energy))
    return audio[start:start + win].astype(np.float32)


def load_wavs_as_array(wav_dir: str | Path, win: int = 32000) -> np.ndarray:
    """Charge tous les WAV, centrés sur la parole, en (n, win) float32 16 kHz.

    `win` = 32000 (2 s) : une fenêtre de 16 frames d'embedding couvre ~2 s,
    donc chaque clip produit ≈ 1 fenêtre contenant le mot.
    """
    import soundfile as sf

    wav_dir = Path(wav_dir)
    clips = []
    for wav in sorted(wav_dir.glob("*.wav")):
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        clips.append(extract_speech_window(audio, win))
    if not clips:
        return np.zeros((0, win), dtype=np.float32)
    return np.stack(clips).astype(np.float32)
