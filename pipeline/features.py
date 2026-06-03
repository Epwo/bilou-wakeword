"""Étape 2 — convertit l'audio en embeddings openWakeWord.

C'est le cœur de la robustesse : les embeddings sont calculés via les modèles
ONNX d'openWakeWord (melspectrogram.onnx + embedding_model.onnx), entièrement
sous onnxruntime — AUCUNE dépendance torchaudio / torchmetrics / tensorflow.

Format de sortie : pour chaque clip audio, une matrice (n_windows, 16, 96)
où chaque fenêtre de 16 frames × 96 dims est un exemple d'entraînement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# Forme attendue par le classifieur : 16 frames d'embedding de dim 96.
N_FRAMES = 16
EMB_DIM = 96


class FeatureExtractor:
    """Enveloppe autour d'openwakeword.utils.AudioFeatures.

    On importe openwakeword *paresseusement* et uniquement le sous-module
    `utils` (ONNX), jamais `openwakeword.train` ni `openwakeword.data` qui
    tirent torchaudio.
    """

    def __init__(self):
        from openwakeword.utils import AudioFeatures
        # ncpu peu importe ici ; inference_framework onnx par défaut.
        self._af = AudioFeatures(inference_framework="onnx")

    def embed_clips(self, clips: np.ndarray) -> np.ndarray:
        """clips : (n_clips, n_samples) int16 ou float32, 16 kHz.
        Retourne les embeddings empilés en fenêtres (N, 16, 96)."""
        # AudioFeatures.embed_clips renvoie (n_clips, n_windows, 16, 96)
        emb = self._af.embed_clips(clips, batch_size=64)
        # Aplatit la dimension clips × windows → (total_windows, 16, 96)
        return emb.reshape(-1, N_FRAMES, EMB_DIM)


def load_wavs_as_array(wav_dir: str | Path, target_len: int = 32000) -> np.ndarray:
    """Charge tous les WAV d'un dossier en un tableau (n, target_len) float32.

    target_len = 2 s à 16 kHz (les clips de mot de réveil sont courts ; on
    pad/tronque à une longueur fixe pour le batch).
    """
    import soundfile as sf

    wav_dir = Path(wav_dir)
    clips = []
    for wav in sorted(wav_dir.glob("*.wav")):
        audio, sr = sf.read(wav, dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        # pad / tronque
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        clips.append(audio)
    if not clips:
        return np.zeros((0, target_len), dtype=np.float32)
    return np.stack(clips).astype(np.float32)
