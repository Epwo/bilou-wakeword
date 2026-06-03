"""Étape 1 — génère des échantillons synthétiques du mot de réveil.

Utilise la NOUVELLE API de piper-sample-generator (le package CLI moderne,
`python -m piper_sample_generator`), pas l'ancien `import generate_samples`
qui a été supprimé en mars 2026.

Le mot est prononcé par une voix TTS française (fr_FR-upmc-medium) avec des
variations de vitesse pour couvrir différentes façons de le dire.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def generate_positive_samples(
    word: str,
    out_dir: str | Path,
    voice_onnx: str | Path,
    n_samples: int = 2000,
    batch_size: int = 10,
    length_scales: tuple[float, ...] = (0.9, 1.0, 1.1, 1.2),
) -> Path:
    """Génère `n_samples` clips WAV du mot via piper.

    `length_scales` = vitesses de parole (plus petit = plus rapide). On
    cycle dessus pour varier la prononciation.

    Retourne le dossier de sortie.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # piper-sample-generator (nouvelle API) s'utilise en CLI :
    #   python -m piper_sample_generator '<texte>' --model <voice.onnx>
    #       --max-samples N --output-dir DIR --batch-size B
    cmd = [
        sys.executable, "-m", "piper_sample_generator",
        word,
        "--model", str(voice_onnx),
        "--max-samples", str(n_samples),
        "--output-dir", str(out_dir),
        "--batch-size", str(batch_size),
    ]
    # Les versions récentes acceptent plusieurs --length-scale pour varier
    # la vitesse. On les passe une par une.
    for ls in length_scales:
        cmd += ["--length-scale", str(ls)]

    print(f"[generate] {n_samples} échantillons de « {word} » → {out_dir}")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    wavs = list(out_dir.glob("*.wav"))
    print(f"[generate] {len(wavs)} fichiers WAV produits.")
    return out_dir
