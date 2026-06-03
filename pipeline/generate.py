"""Étape 1 — génère des échantillons synthétiques du mot de réveil.

IMPORTANT : le package PyPI `piper-sample-generator` est mal packagé (il
référence `piper_train` qui n'est pas inclus dans le wheel). On CLONE donc le
repo et on lance la commande depuis sa racine, où `piper_sample_generator/`
ET `piper_train/` sont tous deux présents.

API CLI réelle (vérifiée dans __main__.py) :
    python -m piper_sample_generator <texte>
        --model <voice.onnx>  --max-samples N  --output-dir DIR
        --length-scales 0.9 1.0 1.1 1.2          # PLURIEL, nargs="+"
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PIPER_REPO = "https://github.com/rhasspy/piper-sample-generator"


def ensure_piper(clone_dir: str | Path = "piper-sample-generator") -> Path:
    """Clone le repo piper-sample-generator (avec piper_train) + installe la
    phonémisation nécessaire aux voix .onnx. Idempotent."""
    clone_dir = Path(clone_dir)
    if not (clone_dir / "piper_train").exists():
        if clone_dir.exists():
            subprocess.run(["rm", "-rf", str(clone_dir)], check=True)
        print(f"[generate] clone de {PIPER_REPO}")
        subprocess.run(["git", "clone", "--depth", "1", PIPER_REPO, str(clone_dir)],
                       check=True)
        # __main__.py importe la NOUVELLE API piper-tts :
        #   from piper import PiperVoice, SynthesisConfig
        #   from piper.phonemize_espeak import EspeakPhonemizer
        # → il faut le package `piper-tts` (récent) + le binaire espeak-ng.
        print("[generate] install de piper-tts")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "piper-tts"],
                       check=True)
        # espeak-ng (binaire) pour la phonémisation. check=False : si apt
        # n'existe pas (hors Colab/Debian), piper-tts a souvent un fallback.
        print("[generate] install de espeak-ng (apt)")
        subprocess.run(["apt-get", "install", "-y", "-q", "espeak-ng"], check=False)
    return clone_dir.resolve()


def generate_positive_samples(
    word: str,
    out_dir: str | Path,
    voice_onnx: str | Path,
    n_samples: int = 2000,
    length_scales: tuple[float, ...] = (0.9, 1.0, 1.1, 1.2),
    clone_dir: str | Path = "piper-sample-generator",
) -> Path:
    """Génère `n_samples` clips WAV du mot via piper. Retourne le dossier."""
    piper_root = ensure_piper(clone_dir)

    # Chemins absolus car on lance avec cwd = repo cloné.
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    voice_onnx = Path(voice_onnx).resolve()
    if not voice_onnx.exists():
        raise FileNotFoundError(f"Voix introuvable : {voice_onnx}")

    # piper répartit les samples sur les length_scales ; il faut au moins
    # autant de samples que de scales, sinon certaines configs sont vides.
    scales = list(length_scales)
    if n_samples < len(scales):
        scales = scales[:max(1, n_samples)]

    cmd = [
        sys.executable, "-m", "piper_sample_generator",
        word,
        "--model", str(voice_onnx),
        "--max-samples", str(n_samples),
        "--output-dir", str(out_dir),
        "--length-scales", *[str(x) for x in scales],
    ]
    print(f"[generate] {n_samples} échantillons de « {word} » → {out_dir}")
    print("  (cwd=%s) %s" % (piper_root, " ".join(cmd)))
    # cwd = racine du repo → piper_train importable. On capture la sortie pour
    # afficher le vrai message d'erreur de piper (sinon masqué par subprocess).
    proc = subprocess.run(cmd, cwd=str(piper_root), capture_output=True, text=True)
    if proc.returncode != 0:
        print("=== STDOUT piper ===\n" + (proc.stdout or "")[-2000:])
        print("=== STDERR piper ===\n" + (proc.stderr or "")[-4000:])
        raise RuntimeError(f"piper a échoué (exit {proc.returncode}) — voir stderr ci-dessus")

    wavs = list(out_dir.glob("*.wav"))
    print(f"[generate] {len(wavs)} fichiers WAV produits.")
    return out_dir
