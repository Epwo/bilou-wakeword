"""Enregistre des échantillons de TA voix disant le mot de réveil.

Tourne sur ton Mac (micro). Ces vrais clips, ajoutés aux samples synthétiques
à l'entraînement, améliorent énormément la détection sur ta voix réelle.

    python record_samples.py --word bilou --n 30 --out my_voice

Pour chaque enregistrement : appuie sur Entrée, attends le « parle ! »,
prononce le mot une fois, clairement. Varie un peu (proche/loin, ton normal/
enjoué) pour couvrir différentes conditions.

Dépendances : sounddevice, soundfile, numpy (présents dans .venv_wake).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--word", default="bilou", help="Mot à enregistrer.")
    p.add_argument("--n", type=int, default=30, help="Nombre d'enregistrements.")
    p.add_argument("--out", default="my_voice", help="Dossier de sortie.")
    p.add_argument("--duration", type=float, default=1.5,
                   help="Durée de chaque enregistrement (s).")
    p.add_argument("--device", type=int, default=None,
                   help="Index du micro (voir python -m sounddevice).")
    args = p.parse_args()

    sr = 16000
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_frames = int(args.duration * sr)

    # Reprend la numérotation si des fichiers existent déjà
    existing = len(list(out.glob(f"{args.word}_*.wav")))
    if existing:
        print(f"({existing} enregistrements déjà présents — on continue)")

    print(f"\nOn enregistre {args.n} fois « {args.word} ».")
    print("Pour chaque : Entrée → attends « parle ! » → dis le mot UNE fois.\n")

    i = existing
    target = existing + args.n
    while i < target:
        try:
            input(f"[{i - existing + 1}/{args.n}] Entrée pour enregistrer "
                  f"(ou Ctrl-C pour arrêter)... ")
        except (EOFError, KeyboardInterrupt):
            print("\nArrêt.")
            break

        # petit délai + signal
        time.sleep(0.2)
        print("  🔴 parle !", flush=True)
        audio = sd.rec(n_frames, samplerate=sr, channels=1,
                       dtype="float32", device=args.device)
        sd.wait()
        audio = audio.flatten()

        # vérifie qu'il y a bien du signal
        rms = float(np.sqrt(np.mean(audio ** 2) + 1e-9))
        if rms < 0.005:
            print("  ⚠ trop faible / silence — on recommence celui-là")
            continue

        path = out / f"{args.word}_{i:03d}.wav"
        sf.write(path, audio, sr, subtype="PCM_16")
        print(f"  ✓ {path}  (niveau {rms:.3f})")
        i += 1

    total = len(list(out.glob(f"{args.word}_*.wav")))
    print(f"\n{total} échantillons dans {out}/")
    print(f"Zippe ce dossier et upload-le sur Colab : "
          f"  zip -r {args.out}.zip {args.out}")


if __name__ == "__main__":
    main()
