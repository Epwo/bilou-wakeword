"""Pipeline complète d'entraînement d'un wake word, de A à Z.

    python run_all.py --word bilou --n-samples 2000 --epochs 20

Étapes :
  1. génère des échantillons synthétiques du mot (piper, voix française)
  2. (optionnel) augmente avec bruit/réverbération
  3. calcule les embeddings openWakeWord (ONNX)
  4. entraîne le classifieur sur positifs vs négatifs pré-calculés
  5. exporte <word>.onnx

Conçu pour tourner sur Colab (GPU) ou en local. Voir README.md pour le setup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--word", default="bilou", help="Mot de réveil.")
    p.add_argument("--voice", default="voices/fr_FR-upmc-medium.onnx",
                   help="Voix piper (.onnx) pour générer les samples.")
    p.add_argument("--neg-features", default="data/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
                   help="Features négatives pré-calculées openWakeWord (.npy).")
    p.add_argument("--n-samples", type=int, default=2000,
                   help="Nombre d'échantillons positifs à générer.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--out", default=None,
                   help="Chemin du .onnx de sortie (défaut: models/<word>.onnx).")
    p.add_argument("--workdir", default="work",
                   help="Dossier de travail (samples, etc.).")
    p.add_argument("--no-augment", action="store_true",
                   help="Désactive l'augmentation bruit/réverbération.")
    p.add_argument("--noise-dir", default=None,
                   help="Dossier de WAV de bruit pour l'augmentation (optionnel).")
    p.add_argument("--extra-positives", default=None,
                   help="Dossier de WAV de TA voix disant le mot (record_samples.py). "
                        "Fortement augmentés et ajoutés aux positifs synthétiques.")
    p.add_argument("--extra-aug", type=int, default=30,
                   help="Nombre de variantes par vrai clip (défaut 30).")
    args = p.parse_args()

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    out_onnx = args.out or f"models/{args.word}.onnx"

    # --- 1. génération -----------------------------------------------------
    from pipeline.generate import generate_positive_samples
    sample_dir = generate_positive_samples(
        args.word, work / "positive_samples", args.voice,
        n_samples=args.n_samples,
    )

    # --- charge l'audio ----------------------------------------------------
    from pipeline.features import load_wavs_as_array, FeatureExtractor
    clips = load_wavs_as_array(sample_dir)
    print(f"[run] {len(clips)} clips chargés.")

    # --- 2. augmentation (optionnelle) ------------------------------------
    if not args.no_augment and args.noise_dir:
        from pipeline.augment import augment_batch
        noise = list(load_wavs_as_array(args.noise_dir, target_len=32000))
        print(f"[run] augmentation avec {len(noise)} clips de bruit.")
        aug = augment_batch(clips, noise_clips=noise, seed=1)
        clips = np.concatenate([clips, aug])  # double le dataset
        print(f"[run] dataset positif augmenté → {len(clips)} clips.")

    # --- 2c. vrais enregistrements de ta voix (très efficace) -------------
    if args.extra_positives:
        from pipeline.augment import augment_simple
        real = load_wavs_as_array(args.extra_positives)
        if len(real) == 0:
            print(f"[run] ⚠ aucun WAV dans {args.extra_positives}")
        else:
            real_aug = augment_simple(real, n_aug=args.extra_aug)
            print(f"[run] {len(real)} vrais clips → {len(real_aug)} après augmentation.")
            clips = np.concatenate([clips, real_aug])
            print(f"[run] dataset positif total → {len(clips)} clips.")

    # --- 3. embeddings -----------------------------------------------------
    print("[run] calcul des embeddings (ONNX)...")
    fe = FeatureExtractor()
    pos_features = fe.embed_clips(clips)
    print(f"[run] {len(pos_features)} fenêtres d'embedding positives.")

    # --- 4. entraînement ---------------------------------------------------
    from pipeline.train import train
    metrics = train(
        pos_features, args.neg_features, out_onnx,
        epochs=args.epochs,
    )
    print(f"\n[run] terminé. métriques: {metrics}")
    print(f"[run] modèle: {out_onnx}")


if __name__ == "__main__":
    main()
