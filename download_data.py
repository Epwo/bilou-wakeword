"""Télécharge les ressources nécessaires : voix piper française + features
négatives pré-calculées d'openWakeWord.

    python download_data.py

Place :
  voices/fr_FR-upmc-medium.onnx(.json)
  data/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
  data/validation_set_features.npy   (optionnel, pour estimer le taux de FP)
"""

from __future__ import annotations

import urllib.request
from pathlib import Path


FILES = {
    "voices/fr_FR-upmc-medium.onnx":
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx",
    "voices/fr_FR-upmc-medium.onnx.json":
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx.json",
    "data/openwakeword_features_ACAV100M_2000_hrs_16bit.npy":
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
    "data/validation_set_features.npy":
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy",
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  ✓ déjà présent : {dest}")
        return
    print(f"  ↓ {dest}  ←  {url}")
    urllib.request.urlretrieve(url, dest)
    mb = dest.stat().st_size / 1e6
    print(f"    ({mb:.1f} Mo)")


def main():
    print("Téléchargement des ressources...")
    for rel, url in FILES.items():
        _download(url, Path(rel))
    print("\nTerminé. Tu peux lancer : python run_all.py --word bilou")


if __name__ == "__main__":
    main()
