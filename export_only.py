"""Ré-exporte un modèle déjà entraîné (.pt) en ONNX, sans refaire la pipeline.

Pratique si l'entraînement a réussi mais l'export ONNX a planté : les poids
ont été sauvés en .pt, on les recharge et on exporte en quelques secondes.

    python export_only.py --ckpt models/bilou.pt --out models/bilou.onnx
"""

from __future__ import annotations

import argparse

import torch

from pipeline.model import WakeWordModel, export_onnx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/bilou.pt", help="Poids .pt sauvés.")
    p.add_argument("--out", default=None, help="Sortie .onnx (défaut: même nom).")
    args = p.parse_args()

    out = args.out or args.ckpt.replace(".pt", ".onnx")
    model = WakeWordModel()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    export_onnx(model, out)
    print(f"OK → {out}")


if __name__ == "__main__":
    main()
