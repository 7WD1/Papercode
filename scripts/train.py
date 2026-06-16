#!/usr/bin/env python3
"""
Standalone training entry point for HRLAD and baselines.

Usage:
    python scripts/train.py --method HRLAD --dataset SPSD --output-dir checkpoints/
    python scripts/train.py --method LSTM-AE --dataset BMSD --seed 42

This script trains a single method on a single dataset and saves the
fitted detector to disk. For full multi-seed experiments, use run_experiment.py.
"""

import os
import sys
import argparse
import pickle

# Ensure the parent directory (code/) is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import Config
from models.feature_extractor import FeatureExtractor
from models.homotopy_rl import HRLADDetector
from models.baselines import BASELINES
from data.data_loader import generate_synthetic_ce_data


def main():
    parser = argparse.ArgumentParser(description="Train HRLAD or a baseline detector")
    parser.add_argument("--method", type=str, default="HRLAD",
                        help="Method name (HRLAD, LSTM-AE, TranAD, etc.)")
    parser.add_argument("--dataset", type=str, default="SPSD",
                        choices=["SPSD", "HPMD", "BMSD"],
                        help="Dataset name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-samples", type=int, default=5000,
                        help="Number of synthetic samples to generate")
    parser.add_argument("--anomaly-ratio", type=float, default=0.05,
                        help="Anomaly ratio in generated data")
    parser.add_argument("--output-dir", type=str, default="checkpoints/",
                        help="Directory to save the trained model")
    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    # --- Load data ---
    dataset_map = {"SPSD": "smartphone", "HPMD": "appliance", "BMSD": "battery"}
    values, labels = generate_synthetic_ce_data(
        dataset_type=dataset_map[args.dataset],
        n_samples=args.n_samples,
        anomaly_ratio=args.anomaly_ratio,
        seed=args.seed,
    )

    # Chronological split: 70% train
    config = Config()
    train_end = int(len(values) * config.train_ratio)

    print(f"[INFO] Dataset: {args.dataset} | Method: {args.method} | Seed: {args.seed}")
    print(f"[INFO] Total samples: {len(values)} | Train: {train_end}")

    # --- Train ---
    if args.method == "HRLAD":
        # Feature extraction
        fe = FeatureExtractor(
            window_size=config.window_size,
            n_fft_components=config.n_fft_components,
        )
        fe.fit(values[:train_end])
        features, _ = fe.transform(values)
        train_labels = labels[: len(features)]

        # Fit HRLAD detector
        detector = HRLADDetector(config=config)
        detector.fit(features[: len(train_labels)], train_labels, verbose=True)
        model_obj = {"detector": detector, "feature_extractor": fe, "config": config}

    elif args.method in BASELINES:
        detector = BASELINES[args.method]()
        detector.fit_predict(values[:train_end], labels[:train_end])
        model_obj = {"detector": detector, "config": config}

    else:
        raise ValueError(f"Unknown method: {args.method}. "
                         f"Available: HRLAD, {list(BASELINES.keys())}")

    # --- Save ---
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(
        args.output_dir,
        f"{args.method.lower().replace(' ', '_')}_{args.dataset.lower()}_seed{args.seed}.pkl",
    )
    with open(model_path, "wb") as f:
        pickle.dump(model_obj, f)

    print(f"[OK] Model saved to {model_path}")


if __name__ == "__main__":
    main()
