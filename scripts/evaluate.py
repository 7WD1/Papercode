#!/usr/bin/env python3
"""
Standalone evaluation entry point for HRLAD and baselines.

Usage:
    python scripts/evaluate.py --method HRLAD --dataset BMSD \
        --checkpoint checkpoints/hrlad_bmsd_seed42.pkl
"""

import os
import sys
import argparse
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import Config
from data.data_loader import generate_synthetic_ce_data
from utils.metrics import compute_metrics, compute_detection_delay, print_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained detector")
    parser.add_argument("--method", type=str, default="HRLAD")
    parser.add_argument("--dataset", type=str, default="SPSD",
                        choices=["SPSD", "HPMD", "BMSD"])
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the saved model pickle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--anomaly-ratio", type=float, default=0.05)
    args = parser.parse_args()

    np.random.seed(args.seed)

    # --- Load data ---
    dataset_map = {"SPSD": "smartphone", "HPMD": "appliance", "BMSD": "battery"}
    values, labels = generate_synthetic_ce_data(
        dataset_type=dataset_map[args.dataset],
        n_samples=args.n_samples,
        anomaly_ratio=args.anomaly_ratio,
        seed=args.seed,
    )

    config = Config()
    train_end = int(len(values) * config.train_ratio)
    val_end = int(len(values) * (config.train_ratio + config.val_ratio))

    # --- Load model ---
    with open(args.checkpoint, "rb") as f:
        model_obj = pickle.load(f)

    print(f"[INFO] Evaluating {args.method} on {args.dataset} (seed {args.seed})")

    # --- Predict on test set ---
    if args.method == "HRLAD":
        detector = model_obj["detector"]
        fe = model_obj["feature_extractor"]
        features, _ = fe.transform(values)
        test_features = features[val_end:]
        test_labels = labels[val_end: len(test_features) + val_end]

        predictions, scores = detector.predict(test_features)
        metrics = compute_metrics(test_labels, predictions, scores)
        metrics["detection_delay"] = compute_detection_delay(test_labels, predictions)

    elif "detector" in model_obj:
        detector = model_obj["detector"]
        test_values = values[val_end:]
        test_labels = labels[val_end:]
        # Baselines use fit_predict interface; re-fit on train then predict test
        detector.fit_predict(values[:train_end], labels[:train_end])
        result = detector.fit_predict(test_values, test_labels)
        # fit_predict returns predictions array (1-D); unpack safely
        preds = result[0] if isinstance(result, tuple) else result
        scores = result[1] if isinstance(result, tuple) and len(result) > 1 else None
        metrics = compute_metrics(test_labels, preds, scores)
        metrics["detection_delay"] = compute_detection_delay(test_labels, preds)
    else:
        raise ValueError("Invalid checkpoint format")

    # --- Report ---
    print_metrics(metrics, dataset_name=args.dataset)


if __name__ == "__main__":
    main()
