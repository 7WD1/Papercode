"""Evaluation Metrics for HRLAD"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def compute_metrics(y_true, y_pred, y_score=None):
    """Compute comprehensive anomaly detection metrics.

    Args:
        y_true: ground truth labels (0=normal, 1=anomaly)
        y_pred: predicted labels
        y_score: anomaly scores (optional, for AUC)
    Returns:
        dict of metrics
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Handle case where no anomalies predicted or no true anomalies
    tn, fp, fn, tp = 0, 0, 0, 0
    if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            tn = np.sum((y_pred == 0) & (y_true == 0))
    else:
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'fpr': fpr,
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
    }

    # AUC-ROC if scores provided
    if y_score is not None:
        y_score = np.asarray(y_score).flatten()
        if len(np.unique(y_true)) > 1:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_score)
            except ValueError:
                metrics['auc_roc'] = 0.0
        else:
            metrics['auc_roc'] = 0.0

    return metrics


def compute_detection_delay(y_true, y_pred):
    """Compute average detection delay in time steps.

    For each anomaly event (consecutive anomaly period), measure how many
    steps after the start the first correct detection occurs.

    Args:
        y_true: ground truth labels
        y_pred: predicted labels
    Returns:
        average detection delay (0 if detected at first step)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Find anomaly events (start and end indices)
    anomaly_starts = []
    in_anomaly = False
    for i in range(len(y_true)):
        if y_true[i] == 1 and not in_anomaly:
            anomaly_starts.append(i)
            in_anomaly = True
        elif y_true[i] == 0:
            in_anomaly = False

    if not anomaly_starts:
        return 0.0

    delays = []
    for start in anomaly_starts:
        # Find end of this anomaly event
        end = start
        while end < len(y_true) and y_true[end] == 1:
            end += 1

        # Find first detection within this event
        detected = False
        for t in range(start, min(end, len(y_pred))):
            if y_pred[t] == 1:
                delays.append(t - start)
                detected = True
                break

        if not detected:
            delays.append(end - start)  # missed detection -> max delay

    return np.mean(delays) if delays else 0.0


def print_metrics(metrics, dataset_name=''):
    """Pretty print metrics."""
    header = f" {dataset_name} " if dataset_name else ""
    print(f"\n{'='*50}")
    print(f"  Evaluation Results{header}")
    print(f"{'='*50}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1 Score:     {metrics['f1_score']:.4f}")
    print(f"  FPR:          {metrics['fpr']:.4f}")
    if 'auc_roc' in metrics:
        print(f"  AUC-ROC:      {metrics['auc_roc']:.4f}")
    if 'detection_delay' in metrics:
        print(f"  Det. Delay:   {metrics['detection_delay']:.1f} steps")
    print(f"  TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}, TN={metrics['tn']}")
    print(f"{'='*50}\n")
