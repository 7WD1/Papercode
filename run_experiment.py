"""HRLAD Multi-Seed Experiment Runner

Evaluates HRLAD and baselines across multiple seeds following the paper protocol:
- 10 independent runs (different seeds), reporting mean +/- std
- 70:15:15 data splits (train/val/test)
- AUROC, FAR (FPR), DD metrics
- Paired t-test, Wilcoxon signed-rank test, Cohen's d effect size
"""

import sys
import os
import time
import json
import argparse
import numpy as np
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

# Graceful torch handling
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from scipy import stats as sp_stats

from config import Config
from models.feature_extractor import FeatureExtractor
from models.homotopy_rl import HRLADDetector
from models.baselines import BASELINES
from utils.metrics import compute_metrics, compute_detection_delay
from data.data_loader import get_all_datasets


# ---------------------------------------------------------------------------
# Statistical test helpers
# ---------------------------------------------------------------------------

def cohens_d(x, y):
    """Compute Cohen's d effect size between two samples."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return 0.0
    pooled_std = np.sqrt(
        ((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2)
        / (nx + ny - 2)
    )
    if pooled_std < 1e-12:
        return 0.0
    return (np.mean(x) - np.mean(y)) / pooled_std


def statistical_tests(hrlad_scores, baseline_scores):
    """Run paired t-test, Wilcoxon signed-rank test, and Cohen's d."""
    result = {
        't_stat': 0.0, 'p_ttest': 1.0,
        'w_stat': 0.0, 'p_wilcoxon': 1.0,
        'cohens_d': 0.0,
    }
    hrlad_scores = np.asarray(hrlad_scores, dtype=float)
    baseline_scores = np.asarray(baseline_scores, dtype=float)
    if len(hrlad_scores) < 2 or len(baseline_scores) < 2:
        return result
    diff = hrlad_scores - baseline_scores
    if np.all(diff == 0):
        return result
    try:
        t_stat, p_ttest = sp_stats.ttest_rel(hrlad_scores, baseline_scores)
        result['t_stat'] = float(t_stat)
        result['p_ttest'] = float(p_ttest)
    except Exception:
        pass
    try:
        w_stat, p_wilcoxon = sp_stats.wilcoxon(hrlad_scores, baseline_scores)
        result['w_stat'] = float(w_stat)
        result['p_wilcoxon'] = float(p_wilcoxon)
    except Exception:
        pass
    result['cohens_d'] = float(cohens_d(hrlad_scores, baseline_scores))
    return result


# ---------------------------------------------------------------------------
# Seed setting
# ---------------------------------------------------------------------------

def set_seed(seed):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Single-method runner for one (dataset, seed) pair
# ---------------------------------------------------------------------------

def run_method(method_name, values, labels, config, train_end, val_end):
    """Run a single method and return metrics dict."""
    start_t = time.time()
    try:
        n = len(values)
        if method_name == 'HRLAD':
            ext = FeatureExtractor(config.window_size, config.n_fft_components)
            ext.fit(values[:train_end])
            train_feat, _ = ext.transform(values[:train_end])
            train_lab = labels[config.window_size - 1:train_end]
            ml = min(len(train_feat), len(train_lab))
            train_feat = train_feat[:ml]
            train_lab = train_lab[:ml]
            det = HRLADDetector(config)
            det.fit(train_feat, train_lab, verbose=False)
            test_feat, test_idx = ext.transform(values[val_end:])
            test_lab_arr = np.array([
                labels[i] if 0 <= i < len(labels) else 0
                for i in test_idx
            ])[:len(test_feat)]
            predictions, scores = det.predict(test_feat)
            eval_len = min(len(predictions), len(test_lab_arr))
            predictions = predictions[:eval_len]
            scores = scores[:eval_len]
            eval_labels = test_lab_arr[:eval_len]

        elif method_name == 'RL-AD':
            ext = FeatureExtractor(config.window_size, config.n_fft_components)
            ext.fit(values[:train_end])
            all_feat, all_idx = ext.transform(values)
            all_lab = labels[all_idx]
            rlad = BASELINES[method_name](window_size=config.window_size)
            predictions, scores = rlad.fit_predict(all_feat, all_lab)
            mask = all_idx >= val_end
            predictions = predictions[mask]
            scores = scores[mask]
            eval_labels = all_lab[mask]
            eval_len = min(len(predictions), len(eval_labels))
            predictions = predictions[:eval_len]
            scores = scores[:eval_len]
            eval_labels = eval_labels[:eval_len]

        else:
            bl = BASELINES[method_name]()
            predictions, scores = bl.fit_predict(values, labels)
            predictions = predictions[val_end:]
            scores = scores[val_end:] if len(scores) > val_end else scores
            eval_labels = labels[val_end:]
            eval_len = min(len(predictions), len(eval_labels))
            predictions = predictions[:eval_len]
            scores = scores[:eval_len] if len(scores) > 0 else np.zeros(eval_len)
            eval_labels = eval_labels[:eval_len]

        elapsed = time.time() - start_t
        metrics = compute_metrics(eval_labels, predictions, scores)
        metrics['detection_delay'] = compute_detection_delay(eval_labels, predictions)
        metrics['time'] = elapsed

    except Exception as e:
        print(f'  Error: {e}')
        import traceback
        traceback.print_exc()
        metrics = {
            'f1_score': 0.0, 'precision': 0.0, 'recall': 0.0,
            'fpr': 0.0, 'auc_roc': 0.0, 'detection_delay': 0.0,
            'time': 0.0, 'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0,
        }

    return metrics


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='HRLAD Multi-Seed Experiment')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                        help='List of random seeds (default: config.seeds)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Datasets to run (default: all)')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        help='Methods to run (default: all including HRLAD)')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory (default: results)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode: 1 seed for testing')
    args = parser.parse_args()

    config = Config()

    # Determine seeds
    if args.quick:
        seeds = [42]
    elif args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = config.seeds

    # Determine methods
    all_methods = ['HRLAD'] + list(BASELINES.keys())
    if args.methods is not None:
        methods = args.methods
        # Validate
        for m in methods:
            if m not in all_methods:
                print(f'Warning: unknown method "{m}", skipping.')
        methods = [m for m in methods if m in all_methods]
    else:
        methods = all_methods

    # Load datasets
    print('=' * 70)
    print('  HRLAD Multi-Seed Experiment')
    print('=' * 70)
    print(f'  Seeds:    {seeds}')
    print(f'  Methods:  {methods}')
    print(f'  Splits:   {config.train_ratio}/{config.val_ratio}/{config.test_ratio}')
    print(f'  Torch:    {"available" if HAS_TORCH else "not available (using numpy fallbacks)"}')
    print('=' * 70)

    print('\nLoading datasets...')
    all_datasets = get_all_datasets()

    # Filter datasets if requested
    if args.datasets is not None:
        filtered = {}
        for name in args.datasets:
            matching = [k for k in all_datasets if name.lower() in k.lower()]
            if matching:
                for k in matching:
                    filtered[k] = all_datasets[k]
            else:
                print(f'  Warning: no dataset matching "{name}" found.')
        all_datasets = filtered

    dataset_names = list(all_datasets.keys())
    print(f'Loaded {len(dataset_names)} datasets: {dataset_names}')

    # -----------------------------------------------------------------------
    # Multi-seed evaluation loop
    # -----------------------------------------------------------------------
    # per_seed[seed][(ds_name, method_name)] = metrics_dict
    per_seed = {}

    for seed in seeds:
        print(f'\n{"=" * 60}')
        print(f'  Seed {seed}')
        print(f'{"=" * 60}')
        set_seed(seed)

        seed_results = {}
        for ds_name, ds_data in all_datasets.items():
            values, labels = ds_data
            print(f'\n  Dataset: {ds_name} (n={len(values)}, '
                  f'anomaly_rate={labels.mean():.3f})')

            n = len(values)
            train_end = int(n * config.train_ratio)
            val_end = int(n * (config.train_ratio + config.val_ratio))

            for method_name in methods:
                # Re-seed before each method for strict reproducibility
                set_seed(seed)

                print(f'    {method_name:<22s}', end='', flush=True)
                start_t = time.time()
                metrics = run_method(method_name, values, labels,
                                     config, train_end, val_end)
                elapsed = time.time() - start_t
                seed_results[(ds_name, method_name)] = metrics
                print(f' F1={metrics["f1_score"]:.4f}  '
                      f'AUC={metrics.get("auc_roc", 0.0):.4f}  '
                      f'({elapsed:.1f}s)')

        per_seed[seed] = seed_results

    # -----------------------------------------------------------------------
    # Aggregate: mean +/- std across seeds
    # -----------------------------------------------------------------------
    metric_keys = ['f1_score', 'precision', 'recall', 'fpr',
                   'auc_roc', 'detection_delay', 'time',
                   'tp', 'fp', 'fn', 'tn']

    # Collect per-seed values keyed by (ds, method) -> list of metric values
    aggregated = {}  # (ds, method) -> {metric_mean, metric_std}
    for ds_name in dataset_names:
        for method_name in methods:
            key = (ds_name, method_name)
            agg = {}
            for mk in metric_keys:
                vals = []
                for seed in seeds:
                    m = per_seed[seed].get(key, {})
                    v = m.get(mk, 0.0)
                    vals.append(float(v))
                agg[f'{mk}_mean'] = float(np.mean(vals))
                agg[f'{mk}_std'] = float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                agg[f'{mk}_values'] = vals
            aggregated[key] = agg

    # -----------------------------------------------------------------------
    # Statistical tests: HRLAD vs each baseline (paired across seeds)
    # -----------------------------------------------------------------------
    stat_tests = {}  # (ds, baseline) -> test_results
    if 'HRLAD' in methods:
        for ds_name in dataset_names:
            for method_name in methods:
                if method_name == 'HRLAD':
                    continue
                hrlad_f1 = aggregated[(ds_name, 'HRLAD')]['f1_score_values']
                bl_f1 = aggregated[(ds_name, method_name)]['f1_score_values']
                stat_tests[(ds_name, method_name)] = statistical_tests(
                    hrlad_f1, bl_f1
                )

    # -----------------------------------------------------------------------
    # Print results table
    # -----------------------------------------------------------------------
    print('\n' + '=' * 120)
    print('  FINAL RESULTS  (mean +/- std across seeds)')
    print('=' * 120)
    print(f'{"Method":<22s} {"Dataset":<28s} | '
          f'{"F1":>14s} {"Prec":>14s} {"Rec":>14s} '
          f'{"FPR":>14s} {"AUC":>14s} {"DD":>14s} | '
          f'{"p(t)":>8s} {"p(W)":>8s} {"d":>6s} |')
    print('-' * 120)

    for ds_name in dataset_names:
        for method_name in methods:
            agg = aggregated[(ds_name, method_name)]
            f1_s = f'{agg["f1_score_mean"]:.4f}+/-{agg["f1_score_std"]:.4f}'
            prec_s = f'{agg["precision_mean"]:.4f}+/-{agg["precision_std"]:.4f}'
            rec_s = f'{agg["recall_mean"]:.4f}+/-{agg["recall_std"]:.4f}'
            fpr_s = f'{agg["fpr_mean"]:.4f}+/-{agg["fpr_std"]:.4f}'
            auc_s = f'{agg["auc_roc_mean"]:.4f}+/-{agg["auc_roc_std"]:.4f}'
            dd_s = f'{agg["detection_delay_mean"]:.1f}+/-{agg["detection_delay_std"]:.1f}'

            # Statistical test columns
            if method_name == 'HRLAD':
                pt_s, pw_s, d_s = '', '', ''
            else:
                st = stat_tests.get((ds_name, method_name), {})
                pt_s = f'{st.get("p_ttest", 1.0):.4f}'
                pw_s = f'{st.get("p_wilcoxon", 1.0):.4f}'
                d_s = f'{st.get("cohens_d", 0.0):.2f}'

            print(f'{method_name:<22s} {ds_name:<28s} | '
                  f'{f1_s:>14s} {prec_s:>14s} {rec_s:>14s} '
                  f'{fpr_s:>14s} {auc_s:>14s} {dd_s:>14s} | '
                  f'{pt_s:>8s} {pw_s:>8s} {d_s:>6s} |')
        print('-' * 120)

    # -----------------------------------------------------------------------
    # Averaged across all datasets (summary row per method)
    # -----------------------------------------------------------------------
    print('\n  Averaged across all datasets:')
    print(f'{"Method":<22s} | '
          f'{"F1":>14s} {"Prec":>14s} {"Rec":>14s} '
          f'{"FPR":>14s} {"AUC":>14s} {"DD":>14s} | '
          f'{"p(t)":>8s} {"p(W)":>8s} {"d":>6s} |')
    print('-' * 110)

    overall_agg = {}
    for method_name in methods:
        ov = {}
        for mk in ['f1_score', 'precision', 'recall', 'fpr',
                    'auc_roc', 'detection_delay']:
            # Average across datasets for each seed, then mean/std
            per_seed_avg = []
            for seed in seeds:
                ds_vals = []
                for ds_name in dataset_names:
                    m = per_seed[seed].get((ds_name, method_name), {})
                    ds_vals.append(float(m.get(mk, 0.0)))
                per_seed_avg.append(float(np.mean(ds_vals)))
            ov[f'{mk}_mean'] = float(np.mean(per_seed_avg))
            ov[f'{mk}_std'] = float(np.std(per_seed_avg, ddof=1)
                                     if len(per_seed_avg) > 1 else 0.0)
            ov[f'{mk}_values'] = per_seed_avg
        overall_agg[method_name] = ov

    for method_name in methods:
        agg = overall_agg[method_name]
        f1_s = f'{agg["f1_score_mean"]:.4f}+/-{agg["f1_score_std"]:.4f}'
        prec_s = f'{agg["precision_mean"]:.4f}+/-{agg["precision_std"]:.4f}'
        rec_s = f'{agg["recall_mean"]:.4f}+/-{agg["recall_std"]:.4f}'
        fpr_s = f'{agg["fpr_mean"]:.4f}+/-{agg["fpr_std"]:.4f}'
        auc_s = f'{agg["auc_roc_mean"]:.4f}+/-{agg["auc_roc_std"]:.4f}'
        dd_s = f'{agg["detection_delay_mean"]:.1f}+/-{agg["detection_delay_std"]:.1f}'

        if method_name == 'HRLAD' or 'HRLAD' not in methods:
            pt_s, pw_s, d_s = '', '', ''
        else:
            st = statistical_tests(
                overall_agg['HRLAD']['f1_score_values'],
                agg['f1_score_values'],
            )
            pt_s = f'{st["p_ttest"]:.4f}'
            pw_s = f'{st["p_wilcoxon"]:.4f}'
            d_s = f'{st["cohens_d"]:.2f}'

        print(f'{method_name:<22s} | '
              f'{f1_s:>14s} {prec_s:>14s} {rec_s:>14s} '
              f'{fpr_s:>14s} {auc_s:>14s} {dd_s:>14s} | '
              f'{pt_s:>8s} {pw_s:>8s} {d_s:>6s} |')

    # -----------------------------------------------------------------------
    # Save results to JSON
    # -----------------------------------------------------------------------
    results_dir = args.output_dir
    os.makedirs(results_dir, exist_ok=True)

    # Build serializable output
    output = {
        'per_seed': {},
        'summary': {},
        'statistical_tests': {},
        'config': {
            'seeds': seeds,
            'methods': methods,
            'datasets': dataset_names,
            'train_ratio': config.train_ratio,
            'val_ratio': config.val_ratio,
            'test_ratio': config.test_ratio,
            'window_size': config.window_size,
            'n_fft_components': config.n_fft_components,
            'n_modes': config.n_modes,
            'gamma': config.gamma,
            'lambda_max': config.lambda_max,
            'alarm_threshold': config.alarm_threshold,
            'has_torch': HAS_TORCH,
        },
    }

    # Per-seed results
    for seed in seeds:
        for ds_name in dataset_names:
            for method_name in methods:
                key_str = f'{seed}_{method_name}_{ds_name}'
                m = per_seed[seed].get((ds_name, method_name), {})
                output['per_seed'][key_str] = {
                    k: float(v) if isinstance(v, (np.floating, float)) else v
                    for k, v in m.items()
                }

    # Summary (mean +/- std)
    for ds_name in dataset_names:
        for method_name in methods:
            key_str = f'{method_name}_{ds_name}'
            agg = aggregated[(ds_name, method_name)]
            summary_entry = {}
            for mk in metric_keys:
                summary_entry[f'{mk}_mean'] = agg[f'{mk}_mean']
                summary_entry[f'{mk}_std'] = agg[f'{mk}_std']
            output['summary'][key_str] = summary_entry

    # Statistical tests
    if 'HRLAD' in methods:
        for ds_name in dataset_names:
            for method_name in methods:
                if method_name == 'HRLAD':
                    continue
                key_str = f'vs_{method_name}_{ds_name}'
                st = stat_tests.get((ds_name, method_name), {})
                output['statistical_tests'][key_str] = st

    results_path = os.path.join(results_dir, 'experiment_results.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nResults saved to {results_path}')

    # Also save a compact CSV-style table for easy import
    csv_path = os.path.join(results_dir, 'experiment_summary.csv')
    with open(csv_path, 'w') as f:
        header = 'method,dataset'
        for mk in ['f1_score', 'precision', 'recall', 'fpr',
                    'auc_roc', 'detection_delay']:
            header += f',{mk}_mean,{mk}_std'
        f.write(header + '\n')
        for ds_name in dataset_names:
            for method_name in methods:
                agg = aggregated[(ds_name, method_name)]
                row = f'{method_name},{ds_name}'
                for mk in ['f1_score', 'precision', 'recall', 'fpr',
                            'auc_roc', 'detection_delay']:
                    row += f',{agg[f"{mk}_mean"]:.6f},{agg[f"{mk}_std"]:.6f}'
                f.write(row + '\n')
    print(f'Summary CSV saved to {csv_path}')

    print('\nDone!')


if __name__ == '__main__':
    main()
