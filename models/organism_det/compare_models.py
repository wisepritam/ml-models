#!/usr/bin/env python3
"""
Compare YOLO26n / YOLO26s / YOLO26m benchmark results.

Run from models/organism_det/:
    python compare_models.py

Reads:    results/metrics/{model}_metrics.csv  (produced by training notebooks)
Writes:   results/plots/benchmark_*.png
          results/reports/benchmark_summary.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT    = Path(__file__).parent
METRICS = ROOT / 'results/metrics'
PLOTS   = ROOT / 'results/plots'
REPORTS = ROOT / 'results/reports'

MODELS = ['yolo26n', 'yolo26s', 'yolo26m']

CLASS_NAMES = [
    'Trichomonas vaginalis',
    'Bacterial vaginosis flora shift',
    'Candida spp.',
    'Actinomyces spp.',
]
CLASS_SHORT  = ['TV', 'BV', 'Candida', 'Actinomyces']
AP_COLS      = [f'AP50_{n.replace(" ", "_")}' for n in CLASS_NAMES]

COLORS      = {'yolo26n': '#3498db', 'yolo26s': '#2ecc71', 'yolo26m': '#e74c3c'}
LATENCY_MS  = {'yolo26n': 1.7, 'yolo26s': 2.5, 'yolo26m': 4.7}
PARAMS_M    = {'yolo26n': 2.4, 'yolo26s': 9.5, 'yolo26m': 20.4}


def load_metrics() -> pd.DataFrame:
    frames = []
    for model in MODELS:
        path = METRICS / f'{model}_metrics.csv'
        if not path.exists():
            print(f'  [skip] missing {path.name}')
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(
            'No metrics CSVs found in results/metrics/. Run the training notebooks first.'
        )
    return pd.concat(frames, ignore_index=True)


def _test_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.split == 'test'].set_index('model')


def plot_map_comparison(df: pd.DataFrame) -> None:
    tdf = _test_df(df)
    present = [m for m in MODELS if m in tdf.index]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, label in [
        (axes[0], 'mAP50',    'mAP@50'),
        (axes[1], 'mAP50_95', 'mAP@50-95'),
    ]:
        vals   = [tdf.loc[m, metric] for m in present]
        colors = [COLORS[m] for m in present]
        bars   = ax.bar(present, vals, color=colors, width=0.45, edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold'
            )
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'Test {label}', fontsize=13, fontweight='bold')
        ax.set_ylim(0, min(1.0, max(vals) * 1.25))
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    fig.suptitle('YOLO26 Variant Comparison — Test Set', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = PLOTS / 'benchmark_mAP_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved → {out}')


def plot_speed_accuracy(df: pd.DataFrame) -> None:
    tdf = _test_df(df)
    fig, ax = plt.subplots(figsize=(8, 6))

    for model in MODELS:
        if model not in tdf.index:
            continue
        x = LATENCY_MS[model]
        y = tdf.loc[model, 'mAP50']
        ax.scatter(x, y, color=COLORS[model], s=250, zorder=3, edgecolors='white', linewidths=1.5)
        ax.annotate(f'  {model}', (x, y), fontsize=11, va='center', fontweight='bold',
                    color=COLORS[model])

    ax.set_xlabel('T4 TensorRT Latency (ms)', fontsize=12)
    ax.set_ylabel('mAP@50 (test)', fontsize=12)
    ax.set_title('Speed vs. Accuracy', fontsize=13, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, alpha=0.25, linestyle='--')
    plt.tight_layout()
    out = PLOTS / 'benchmark_speed_accuracy.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved → {out}')


def plot_per_class_heatmap(df: pd.DataFrame) -> None:
    tdf     = _test_df(df)
    present = [m for m in MODELS if m in tdf.index]
    matrix  = np.zeros((len(present), len(CLASS_NAMES)))

    for i, model in enumerate(present):
        for j, col in enumerate(AP_COLS):
            val = tdf.loc[model, col] if col in tdf.columns else 0.0
            matrix[i, j] = val if pd.notna(val) else 0.0

    fig, ax = plt.subplots(figsize=(11, len(present) * 1.4 + 1.5))
    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, label='AP@50', fraction=0.03)

    ax.set_xticks(range(len(CLASS_SHORT)))
    ax.set_xticklabels(CLASS_SHORT, fontsize=12)
    ax.set_yticks(range(len(present)))
    ax.set_yticklabels(present, fontsize=12)
    ax.set_title('Per-Class AP@50 — Test Set', fontsize=13, fontweight='bold')

    for i in range(len(present)):
        for j in range(len(CLASS_NAMES)):
            val   = matrix[i, j]
            color = 'white' if val < 0.45 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=11, color=color, fontweight='bold')

    plt.tight_layout()
    out = PLOTS / 'benchmark_per_class_heatmap.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved → {out}')


def write_summary(df: pd.DataFrame) -> None:
    tdf     = _test_df(df)
    present = [m for m in MODELS if m in tdf.index]

    best_model = max(present, key=lambda m: tdf.loc[m, 'mAP50'])
    best_tv    = max(present, key=lambda m: tdf.loc[m, AP_COLS[0]] if AP_COLS[0] in tdf.columns else 0)

    lines = [
        '# YOLO26 Organism Detection — Benchmark Summary\n',
        '\n',
        '## Test Set — Overall Metrics\n',
        '\n',
        '| Model | mAP@50 | mAP@50-95 | Precision | Recall | Latency (ms) | Params (M) |\n',
        '|---|---|---|---|---|---|---|\n',
    ]
    for m in present:
        r = tdf.loc[m]
        lines.append(
            f'| {m} | {r.mAP50:.4f} | {r.mAP50_95:.4f} | {r.precision:.4f} '
            f'| {r.recall:.4f} | {LATENCY_MS[m]} | {PARAMS_M[m]} |\n'
        )

    lines += [
        '\n',
        '## Test Set — Per-Class AP@50\n',
        '\n',
        '| Model | ' + ' | '.join(CLASS_SHORT) + ' |\n',
        '|' + '|'.join(['---'] * (len(CLASS_SHORT) + 1)) + '|\n',
    ]
    for m in present:
        r    = tdf.loc[m]
        vals = [f'{r[c]:.4f}' if c in tdf.columns and pd.notna(r[c]) else 'N/A' for c in AP_COLS]
        lines.append(f'| {m} | ' + ' | '.join(vals) + ' |\n')

    lines += [
        '\n',
        '## Verdict\n',
        '\n',
        f'**Best overall mAP@50:** `{best_model}` ({tdf.loc[best_model, "mAP50"]:.4f})\n',
        f'**Best TV (Trichomonas) recall:** `{best_tv}`\n',
        '\n',
        '### Deployment decision\n',
        '\n',
        '- Deploy **`yolo26s`** unless:\n',
        '  - TV AP@50 gap between `yolo26s` and `yolo26m` > 0.03 → deploy `m`\n',
        '  - Latency budget < 2ms → deploy `n` (expect lower TV recall)\n',
        '\n',
        'Trichomonas vaginalis AP@50 is the primary clinical metric — not overall mAP.\n',
        '\n',
        '### Output files\n',
        '\n',
        '```\n',
        'results/plots/benchmark_mAP_comparison.png\n',
        'results/plots/benchmark_speed_accuracy.png\n',
        'results/plots/benchmark_per_class_heatmap.png\n',
        'results/plots/{model}/confusion_matrix.png\n',
        'results/plots/{model}/PR_curve.png\n',
        'results/plots/{model}/sample_predictions.png\n',
        '```\n',
    ]

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / 'benchmark_summary.md'
    out.write_text(''.join(lines))
    print(f'Saved → {out}')


def main() -> None:
    print('Loading metrics CSVs...')
    df = load_metrics()
    present = df.model.nunique()
    print(f'Loaded {len(df)} rows from {present} model(s): {df.model.unique().tolist()}\n')

    PLOTS.mkdir(parents=True, exist_ok=True)

    print('Generating plots...')
    plot_map_comparison(df)
    plot_speed_accuracy(df)
    plot_per_class_heatmap(df)

    print('\nWriting benchmark summary...')
    write_summary(df)

    print(f'\nDone.')
    print(f'  Plots   → {PLOTS}')
    print(f'  Reports → {REPORTS}')


if __name__ == '__main__':
    main()
