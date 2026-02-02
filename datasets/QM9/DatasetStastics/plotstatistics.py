#!/usr/bin/env python3
"""
Extensive QM9 statistics plotting

Usage:
    python plot_qm9_all_stats.py /path/to/qm9_full_stats.npz --out_dir run_stats --prefix qm9_full

Accepts either:
 - NPZ created by compute_qm9_stats (recommended, contains raw data)
 - JSON created by compute_qm9_stats (will attempt to find corresponding NPZ in same folder)

Produces many PNGs into out_dir/prefix_plots/
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

PROPERTY_NAMES = ['α', 'Δε', 'ε_HOMO', 'ε_LUMO', 'μ', 'C_v',
                  'G', 'H', 'R²', 'U', 'U₀', 'ZPVE']

def load_stats(npz_or_json_path):
    """
    Loads raw data (n_samples, 12) and valid_mask.
    Accepts .npz or .json path. If json is given, will search for <prefix>_stats.npz
    in the same directory.
    Returns (data, valid_mask, names).
    """
    path = npz_or_json_path
    if path.endswith('.json'):
        j = json.load(open(path, 'r'))
        # try to find an npz alongside json with same prefix
        base = os.path.splitext(os.path.basename(path))[0]
        d = os.path.dirname(path) or '.'
        candidates = [os.path.join(d, f"{base}.npz"),
                      os.path.join(d, f"{base.replace('_stats','')}_stats.npz"),
                      os.path.join(d, "qm9_stats.npz")]
        npz_path = None
        for c in candidates:
            if os.path.exists(c):
                npz_path = c
                break
        if npz_path is None:
            raise FileNotFoundError("Could not find corresponding .npz for JSON. "
                                    "Please provide the .npz produced by compute_qm9_stats.")
        path = npz_path

    if not path.endswith('.npz'):
        raise ValueError("Please provide a .npz (preferred) or .json pointing to it.")

    npz = np.load(path, allow_pickle=True)
    # expected keys: data, valid_mask, property_names (maybe)
    if 'data' in npz:
        data = npz['data']
    else:
        # backward compat: try 'arr_0'
        data = npz[npz.files[0]]
    if 'valid_mask' in npz:
        valid_mask = npz['valid_mask']
    else:
        valid_mask = ~np.isnan(data).any(axis=1)

    if 'property_names' in npz:
        names = [str(x) for x in npz['property_names']]
    else:
        names = PROPERTY_NAMES

    return data, valid_mask.astype(bool), names

# ---- Extra stats helpers ----
def extended_stats(col):
    """Compute extra stats for 1D numpy column (ignores NaNs)."""
    col = col[~np.isnan(col)]
    n = col.size
    if n == 0:
        return {}
    mean = float(np.mean(col))
    std = float(np.std(col, ddof=1)) if n > 1 else 0.0
    med = float(np.median(col))
    mn = float(np.min(col))
    mx = float(np.max(col))
    p25 = float(np.percentile(col, 25.0))
    p75 = float(np.percentile(col, 75.0))
    # skewness & kurtosis (Fisher definition: kurtosis - 3)
    if std == 0 or n < 3:
        skew = 0.0
        kurt = -3.0  # degenerate
    else:
        skew = float(np.mean(((col - mean) / std) ** 3))
        kurt = float(np.mean(((col - mean) / std) ** 4)) - 3.0
    zeros = int(np.sum(col == 0))
    nan_count = int(np.sum(np.isnan(col)))
    return {
        'count': int(n),
        'mean': mean,
        'std': std,
        'min': mn,
        'max': mx,
        'median': med,
        'p25': p25,
        'p75': p75,
        'skewness': skew,
        'kurtosis': kurt,
        'zeros': zeros,
        'nan_count': nan_count
    }

# ---- Plotting routines ----
def plot_mean_std(means, stds, names, outpath):
    x = np.arange(len(names))
    plt.figure(figsize=(12,5))
    plt.errorbar(x, means, yerr=stds, fmt='o', capsize=5, markersize=6)
    plt.xticks(x, names, rotation=45, ha='right')
    plt.title('QM9 target means ± std (paper units)')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_median_iqr(medians, p25, p75, names, outpath):
    x = np.arange(len(names))
    plt.figure(figsize=(12,6))
    for i in range(len(names)):
        plt.plot([i, i], [p25[i], p75[i]], linewidth=6)
        plt.plot(i, medians[i], marker='o', markersize=6, markeredgecolor='k')
    plt.xticks(x, names, rotation=45, ha='right')
    plt.title('QM9 target median and IQR (p25-p75)')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_boxplot_all(data, valid_mask, names, outpath):
    # build list of columns (valid only)
    cols = [data[valid_mask, j] for j in range(data.shape[1])]
    # remove NaNs per col
    cols_clean = [col[~np.isnan(col)] for col in cols]
    plt.figure(figsize=(12,6))
    plt.boxplot(cols_clean, labels=names, showfliers=True)
    plt.xticks(rotation=45, ha='right')
    plt.title('QM9 property boxplots')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_histograms(data, valid_mask, names, outdir, bins=100, sample_limit=100000):
    os.makedirs(outdir, exist_ok=True)
    n_props = data.shape[1]
    for j in range(n_props):
        col = data[valid_mask, j]
        col = col[~np.isnan(col)]
        if col.size == 0:
            continue
        # sample if huge
        if col.size > sample_limit:
            idx = np.random.choice(col.size, sample_limit, replace=False)
            col_s = col[idx]
        else:
            col_s = col
        plt.figure(figsize=(8,4))
        plt.hist(col_s, bins=bins)
        plt.title(f'Histogram: {names[j]}')
        plt.xlabel('value')
        plt.ylabel('count')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"hist_{j}_{names[j]}.png"), dpi=200)
        plt.close()

        # log-y histogram
        plt.figure(figsize=(8,4))
        counts, edges = np.histogram(col_s, bins=bins)
        plt.bar((edges[:-1]+edges[1:])/2, counts, width=(edges[1]-edges[0]))
        plt.yscale('log')
        plt.title(f'Histogram (log-y): {names[j]}')
        plt.xlabel('value')
        plt.ylabel('count (log scale)')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"hist_logy_{j}_{names[j]}.png"), dpi=200)
        plt.close()

        # cumulative distribution
        sorted_col = np.sort(col_s)
        cdf = np.arange(1, sorted_col.size+1) / float(sorted_col.size)
        plt.figure(figsize=(8,4))
        plt.plot(sorted_col, cdf)
        plt.title(f'CDF: {names[j]}')
        plt.xlabel('value')
        plt.ylabel('cumulative prob')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"cdf_{j}_{names[j]}.png"), dpi=200)
        plt.close()

def plot_violin_like(data, valid_mask, names, outpath, sample_per_col=2000):
    # violin-like: IQR rectangle + median + sampled points jitter
    n = data.shape[1]
    plt.figure(figsize=(12,6))
    ax = plt.gca()
    xs = []
    for j in range(n):
        col = data[valid_mask, j]
        col = col[~np.isnan(col)]
        if col.size == 0:
            xs.append([])
            continue
        # sample some points for swarm
        if col.size > sample_per_col:
            col_s = np.random.choice(col, sample_per_col, replace=False)
        else:
            col_s = col
        x_j = np.random.normal(loc=j, scale=0.06, size=col_s.size)
        ax.scatter(x_j, col_s, alpha=0.3, s=6)
        # IQR box
        p25 = np.percentile(col, 25)
        p75 = np.percentile(col, 75)
        med = np.median(col)
        ax.plot([j-0.15, j+0.15], [med, med], color='k', linewidth=3)
        ax.add_patch(plt.Rectangle((j-0.15, p25), 0.3, p75-p25, alpha=0.2))
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_title('Violin-like (sampled points + IQR + median)')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

def plot_corr_heatmap(data, valid_mask, names, outpath):
    # compute corr on valid rows
    good = data[valid_mask]
    # remove columns with all nan
    col_ok = ~np.isnan(good).all(axis=0)
    sub = good[:, col_ok]
    # replace nans per column with column mean (for correlation computation)
    col_means = np.nanmean(sub, axis=0)
    inds = np.where(np.isnan(sub))
    sub[inds] = np.take(col_means, inds[1])
    corr = np.corrcoef(sub, rowvar=False)
    plt.figure(figsize=(8,6))
    im = plt.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_names = [names[i] for i, ok in enumerate(col_ok) if ok]
    plt.xticks(range(len(tick_names)), tick_names, rotation=45, ha='right')
    plt.yticks(range(len(tick_names)), tick_names)
    plt.title('Correlation matrix (Pearson)')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return corr, col_ok

def plot_topk_pairscatter(data, valid_mask, names, outpath_prefix, topk=6, sample_limit=20000):
    corr, col_ok = plot_corr_heatmap(data, valid_mask, names, outpath_prefix + "_corrheatmap.png")
    # flatten upper triangle pairs
    n = corr.shape[0]
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((abs(corr[i,j]), i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])
    plotted = 0
    for score, i, j in pairs:
        if plotted >= topk:
            break
        # map back to original column indices
        col_indices = np.nonzero(col_ok)[0]
        a_idx = col_indices[i]
        b_idx = col_indices[j]
        col_a = data[valid_mask, a_idx]
        col_b = data[valid_mask, b_idx]
        # sample for speed
        mask = ~np.isnan(col_a) & ~np.isnan(col_b)
        col_a = col_a[mask]
        col_b = col_b[mask]
        if col_a.size == 0:
            continue
        if col_a.size > sample_limit:
            idx = np.random.choice(col_a.size, sample_limit, replace=False)
            col_a = col_a[idx]
            col_b = col_b[idx]
        plt.figure(figsize=(5,5))
        plt.scatter(col_a, col_b, s=6, alpha=0.3)
        plt.xlabel(names[a_idx])
        plt.ylabel(names[b_idx])
        plt.title(f"Pair scatter: {names[a_idx]} vs {names[b_idx]} (|corr|={score:.3f})")
        plt.tight_layout()
        plt.savefig(f"{outpath_prefix}_pair_{plotted}_{names[a_idx]}_vs_{names[b_idx]}.png", dpi=200)
        plt.close()
        plotted += 1

def save_extended_stats(data, valid_mask, names, outpath):
    # compute extended stats and save JSON
    stats = {}
    for j, name in enumerate(names):
        col = data[valid_mask, j]
        stats[name] = extended_stats(col)
    with open(outpath, 'w') as f:
        json.dump({'property_names': names, 'extended_stats': stats}, f, indent=2)
    print("Saved extended stats JSON:", outpath)
    return stats

# ---- main CLI ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stats_path', help='Path to stats .npz (preferred) or .json (will search for .npz)')
    parser.add_argument('--out_dir', default='stats_plots', help='folder to save plots')
    parser.add_argument('--prefix', default='qm9', help='prefix for output files')
    parser.add_argument('--topk_pairs', type=int, default=6, help='number of top correlated pairs to scatter-plot')
    args = parser.parse_args()

    data, valid_mask, names = load_stats(args.stats_path)
    outdir = os.path.join(args.out_dir, f"{args.prefix}_plots")
    os.makedirs(outdir, exist_ok=True)

    # basic numeric summaries (means, stds, median, p25/p75)
    good = data[valid_mask]
    means = np.nanmean(good, axis=0)
    stds  = np.nanstd(good, axis=0, ddof=1)
    medians = np.nanmedian(good, axis=0)
    p25 = np.nanpercentile(good, 25, axis=0)
    p75 = np.nanpercentile(good, 75, axis=0)

    # 1) mean ± std
    plot_mean_std(means, stds, names, os.path.join(outdir, f"{args.prefix}_means_std.png"))
    # 2) median + IQR
    plot_median_iqr(medians, p25, p75, names, os.path.join(outdir, f"{args.prefix}_median_iqr.png"))
    # 3) boxplot all
    plot_boxplot_all(data, valid_mask, names, os.path.join(outdir, f"{args.prefix}_boxplots.png"))
    # 4) histograms + CDFs per property
    plot_histograms(data, valid_mask, names, outdir)
    # 5) violin-like
    plot_violin_like(data, valid_mask, names, os.path.join(outdir, f"{args.prefix}_violin_like.png"))
    # 6) corr heatmap (returns corr)
    corr, col_ok = plot_corr_heatmap(data, valid_mask, names, os.path.join(outdir, f"{args.prefix}_corr_heatmap.png"))
    # 7) pair scatter top-k correlations
    plot_topk_pairscatter(data, valid_mask, names, os.path.join(outdir, f"{args.prefix}"), topk=args.topk_pairs)
    # 8) extended stats JSON
    extended = save_extended_stats(data, valid_mask, names, os.path.join(outdir, f"{args.prefix}_extended_stats.json"))

    print("All plots saved to:", outdir)
    print("Done.")

if __name__ == "__main__":
    main()
