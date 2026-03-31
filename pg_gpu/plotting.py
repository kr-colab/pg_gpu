"""
Population genetics visualization functions.

Provides publication-quality plots for SFS, LD, haplotype structure,
PCA, distance matrices, and windowed statistics using matplotlib and seaborn.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import Optional, Union, Tuple


# ---------------------------------------------------------------------------
# Site Frequency Spectrum
# ---------------------------------------------------------------------------

def plot_sfs(s, ax=None, folded=False, scaled=False, clip_endpoints=True,
             log_scale=True, color=None, **kwargs):
    """Plot a site frequency spectrum.

    Parameters
    ----------
    s : array_like
        SFS array (from pg_gpu.sfs.sfs, sfs_folded, etc.).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates a new figure.
    folded : bool
        Label x-axis as minor allele count.
    scaled : bool
        Label y-axis as scaled count.
    clip_endpoints : bool
        If True, exclude the first and last entries (monomorphic classes).
    log_scale : bool
        If True, use log scale for y-axis.
    color : str, optional
        Bar color.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    s = np.asarray(s, dtype=float)

    if clip_endpoints:
        s = s[1:-1] if not folded else s[1:]
        x = np.arange(1, len(s) + 1)
    else:
        x = np.arange(len(s))

    if color is None:
        color = sns.color_palette()[0]

    ax.bar(x, s, color=color, edgecolor='none', **kwargs)

    if log_scale and np.any(s > 0):
        ax.set_yscale('log')

    xlabel = 'Minor allele count' if folded else 'Derived allele count'
    ylabel = 'Scaled count' if scaled else 'Count'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    sns.despine(ax=ax)

    return ax


def plot_joint_sfs(s, ax=None, log_scale=True, cmap='YlOrRd',
                   clip_endpoints=True, **kwargs):
    """Plot a joint site frequency spectrum as a heatmap.

    Parameters
    ----------
    s : array_like, 2D
        Joint SFS array (from pg_gpu.sfs.joint_sfs, etc.).
    ax : matplotlib.axes.Axes, optional
    log_scale : bool
        If True, use log color scale.
    cmap : str
        Colormap name.
    clip_endpoints : bool
        If True, exclude monomorphic classes (row/col 0 and last).

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    s = np.asarray(s, dtype=float)

    if clip_endpoints:
        s = s[1:, 1:]

    # mask zeros for log scale
    s_plot = np.ma.masked_equal(s, 0)

    norm = mcolors.LogNorm(vmin=max(1, s_plot.min()), vmax=s_plot.max()) if log_scale else None

    im = ax.pcolormesh(s_plot, norm=norm, cmap=cmap, **kwargs)
    plt.colorbar(im, ax=ax, label='Count')

    offset = 1 if clip_endpoints else 0
    ax.set_xlabel(f'Derived allele count (pop 2)')
    ax.set_ylabel(f'Derived allele count (pop 1)')
    sns.despine(ax=ax)

    return ax


# ---------------------------------------------------------------------------
# Linkage Disequilibrium
# ---------------------------------------------------------------------------

def plot_pairwise_ld(r2_matrix, ax=None, cmap='Greys', vmin=0, vmax=1,
                     positions=None, **kwargs):
    """Plot a pairwise LD (r-squared) matrix as a heatmap.

    Parameters
    ----------
    r2_matrix : array_like, 2D
        Square r-squared matrix (from HaplotypeMatrix.pairwise_r2()).
    ax : matplotlib.axes.Axes, optional
    cmap : str
        Colormap.
    vmin, vmax : float
        Color scale range.
    positions : array_like, optional
        Variant positions for axis labels.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    r2 = np.asarray(r2_matrix)

    im = ax.imshow(r2, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='none', origin='lower', **kwargs)
    plt.colorbar(im, ax=ax, label='$r^2$')

    if positions is not None:
        positions = np.asarray(positions)
        n_ticks = min(6, len(positions))
        tick_idx = np.linspace(0, len(positions) - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([f'{positions[i]/1000:.0f}kb' for i in tick_idx],
                           rotation=45)
        ax.set_yticks(tick_idx)
        ax.set_yticklabels([f'{positions[i]/1000:.0f}kb' for i in tick_idx])

    ax.set_xlabel('Variant')
    ax.set_ylabel('Variant')

    return ax


def plot_ld_decay(distances, r2_values, ax=None, bins=50, color=None,
                  percentile=50, **kwargs):
    """Plot LD decay curve (r-squared vs distance).

    Parameters
    ----------
    distances : array_like
        Pairwise distances between variants.
    r2_values : array_like
        Pairwise r-squared values.
    ax : matplotlib.axes.Axes, optional
    bins : int or array_like
        Number of distance bins or explicit bin edges.
    color : str, optional
    percentile : float
        Percentile of r-squared to plot per bin (default: median).

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    distances = np.asarray(distances)
    r2_values = np.asarray(r2_values)

    valid = ~np.isnan(r2_values)
    distances = distances[valid]
    r2_values = r2_values[valid]

    if isinstance(bins, int):
        bins = np.linspace(distances.min(), distances.max(), bins + 1)

    bin_idx = np.digitize(distances, bins) - 1
    n_bins = len(bins) - 1
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_vals = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = bin_idx == i
        if np.sum(mask) > 0:
            bin_vals[i] = np.percentile(r2_values[mask], percentile)

    if color is None:
        color = sns.color_palette()[0]

    ax.plot(bin_centers, bin_vals, '-o', color=color, markersize=3, **kwargs)
    ax.set_xlabel('Distance (bp)')
    ax.set_ylabel('$r^2$')
    ax.set_ylim(bottom=0)
    sns.despine(ax=ax)

    return ax


# ---------------------------------------------------------------------------
# PCA / Population Structure
# ---------------------------------------------------------------------------

def plot_pca(coords, ax=None, pc_x=0, pc_y=1, explained_variance=None,
             labels=None, palette='Set2', s=20, alpha=0.8, **kwargs):
    """Scatter plot of PCA coordinates.

    Parameters
    ----------
    coords : array_like, shape (n_samples, n_components)
        PCA coordinates (from pg_gpu.decomposition.pca).
    ax : matplotlib.axes.Axes, optional
    pc_x, pc_y : int
        Which principal components to plot (0-indexed).
    explained_variance : array_like, optional
        Variance explained ratios for axis labels.
    labels : array_like, optional
        Population labels per sample for coloring.
    palette : str
        Seaborn palette name.
    s : float
        Marker size.
    alpha : float
        Marker transparency.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    coords = np.asarray(coords)

    if labels is not None:
        labels = np.asarray(labels)
        unique_labels = np.unique(labels)
        colors = sns.color_palette(palette, len(unique_labels))
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(coords[mask, pc_x], coords[mask, pc_y],
                       c=[colors[i]], label=str(label), s=s, alpha=alpha,
                       **kwargs)
        ax.legend(title='Population', frameon=True)
    else:
        color = sns.color_palette(palette)[0]
        ax.scatter(coords[:, pc_x], coords[:, pc_y], c=[color],
                   s=s, alpha=alpha, **kwargs)

    if explained_variance is not None:
        ev = np.asarray(explained_variance)
        ax.set_xlabel(f'PC{pc_x + 1} ({ev[pc_x]:.1%})')
        ax.set_ylabel(f'PC{pc_y + 1} ({ev[pc_y]:.1%})')
    else:
        ax.set_xlabel(f'PC{pc_x + 1}')
        ax.set_ylabel(f'PC{pc_y + 1}')

    sns.despine(ax=ax)
    return ax


# ---------------------------------------------------------------------------
# Distance Matrix
# ---------------------------------------------------------------------------

def plot_pairwise_distance(dist, ax=None, labels=None, cmap='viridis',
                           **kwargs):
    """Plot a pairwise distance matrix as a heatmap.

    Parameters
    ----------
    dist : array_like
        Condensed or square distance matrix.
    ax : matplotlib.axes.Axes, optional
    labels : array_like, optional
        Sample labels for axes.
    cmap : str
        Colormap.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    from scipy.spatial.distance import squareform

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    dist = np.asarray(dist, dtype=float)
    if dist.ndim == 1:
        D = squareform(dist)
    else:
        D = dist

    im = ax.imshow(D, cmap=cmap, interpolation='none', origin='lower',
                   **kwargs)
    plt.colorbar(im, ax=ax, label='Distance')

    if labels is not None:
        labels = np.asarray(labels)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)

    return ax


# ---------------------------------------------------------------------------
# Windowed Statistics
# ---------------------------------------------------------------------------

def plot_windowed(window_start, values, ax=None, stat_name='',
                  color=None, fill_alpha=0.15, **kwargs):
    """Plot windowed statistics along the genome.

    Parameters
    ----------
    window_start : array_like
        Window start positions (from windowed_statistics results).
    values : array_like
        Statistic values per window.
    ax : matplotlib.axes.Axes, optional
    stat_name : str
        Label for y-axis.
    color : str, optional
    fill_alpha : float
        Alpha for fill between line and zero.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))

    window_start = np.asarray(window_start)
    values = np.asarray(values, dtype=float)

    if color is None:
        color = sns.color_palette()[0]

    # convert to kb or Mb for readability
    if window_start.max() > 1e6:
        x = window_start / 1e6
        xlabel = 'Position (Mb)'
    elif window_start.max() > 1e3:
        x = window_start / 1e3
        xlabel = 'Position (kb)'
    else:
        x = window_start
        xlabel = 'Position (bp)'

    ax.plot(x, values, '-', color=color, linewidth=1, **kwargs)
    ax.fill_between(x, values, alpha=fill_alpha, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(stat_name)
    sns.despine(ax=ax)

    return ax


def plot_windowed_panel(results, statistics=None, figsize=None, colors=None):
    """Plot multiple windowed statistics as stacked panels.

    Parameters
    ----------
    results : dict
        Output from windowed_statistics() or windowed_statistics_fused().
        Must contain 'window_start' and statistic arrays.
    statistics : list of str, optional
        Which statistics to plot. If None, plots all non-metadata keys.
    figsize : tuple, optional
        Figure size.
    colors : list, optional
        Colors for each panel.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
    """
    skip_keys = {'window_start', 'window_stop', 'n_variants'}

    if statistics is None:
        statistics = [k for k in results if k not in skip_keys]

    n_panels = len(statistics)

    if figsize is None:
        figsize = (10, 2.5 * n_panels)

    if colors is None:
        colors = sns.color_palette('Set2', n_panels)

    fig, axes = plt.subplots(n_panels, 1, figsize=figsize, sharex=True)
    if n_panels == 1:
        axes = [axes]

    window_start = results['window_start']

    for i, stat in enumerate(statistics):
        plot_windowed(window_start, results[stat], ax=axes[i],
                      stat_name=stat, color=colors[i])

    plt.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Haplotype Visualization
# ---------------------------------------------------------------------------

def plot_haplotype_frequencies(freqs, ax=None, palette='Paired',
                               singleton_color='white'):
    """Plot haplotype frequencies as a horizontal stacked bar.

    Parameters
    ----------
    freqs : array_like
        Sorted haplotype frequencies (descending), e.g. from
        Garud's H computation.
    ax : matplotlib.axes.Axes, optional
    palette : str
        Seaborn palette for coloring distinct haplotypes.
    singleton_color : str
        Color for singleton haplotypes.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        width = plt.rcParams['figure.figsize'][0]
        fig, ax = plt.subplots(figsize=(width, width / 10))
        sns.despine(ax=ax, left=True)

    freqs = np.asarray(freqs, dtype=float)
    n_colors = np.sum(freqs > 1.0 / len(freqs)) if len(freqs) > 0 else 0
    n_colors = max(n_colors, 1)
    colors = sns.color_palette(palette, n_colors)

    x = 0
    for i, f in enumerate(freqs):
        if i < n_colors:
            color = colors[i]
        else:
            color = singleton_color
        ax.axvspan(x, x + f, color=color)
        x += f

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Haplotype frequency')

    return ax


def plot_variant_locator(positions, ax=None, step=None, color=None):
    """Plot variant positions showing density along the genome.

    Parameters
    ----------
    positions : array_like
        Sorted variant positions.
    ax : matplotlib.axes.Axes, optional
    step : int, optional
        Plot every `step`-th variant (for large datasets).
    color : str, optional

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        width = plt.rcParams['figure.figsize'][0]
        fig, ax = plt.subplots(figsize=(width, width / 7))

    positions = np.asarray(positions)

    if step is not None:
        positions = positions[::step]

    if color is None:
        color = sns.color_palette()[0]

    n = len(positions)
    for i, pos in enumerate(positions):
        ax.plot([i, pos], [1, 0], '-', color=color, linewidth=0.3, alpha=0.5)

    ax.set_xlim(0, max(n - 1, positions[-1]))
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([])

    # dual x-axis labels
    ax.set_xlabel('Genomic position (bp)')
    sns.despine(ax=ax, left=True)

    return ax
