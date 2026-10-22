"""Графики для EDA. Подписи на английском — рисунки пойдут в диссертацию.

Каждая функция принимает таблицу из eda.py и возвращает matplotlib Figure.
save(fig, name, out_dir) сохраняет PNG (300 dpi, для печати) — в ноутбуке
папка указывает на Drive, чтобы рисунки не пропадали вместе с сессией.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Палитра: фиксированный порядок цветов, проверенный на различимость при дальтонизме
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SPLIT_COLORS = {"train": PALETTE[0], "dev": PALETTE[1], "test": PALETTE[2]}
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
THOUSANDS = plt.FuncFormatter(lambda v, _: f"{v:,.0f}")


def style():
    """Единый спокойный стиль: тонкие линии, светлая сетка, без лишних рамок."""
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "normal",
        "axes.titlelocation": "left", "axes.labelcolor": INK_MUTED,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "text.color": INK,
        "legend.frameon": False,
    })


def save(fig, name, out_dir):
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / f"{name}.png")


def _bar_labels(ax, bars, fmt="{:,.0f}", horizontal=False):
    for b in bars:
        v = b.get_width() if horizontal else b.get_height()
        if horizontal:
            ax.annotate(fmt.format(v), (v, b.get_y() + b.get_height() / 2), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK_MUTED)
        else:
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=8, color=INK_MUTED)


# --------------------------------------------------------------- 1. сплиты

def split_sizes(sizes):
    """Малые множители: у каждого датасета своя шкала (размеры отличаются в 100 раз)."""
    names = list(dict.fromkeys(sizes["dataset"]))
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3), sharey=False)
    for ax, name in zip(np.atleast_1d(axes), names):
        d = sizes[sizes["dataset"] == name]
        bars = ax.bar(d["split"], d["sentences"], width=0.6,
                      color=[SPLIT_COLORS[s] for s in d["split"]])
        _bar_labels(ax, bars)
        ax.set_title(name)
        ax.yaxis.set_major_formatter(THOUSANDS)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.15)
    np.atleast_1d(axes)[0].set_ylabel("Sentences")
    fig.tight_layout()
    return fig


def domain_mix(mixes):
    """mixes: {"Official train": Series долей, ...} → 100%-ные полосы по исходным текстам."""
    labels = list(mixes)
    docs = list(next(iter(mixes.values())).index)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(labels) + 1.2))
    left = np.zeros(len(labels))
    for k, doc in enumerate(docs):
        vals = np.array([mixes[l].get(doc, 0) for l in labels])
        ax.barh(labels, vals, left=left, color=PALETTE[k % len(PALETTE)], label=doc,
                edgecolor="white", linewidth=1.5, height=0.6)
        left += vals
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Share of sentences")
    ax.grid(axis="y", visible=False)
    ax.legend(ncol=len(docs), loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=8,
              handlelength=1, columnspacing=1)
    fig.tight_layout()
    return fig


# --------------------------------------------------------- 2. классы

def class_distribution(counts, title, xlabel, log=False):
    """Горизонтальные полосы по убыванию (counts — Series «класс → число»)."""
    counts = counts.sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.26 * len(counts) + 1))
    bars = ax.barh(counts.index, counts.values, color=PALETTE[0], height=0.7)
    _bar_labels(ax, bars, horizontal=True)
    if log:
        ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.12)
    fig.tight_layout()
    return fig


def split_shares(shares, title):
    """Доля каждого класса в train/dev/test: точки совпадают → сплиты согласованы."""
    shares = shares[["train", "dev", "test"]].sort_values("train")
    fig, ax = plt.subplots(figsize=(7, 0.26 * len(shares) + 1))
    y = np.arange(len(shares))
    for k, sp in enumerate(["train", "dev", "test"]):
        ax.scatter(shares[sp], y + (k - 1) * 0.18, s=18, color=SPLIT_COLORS[sp], label=sp, zorder=3)
    ax.set_yticks(y, shares.index)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title(title)
    ax.set_xlabel("Share within split")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# ---------------------------------------------------- 3. длина предложений

def sentence_lengths(lengths, outliers):
    names = list(dict.fromkeys(lengths["dataset"]))
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 3.2))
    for ax, name in zip(np.atleast_1d(axes), names):
        v = lengths.loc[lengths["dataset"] == name, "length"]
        ax.hist(v, bins=range(1, v.max() + 2), color=PALETTE[0], edgecolor="white", linewidth=0.4)
        med, fence = outliers.loc[name, "median"], outliers.loc[name, "upper_fence"]
        for x, lab, ls, dx, ha in ((med, f"median {med:.0f}", "-", -3, "right"),
                                   (fence, f"outlier fence {fence:.1f}", "--", 3, "left")):
            ax.axvline(x, color=INK_MUTED, linewidth=1, linestyle=ls)
            ax.annotate(lab, (x, 1), xycoords=("data", "axes fraction"), xytext=(dx, -12),
                        textcoords="offset points", fontsize=8, color=INK_MUTED, ha=ha,
                        backgroundcolor="white")
        ax.yaxis.set_major_formatter(THOUSANDS)
        ax.set_title(name)
        ax.set_xlabel("Tokens per sentence")
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel("Sentences")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------- 4. редкие слова

def frequency_buckets(buckets):
    """buckets: {"KazNERD": таблица, "UD-KTB": таблица} из eda.frequency_buckets."""
    fig, axes = plt.subplots(1, len(buckets), figsize=(5 * len(buckets), 3.2), sharey=True)
    for ax, (name, b) in zip(np.atleast_1d(axes), buckets.items()):
        x = np.arange(len(b))
        b1 = ax.bar(x - 0.2, b["types_share"], 0.38, color=PALETTE[0], label="share of word types")
        b2 = ax.bar(x + 0.2, b["tokens_share"], 0.38, color=PALETTE[1], label="share of running words")
        _bar_labels(ax, list(b1) + list(b2), fmt="{:.0%}")
        ax.set_xticks(x, b.index)
        ax.set_xlabel("Word frequency in train")
        ax.set_title(name)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.12)
    np.atleast_1d(axes)[0].legend(loc="upper right")
    fig.tight_layout()
    return fig


def oov_rates(vocab):
    """vocab: {"KazNERD": таблица, ...} из eda.vocabulary_stats → OOV dev/test."""
    names = list(vocab)
    fig, ax = plt.subplots(figsize=(5, 3))
    x = np.arange(len(names))
    for k, sp in enumerate(["dev", "test"]):
        vals = [vocab[n].loc[sp, "oov_token_rate"] for n in names]
        bars = ax.bar(x + (k - 0.5) * 0.36, vals, 0.34, color=SPLIT_COLORS[sp], label=sp)
        _bar_labels(ax, bars, fmt="{:.1%}")
    ax.set_xticks(x, names)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("Running words unseen in train")
    ax.set_title("Out-of-vocabulary rate")
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.15)
    ax.legend()
    fig.tight_layout()
    return fig


# ------------------------------------------------- 5. до / после очистки

def cleaning_summary(cleaning_counts):
    """cleaning_counts: Series «правило → число исправлений»."""
    fig, ax = plt.subplots(figsize=(5, 0.45 * len(cleaning_counts) + 1))
    bars = ax.barh(cleaning_counts.index, cleaning_counts.values, color=PALETTE[0], height=0.6)
    _bar_labels(ax, bars, horizontal=True)
    ax.set_title("Corrections applied during preprocessing")
    ax.set_xlabel("Number of corrections")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.15)
    fig.tight_layout()
    return fig
