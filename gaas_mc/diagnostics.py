"""Plotting helpers for validation against [C21] figures.

Axes mimic the paper (log rates 1e10-1e15 s^-1 or 1e5-1e13 s^-1, energy 0-1 eV) so that
plots can be compared side by side with Figs. 7, 8, and 10-13. Colors are bound to the
entity (mechanism), never to its plotting order. Markers give a secondary encoding.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .constants import EV  # noqa: E402

# fixed categorical slots (validated reference palette, light mode)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ENTITY_STYLE = {
    "acoustic": (SLOTS[0], "o"),
    "pop_abs": (SLOTS[1], "s"),
    "pop_em": (SLOTS[2], "D"),
    "impurity": (SLOTS[7], "v"),
    "eh": (SLOTS[6], "^"),
    "intervalley": (SLOTS[4], "P"),
    "EY": (SLOTS[0], "o"),
    "DP": (SLOTS[1], "s"),
    "BAP": (SLOTS[2], "D"),
    "total": ("#444441", None),
}
INK = "#2b2b29"
MUTED = "#8a8a84"


NAMED_STYLE = {
    "iv_abs[Gamma->L]": (SLOTS[3], "^"),
    "iv_em[Gamma->L]": ("#2b2b29", "v"),
    "iv_abs[Gamma->X]": (SLOTS[6], "o"),
    "iv_em[Gamma->X]": (SLOTS[4], "s"),
    "eh_hh[Gamma]": (SLOTS[5], "*"),
    "eh_lh[Gamma]": (SLOTS[5], "x"),
}


def style_for(name):
    if name in NAMED_STYLE:
        return NAMED_STYLE[name]
    key = name.split("[")[0]
    return ENTITY_STYLE.get(key, (MUTED, None))


def setup_axes(ax, ylabel, ylim=(1e10, 1e15), xlim=(0, 1.0)):
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Electron energy, eV", color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, which="major", color="#e4e3dd", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK)


def plot_curve(ax, E, y, name, label=None, ls="-", markevery=None):
    color, marker = style_for(name)
    E_ev = np.asarray(E) / EV
    y = np.where(np.asarray(y) > 0, y, np.nan)
    ax.plot(E_ev, y, ls, color=color, lw=2, marker=marker, ms=5,
            markevery=markevery or max(1, len(E_ev) // 12), label=label or name)


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
