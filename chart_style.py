"""Common matplotlib settings so all the charts look the same.

Colors are handed out in the same order every time, so e.g. heap sort is
always blue. The palette was picked to stay readable for color-blind
viewers.
"""

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

BACKGROUND = "#fcfcfb"
TEXT = "#0b0b0b"
LIGHT_TEXT = "#52514e"
GRID = "#e4e3df"


def setup(plt):
    plt.rcParams.update({
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
        "axes.edgecolor": GRID,
        "axes.labelcolor": LIGHT_TEXT,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": LIGHT_TEXT,
        "ytick.color": LIGHT_TEXT,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "font.family": "sans-serif",
    })
