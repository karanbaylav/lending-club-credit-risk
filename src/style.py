"""One chart style for the project: neutral greys, one blue accent, one orange for warnings."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, GREY, LIGHT, ACCENT, WARN, BG = "#1E2A38", "#9AA3AD", "#D7DCE1", "#1F5FAD", "#C8622A", "#FFFFFF"
GRADE_COLORS = {"A": "#0B3C78", "B": "#1F5FAD", "C": "#4A86C8", "D": "#8CB4DD", "E": "#E0A17A", "F": "#C8622A", "G": "#8E3B12"}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": LIGHT, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#EEF1F4", "grid.linewidth": 0.8, "figure.dpi": 110,
    "savefig.dpi": 160, "savefig.bbox": "tight", "figure.facecolor": BG,
})


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
