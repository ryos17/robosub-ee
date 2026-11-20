"""Figure 1 of the OCEANS 2027 abstract: WER by range, front ends vs recognizer adaptation [SIM].

Numbers are Table II of docs/history/oceans2027_methods_results_draft.md (E7 dry run, uwsim v0, guarded
decoding); replace them with the v0-test pool results when those exist.

    python paper/fig_wer.py   # writes paper/fig_wer.pdf
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RANGES = ["1 m", "3 m", "6 m"]
WER = {  # condition -> system -> WER % at 1 / 3 / 6 m
    "Quiet (30 dB SNR)": {"Unprocessed": [3.1, 3.7, 4.3], "WPE": [3.0, 3.4, 4.4],
                          "DFN3 fine-tuned": [2.3, 6.0, 9.2], "Whisper + LoRA": [2.3, 2.7, 3.4]},
    "Stationary noise, 0 dB SNR": {"Unprocessed": [20.3, 48.4, 56.4], "WPE": [21.1, 48.3, 56.9],
                                   "DFN3 fine-tuned": [41.6, 81.6, 89.9], "Whisper + LoRA": [16.4, 38.2, 43.8]},
}
COLORS = {"Unprocessed": "#C8CCD0", "WPE": "#8FA9D6", "DFN3 fine-tuned": "#3B64AE", "Whisper + LoRA": "#D9833B"}


def main() -> None:
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.8), constrained_layout=True)
    x = np.arange(len(RANGES))
    w = 0.2
    for ax, (cond, systems) in zip(axes, WER.items()):
        for k, (name, v) in enumerate(systems.items()):
            ax.bar(x + (k - 1.5) * w, v, w, color=COLORS[name], label=name, linewidth=0)
        ax.set_xticks(x, RANGES)
        ax.set_title(cond, fontsize=7)
        ax.tick_params(length=2)
        ax.grid(axis="y", linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(max(v) for v in systems.values()) * 1.08)
    axes[0].set_ylabel("WER (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, fontsize=5.5, frameon=False,
               handlelength=1, columnspacing=0.8)
    out = Path(__file__).with_name("fig_wer.pdf")
    fig.savefig(out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
