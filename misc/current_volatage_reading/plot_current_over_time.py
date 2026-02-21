import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta


CURRENT_RE = re.compile(r"\bcurrent:([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\b")
TIME_RE = re.compile(
    r"^(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})\.(?P<ms>\d{3})\s*->"
)


@dataclass(frozen=True)
class Point:
    t: datetime
    current: float


def parse_points(path: str) -> list[Point]:
    points: list[Point] = []

    last_seconds_of_day: float | None = None
    day_offset_seconds = 0.0
    base = datetime(2000, 1, 1)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            tm = TIME_RE.search(line)
            cm = CURRENT_RE.search(line)
            if not tm or not cm:
                continue

            try:
                h = int(tm.group("h"))
                m = int(tm.group("m"))
                s = int(tm.group("s"))
                ms = int(tm.group("ms"))
            except ValueError:
                continue

            try:
                current = float(cm.group(1))
            except ValueError:
                continue

            seconds_of_day = h * 3600.0 + m * 60.0 + s + ms / 1000.0
            if (
                last_seconds_of_day is not None
                and seconds_of_day + 1e-6 < last_seconds_of_day
            ):
                # Time went backwards; assume log crossed midnight.
                day_offset_seconds += 86400.0
            last_seconds_of_day = seconds_of_day

            t = base + timedelta(seconds=day_offset_seconds + seconds_of_day)
            points.append(Point(t=t, current=current))

    return points


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot 'current' over time from output.txt-like logs."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="output.txt",
        help="Path to the log file (default: output.txt)",
    )
    parser.add_argument(
        "--out",
        default="current_over_time.png",
        help="Output image path (default: current_over_time.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show an interactive window (also saves --out).",
    )
    args = parser.parse_args()

    try:
        points = parse_points(args.path)
    except FileNotFoundError:
        print(f"File not found: {args.path}", file=sys.stderr)
        return 2

    if not points:
        print(
            "No points found. Expected lines like 'HH:MM:SS.mmm -> ... current:<value> ...'",
            file=sys.stderr,
        )
        return 1

    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except Exception as e:
        print(
            "matplotlib is required to plot. Install it with:\n"
            "  python -m pip install matplotlib\n\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        return 3

    xs = [p.t for p in points]
    ys = [p.current for p in points]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(xs, ys, linewidth=1)
    ax.set_title("Current over time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Current")
    ax.grid(True, alpha=0.3)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    fig.tight_layout()

    fig.savefig(args.out, dpi=200)
    print(f"Saved plot to: {args.out}")

    if args.show:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

