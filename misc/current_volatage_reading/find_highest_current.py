import argparse
import re
import sys


CURRENT_RE = re.compile(r"\bcurrent:([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)\b")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse output.txt and print the highest 'current:' value."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="output.txt",
        help="Path to the log file (default: output.txt)",
    )
    args = parser.parse_args()

    max_current = None

    try:
        with open(args.path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                m = CURRENT_RE.search(line)
                if not m:
                    continue
                try:
                    val = float(m.group(1))
                except ValueError:
                    continue
                if max_current is None or val > max_current:
                    max_current = val
    except FileNotFoundError:
        print(f"File not found: {args.path}", file=sys.stderr)
        return 2

    if max_current is None:
        print("No 'current:' values found.", file=sys.stderr)
        return 1

    print(max_current)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

