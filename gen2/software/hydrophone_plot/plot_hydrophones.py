"""
Plot four hydrophone microphone channels over time from a log file.

Expected input line format (one per sample):
  HH:MM:SS:ms -> Mic0:  0.269 Mic1:  0.085 Mic2:  0.198 Mic3:  0.106

Usage:
  python plot_hydrophones.py hydrophone_data.txt
  python plot_hydrophones.py hydrophone_data_2.txt --save hydrophones.png
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List, Tuple

import matplotlib.pyplot as plt


LINE_REGEX = re.compile(
	# Example: 13:59:51:906 -> Mic0:  0.269 Mic1:  0.085 Mic2:  0.198 Mic3:  0.106
	r"^(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2}):(?P<millisecond>\d{2,3})\s*->\s*"
	r"Mic0:\s*(?P<mic0>-?[0-9]*\.?[0-9]+)\s+"
	r"Mic1:\s*(?P<mic1>-?[0-9]*\.?[0-9]+)\s+"
	r"Mic2:\s*(?P<mic2>-?[0-9]*\.?[0-9]+)\s+"
	r"Mic3:\s*(?P<mic3>-?[0-9]*\.?[0-9]+)\s*$",
)


def parse_hydrophone_file(path: str) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
	"""Parse the hydrophone log file.

	Returns:
	  (times_seconds, mic0, mic1, mic2, mic3)
	where times are relative to the first sample (t=0 at first line).
	"""
	if not os.path.isfile(path):
		raise FileNotFoundError(f"Input file does not exist: {path}")

	times_seconds: List[float] = []
	mic0: List[float] = []
	mic1: List[float] = []
	mic2: List[float] = []
	mic3: List[float] = []

	first_abs_seconds: float | None = None

	with open(path, "r", encoding="utf-8") as f:
		for line_number, raw_line in enumerate(f, start=1):
			line = raw_line.strip()
			if not line:
				continue
			m = LINE_REGEX.match(line)
			if not m:
				# Silently skip lines that don't match (or could raise if desired)
				continue

			hour = int(m.group("hour"))
			minute = int(m.group("minute"))
			second = int(m.group("second"))
			millisecond = int(m.group("millisecond"))
			absolute_seconds = (
				hour * 3600.0 + minute * 60.0 + second + (millisecond / 1000.0)
			)
			if first_abs_seconds is None:
				first_abs_seconds = absolute_seconds
			relative_seconds = absolute_seconds - first_abs_seconds

			times_seconds.append(relative_seconds)
			mic0.append(float(m.group("mic0")))
			mic1.append(float(m.group("mic1")))
			mic2.append(float(m.group("mic2")))
			mic3.append(float(m.group("mic3")))

	if not times_seconds:
		raise ValueError("No valid data lines parsed. Check the input format.")

	return times_seconds, mic0, mic1, mic2, mic3


def plot_hydrophones(
	times_seconds: List[float],
	mic0: List[float],
	mic1: List[float],
	mic2: List[float],
	mic3: List[float],
	title: str,
	save_path: str | None = None,
) -> None:
	plt.figure(figsize=(11, 6))
	plt.plot(times_seconds, mic0, label="Mic0")
	plt.plot(times_seconds, mic1, label="Mic1")
	plt.plot(times_seconds, mic2, label="Mic2")
	plt.plot(times_seconds, mic3, label="Mic3")
	plt.xlabel("Time (s)")
	plt.ylabel("Amplitude")
	plt.title(title)
	plt.grid(True, linestyle=":", alpha=0.5)
	plt.legend()
	plt.tight_layout()

	if save_path:
		plt.savefig(save_path, dpi=150)
		print(f"Saved plot to: {save_path}")
	else:
		plt.show()


def crop_time_range(
	times_seconds: List[float],
	mic0: List[float],
	mic1: List[float],
	mic2: List[float],
	mic3: List[float],
	start_s: float | None,
	end_s: float | None,
) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
	"""Return a time-cropped view of the data. Keeps points where start_s <= t <= end_s.

	If start_s or end_s is None, the corresponding bound is unbounded.
	"""
	if start_s is not None and end_s is not None and start_s > end_s:
		raise ValueError(f"Invalid range: start ({start_s}) must be <= end ({end_s}).")

	def in_range(t: float) -> bool:
		if start_s is not None and t < start_s:
			return False
		if end_s is not None and t > end_s:
			return False
		return True

	filtered_indices = [i for i, t in enumerate(times_seconds) if in_range(t)]
	if not filtered_indices:
		raise ValueError("No data points fall within the requested time range.")

	idxs = filtered_indices
	return (
		[times_seconds[i] for i in idxs],
		[mic0[i] for i in idxs],
		[mic1[i] for i in idxs],
		[mic2[i] for i in idxs],
		[mic3[i] for i in idxs],
	)


def main() -> None:
	parser = argparse.ArgumentParser(description="Plot four hydrophone channels over time.")
	parser.add_argument(
		"input_file",
		nargs="?",
		default="hydrophone_data.txt",
		help="Path to input data file (default: data/30kHz_100ms_first_run.txt)",
	)
	parser.add_argument(
		"--save",
		dest="save",
		default=None,
		help="Optional path to save the plot instead of displaying it.",
	)
	parser.add_argument(
		"--start",
		dest="start",
		type=float,
		default=None,
		help="Start time in seconds for cropping (inclusive).",
	)
	parser.add_argument(
		"--end",
		dest="end",
		type=float,
		default=None,
		help="End time in seconds for cropping (inclusive).",
	)
	args = parser.parse_args()

	times_seconds, mic0, mic1, mic2, mic3 = parse_hydrophone_file(args.input_file)
	if args.start is not None or args.end is not None:
		times_seconds, mic0, mic1, mic2, mic3 = crop_time_range(
			times_seconds, mic0, mic1, mic2, mic3, args.start, args.end
		)
		range_str = (
			f" [t in {args.start if args.start is not None else '-∞'}–{args.end if args.end is not None else '∞'} s]"
		)
	else:
		range_str = ""

	title = f"Hydrophone Channels vs Time — {os.path.basename(args.input_file)}{range_str}"
	plot_hydrophones(times_seconds, mic0, mic1, mic2, mic3, title=title, save_path=args.save)


if __name__ == "__main__":
	main()


