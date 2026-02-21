"""
Clamp hydrophone microphone values so any value above an upper bound is cropped.

By default, clamps any value after labels like "Mic0:", "Mic1:", ... to a
maximum of 1.000, preserving the original line format and number of decimals.

Usage:
  python clamp_hydrophones.py data/30kHz_100ms_first_run.txt -o data/30kHz_100ms_first_run_clamped.txt
  python clamp_hydrophones.py data/25kHz_1ms_second_run_full_clipped.txt --inplace
  python clamp_hydrophones.py hydrophone_data.txt --upper 0.9 -o hydrophone_data_0p9.txt
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from typing import Tuple


# Matches occurrences like: "Mic0:   2.222" or "Mic3: -0.123" and captures:
#  - group 1: the label and whitespace before the number (kept as-is)
#  - group 2: the numeric value string to be replaced
MIC_VALUE_REGEX = re.compile(r"(Mic\d+:\s+)(-?[0-9]*\.?[0-9]+)")


def clamp_number_text(num_text: str, upper: float) -> str:
	"""Clamp numeric string to the given upper bound, preserving decimal places.

	If the original has N decimals, keep N decimals in the output. If it has no
	decimal point, output an integer when unchanged, or an integer if clamped.
	"""
	try:
		value = float(num_text)
	except ValueError:
		return num_text

	clamped = min(value, upper)

	# Preserve number of decimal places from the original text
	if "." in num_text:
		decimals = len(num_text.split(".")[1])
		formatted = f"{clamped:.{decimals}f}"
	else:
		# No decimals originally
		if clamped.is_integer():
			formatted = str(int(clamped))
		else:
			formatted = str(clamped)
	return formatted


def clamp_line(line: str, upper: float) -> Tuple[str, int]:
	"""Clamp all Mic* values in a line. Returns (new_line, num_clamped_or_changed)."""
	changes = 0

	def repl(match: re.Match) -> str:
		nonlocal changes
		prefix = match.group(1)
		num_text = match.group(2)
		try:
			val = float(num_text)
		except ValueError:
			return match.group(0)
		repl_text = clamp_number_text(num_text, upper)
		if repl_text != num_text:
			changes += 1
		return prefix + repl_text

	return MIC_VALUE_REGEX.sub(repl, line), changes


def process_file(input_path: str, output_path: str, upper: float) -> int:
	"""Read input, clamp values, and write output. Returns count of changes."""
	changes_total = 0
	with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
		for line in fin:
			new_line, changes = clamp_line(line, upper)
			changes_total += changes
			fout.write(new_line)
	return changes_total


def main() -> None:
	parser = argparse.ArgumentParser(description="Clamp Mic* values above a threshold (default 1.000).")
	parser.add_argument("input_file", help="Path to the input hydrophone data file.")
	parser.add_argument("-o", "--output", dest="output", default=None, help="Path to write the clamped file. If omitted and not using --inplace, appends _clamped before extension.")
	parser.add_argument("--inplace", action="store_true", help="Modify the file in place (a .bak backup is created).")
	parser.add_argument("--upper", type=float, default=1.0, help="Upper clamp value (default: 1.0).")
	args = parser.parse_args()

	input_path = args.input_file
	if not os.path.isfile(input_path):
		raise FileNotFoundError(f"Input file does not exist: {input_path}")

	if args.inplace:
		backup_path = input_path + ".bak"
		shutil.copy2(input_path, backup_path)
		output_path = input_path
	else:
		if args.output:
			output_path = args.output
		else:
			root, ext = os.path.splitext(input_path)
			output_path = f"{root}_clamped{ext or '.txt'}"

	changes = process_file(input_path, output_path, upper=args.upper)
	if args.inplace:
		print(f"Wrote clamped data in place to: {output_path} (backup at {backup_path})")
	else:
		print(f"Wrote clamped data to: {output_path}")
	print(f"Values changed (above {args.upper}): {changes}")


if __name__ == "__main__":
	main()


