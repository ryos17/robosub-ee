#!/usr/bin/env python3
"""
jlc_to_robosub — Fetch a JLCPCB/EasyEDA part and merge it into project-local KiCad libs.

USAGE (from the repo root, with this file named jlc_to_robosub.py):

  macOS / Linux:
      chmod +x jlc_to_robosub.py           # once
      ./jlc_to_robosub.py C22383822

  Windows:
      python jlc_to_robosub.py C22383822
  or  python3 jlc_to_robosub.py C22383822

(If you prefer, you can rename the file to "jlc_to_robosub" with no extension and
use "./jlc_to_robosub C22383822". The script itself doesn’t care about the name.)

WHAT THIS SCRIPT DOES:

  • Creates / uses the following project-local KiCad libraries in the repo root:
        robosub_symbols/robosub_symbols.kicad_sym
        robosub_footprints.pretty/
        robosub_3d_models/
        robosub_jlc_parts_log.csv

  • Downloads a single JLC part using JLC2KiCadLib and merges:
        - Symbol(s)   → robosub_symbols/robosub_symbols.kicad_sym
        - Footprint   → robosub_footprints.pretty/
        - 3D models   → robosub_3d_models/

  • Enforces:
        - Footprint lib nickname:  robosub_footprints
        - 3D model path variable:  ${ROBOSUB_3D_MODELS}/filename

  • Maintains a CSV log to avoid re-importing the same part:
        robosub_jlc_parts_log.csv

KICAD SETUP (done once per user):

  1. In KiCad, go to: Preferences → Configure Paths…
     Add a new path variable:
         Name : ROBOSUB_3D_MODELS
         Value: /absolute/path/to/your/repo/robosub_3d_models

  2. In the Symbol Library Manager:
       - Add a library pointing to:
            <repo-root>/robosub_symbols/robosub_symbols.kicad_sym
       - Give it any nickname (e.g. "robosub_symbols") and keep it in the project.

  3. In the Footprint Library Manager:
       - Add a library pointing to:
            <repo-root>/robosub_footprints.pretty
       - Set the nickname EXACTLY to:
            robosub_footprints
       - Add it at least to the project.

REQUIREMENTS:

  • Python 3 (3.7 or newer).
  • Internet access (to fetch data from JLCPCB / EasyEDA).
  • The script will automatically create a temporary virtualenv and install JLC2KiCadLib
    into it each time it runs (isolated from your system Python).
"""

import sys
import os
import re
import csv
import shutil
import subprocess
import tempfile
import venv
from pathlib import Path
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Configuration (project-local, relative to this script)
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent

SYMBOL_DIR = REPO_ROOT / "robosub_symbols"
SYMBOL_LIB = SYMBOL_DIR / "robosub_symbols.kicad_sym"

FP_LIB_DIR = REPO_ROOT / "robosub_footprints.pretty"
MODELS_DIR = REPO_ROOT / "robosub_3d_models"
LOG_FILE = REPO_ROOT / "robosub_jlc_parts_log.csv"

FP_LIB_NICKNAME = "robosub_footprints"
MODELS_ENV = "ROBOSUB_3D_MODELS"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_err(msg: str) -> None:
    print(msg, file=sys.stderr)


def ensure_dirs_and_files() -> None:
    """Create required directories and initial files if they do not exist."""
    SYMBOL_DIR.mkdir(parents=True, exist_ok=True)
    FP_LIB_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not SYMBOL_LIB.exists():
        # Minimal KiCad v6+ style symbol lib with no symbols yet
        SYMBOL_LIB.write_text(
            '(kicad_symbol_lib (version 20211014) (generator "jlc_to_robosub"))\n)\n',
            encoding="utf-8",
        )

    if not LOG_FILE.exists():
        with LOG_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["JLC_Part", "Symbol_Name", "Footprint", "Description"])


def already_processed(part: str) -> bool:
    """Check the log to see if this part has already been imported."""
    if not LOG_FILE.exists():
        return False
    with LOG_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        # skip header
        next(reader, None)
        for row in reader:
            if row and row[0] == part:
                return True
    return False


def create_venv(venv_dir: Path) -> Path:
    """Create a virtualenv at venv_dir and return the path to its Python executable."""
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(venv_dir)

    if os.name == "nt":
        python_path = venv_dir / "Scripts" / "python.exe"
    else:
        python_path = venv_dir / "bin" / "python"

    if not python_path.exists():
        raise RuntimeError("Could not find venv python at {}".format(python_path))

    return python_path


def run_subprocess(cmd, cwd: Optional[Path] = None) -> None:
    """Run a subprocess and raise if it fails."""
    if cwd is not None:
        cwd_str = str(cwd)
    else:
        cwd_str = None
    result = subprocess.run(cmd, cwd=cwd_str)
    if result.returncode != 0:
        raise RuntimeError("Command failed: {}".format(" ".join(cmd)))


def run_jlc2kicadlib(part: str, work_dir: Path) -> None:
    """Install JLC2KiCadLib into a temp venv and run it for the given part."""
    with tempfile.TemporaryDirectory() as tmp_root_str:
        tmp_root = Path(tmp_root_str)
        venv_dir = tmp_root / "venv"

        python_path = create_venv(venv_dir)

        # Install JLC2KiCadLib into the venv
        print("Installing JLC2KiCadLib into temporary virtualenv...")
        run_subprocess(
            [str(python_path), "-m", "pip", "install", "--quiet", "JLC2KiCadLib"]
        )

        # Find the CLI script installed into the venv
        if os.name == "nt":
            cli_path = venv_dir / "Scripts" / "JLC2KiCadLib.exe"
        else:
            cli_path = venv_dir / "bin" / "JLC2KiCadLib"

        if not cli_path.exists():
            raise RuntimeError(
                "Could not find JLC2KiCadLib CLI in venv at {}".format(cli_path)
            )

        # Run JLC2KiCadLib
        print("Running JLC2KiCadLib for part {} ...".format(part))
        work_dir.mkdir(parents=True, exist_ok=True)
        run_subprocess(
            [str(cli_path), part, "-dir", str(work_dir)],
            cwd=work_dir,
        )


def find_generated_files(work_dir: Path):
    """Locate .kicad_sym, .kicad_mod, and 3D model files in the work dir."""
    symbol_src = None
    fp_src_file = None
    model_files = []

    for path in work_dir.rglob("*"):
        if path.suffix == ".kicad_sym" and symbol_src is None:
            symbol_src = path
        elif path.suffix == ".kicad_mod" and fp_src_file is None:
            fp_src_file = path
        elif path.suffix.lower() in [".step", ".stp", ".wrl"]:
            model_files.append(path)

    return symbol_src, fp_src_file, model_files


def coherent_basename_from_symbol(symbol_name: str) -> str:
    """
    Derive a filesystem-safe basename from the symbol name.
    Symbol name remains the source of truth; we only normalize characters
    that are invalid in filenames/paths.
    """
    name = symbol_name.strip()
    name = name.replace("{slash}", "_").replace("/", "_").replace("\\", "_").replace(".", "_")
    name = re.sub(r'[<>:"|?*]', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = name.strip("._")
    return name or "part"


def rewrite_footprint_file_identity(fp_path: Path, new_base: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Rewrite a footprint file so its internal identity matches new_base.
    Also updates its model basename to new_base while preserving extension.
    Returns (old_model_basename, new_model_basename).
    """
    text = fp_path.read_text(encoding="utf-8")

    # Header name: (module "NAME"... or (footprint "NAME"...)
    text = re.sub(
        r'^(\(\s*(?:module|footprint)\s+")([^"]+)(")',
        r"\1" + new_base + r"\3",
        text,
        count=1,
        flags=re.M,
    )
    text = re.sub(r'(\(descr\s+")([^"]*)(\s+footprint"\))', r"\1" + new_base + r"\3", text, count=1)
    text = re.sub(r'(\(tags\s+")([^"]*)(\s+footprint[^"]*")', r"\1" + new_base + r"\3", text, count=1)
    text = re.sub(r'(\(fp_text\s+value\s+)([^\s\)]+)', r"\1" + new_base, text, count=1)

    old_model = None
    new_model = None
    m = re.search(r'(\(model\s+"[^"/\\]+[/\\])([^"/\\]+)(")', text)
    if m:
        old_model = m.group(2)
        ext = Path(old_model).suffix or ".step"
        new_model = "{}{}".format(new_base, ext)
        text = text.replace(old_model, new_model, 1)

    fp_path.write_text(text, encoding="utf-8")
    return old_model, new_model


def patch_symbol_source_footprint(symbol_src: Path, fp_basename_noext: str) -> None:
    """
    Ensure generated source symbol footprint points to the coherent footprint basename.
    """
    text = symbol_src.read_text(encoding="utf-8")
    text_new = re.sub(
        r'(\(property\s+"Footprint"\s+")([^"]+)(")',
        r"\1{}:{}\3".format(FP_LIB_NICKNAME, fp_basename_noext),
        text,
    )
    if text_new != text:
        symbol_src.write_text(text_new, encoding="utf-8")


def merge_symbol_into_master(symbol_src: Path) -> None:
    """
    Merge symbol(s) from symbol_src into SYMBOL_LIB.

    We:
      - Drop the final ')' of the master lib.
      - Drop the first line and final ')' of the generated file.
      - Append the inner symbol definitions.
      - Close with a final ')'.
    """
    master_lines = SYMBOL_LIB.read_text(encoding="utf-8").splitlines()
    src_lines = symbol_src.read_text(encoding="utf-8").splitlines()

    if not master_lines:
        raise RuntimeError("{} is empty or invalid.".format(SYMBOL_LIB))

    src_blocks = _extract_parent_symbol_blocks(src_lines)
    if not src_blocks:
        raise RuntimeError(
            "{} does not contain parent symbol blocks.".format(symbol_src)
        )

    if master_lines[-1].strip() != ")":
        raise RuntimeError("{} is malformed (missing final ')').".format(SYMBOL_LIB))

    merged_lines = master_lines[:-1]
    for block in src_blocks:
        merged_lines.extend(block)
    merged_lines.append(")")
    SYMBOL_LIB.write_text("\n".join(merged_lines) + "\n", encoding="utf-8")


def patch_footprint_properties() -> None:
    """Patch all 'Footprint' properties in SYMBOL_LIB to use FP_LIB_NICKNAME."""
    text = SYMBOL_LIB.read_text(encoding="utf-8")
    pattern = re.compile(r'(property\s+"Footprint"\s+")([^"]+)(")')

    def repl(match):
        current = match.group(2)
        fp_name = current.split(":", 1)[1] if ":" in current else current
        return '{}{}:{}{}'.format(match.group(1), FP_LIB_NICKNAME, fp_name, match.group(3))

    text_new = pattern.sub(repl, text)

    if text_new != text:
        SYMBOL_LIB.write_text(text_new, encoding="utf-8")


def _jlc_part_url(part: str) -> str:
    """
    Build a JLCPCB part-detail URL from a JLC/LCSC code.
    Example: C5120796 -> https://jlcpcb.com/partdetail/C5120796
    """
    return "https://jlcpcb.com/partdetail/{}".format(part.strip())


def patch_datasheet_for_symbol(symbol_name: str, part: str) -> None:
    """
    Point the parent symbol Datasheet property to the JLCPCB part page.
    Only patches the just-imported parent symbol block.
    """
    if not symbol_name:
        return

    lines = SYMBOL_LIB.read_text(encoding="utf-8").splitlines()
    target_url = _jlc_part_url(part)

    # Find parent symbol block by exact symbol name.
    start_idx = None
    end_idx = None
    depth = 0
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if depth == 1 and stripped.startswith('(symbol "{}"'.format(symbol_name)):
            start_idx = i
        depth += line.count("(") - line.count(")")
        if start_idx is not None and end_idx is None and depth == 1 and i > start_idx:
            end_idx = i
            break

    if start_idx is None or end_idx is None:
        return

    block = lines[start_idx : end_idx + 1]
    changed = False

    # Try to replace existing Datasheet property value.
    for i, line in enumerate(block):
        if '(property "Datasheet"' in line:
            new_line = re.sub(
                r'(\(property\s+"Datasheet"\s+")([^"]*)(")',
                lambda m: m.group(1) + target_url + m.group(3),
                line,
            )
            if new_line != line:
                block[i] = new_line
                changed = True
            break
    else:
        # Datasheet property not present: inject it before first child symbol.
        indent_match = re.match(r"^(\s*)", block[0])
        indent = (indent_match.group(1) if indent_match else "") + "  "
        insert_at = len(block) - 1
        for i, line in enumerate(block[1:], start=1):
            if line.lstrip().startswith('(symbol "'):
                insert_at = i
                break

        ds_lines = [
            '{}(property "Datasheet" "{}" (id 3) (at 0 0 0)'.format(indent, target_url),
            "{}  (effects (font (size 1.27 1.27) (italic yes)) hide)".format(indent),
            "{})".format(indent),
        ]
        block = block[:insert_at] + ds_lines + block[insert_at:]
        changed = True

    if changed:
        new_lines = lines[:start_idx] + block + lines[end_idx + 1 :]
        SYMBOL_LIB.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _top_level_symbol_ranges(lines):
    """Return [(start_idx, end_idx)] for top-level symbol blocks."""
    ranges = []
    depth = 0
    start = None
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if start is None and depth == 1 and stripped.startswith('(symbol "'):
            start = idx
        depth += line.count("(") - line.count(")")
        if start is not None and depth == 1:
            ranges.append((start, idx))
            start = None
    return ranges


def _extract_parent_symbol_blocks(lines):
    """
    Extract parent symbol blocks by scanning '(symbol "...")' entries.
    Child unit symbols matching *_<n>_<m> are skipped.
    """
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.lstrip().startswith('(symbol "'):
            i += 1
            continue

        m = re.search(r'\(symbol\s+"([^"]+)"', line)
        if not m:
            i += 1
            continue
        name = m.group(1)

        if re.search(r"_[0-9]+_[0-9]+$", name):
            i += 1
            continue

        start = i
        bal = 0
        while i < len(lines):
            bal += lines[i].count("(") - lines[i].count(")")
            i += 1
            if bal == 0:
                break
        blocks.append(lines[start:i])
    return blocks


def _extract_property_value(block_text: str, prop_name: str) -> str:
    m = re.search(r'\(property\s+"{}"\s+"([^"]*)"'.format(re.escape(prop_name)), block_text)
    return m.group(1).strip() if m else ""


def _make_meaningful_description(block_text: str) -> str:
    value = _extract_property_value(block_text, "Value")
    desc = _extract_property_value(block_text, "Description")
    datasheet = _extract_property_value(block_text, "Datasheet")
    lcsc = _extract_property_value(block_text, "LCSC")
    keywords = _extract_property_value(block_text, "ki_keywords")

    if desc:
        return desc
    if lcsc and value:
        return "{} (LCSC {})".format(value, lcsc)
    if keywords.startswith("C") and keywords[1:].isdigit() and value:
        return "{} (LCSC {})".format(value, keywords)
    if datasheet and value and datasheet != value:
        return "{} (see datasheet)".format(value)
    if value:
        return "Symbol for {}".format(value)
    return "Custom symbol"


def patch_ki_descriptions() -> None:
    """Ensure all parent symbols include a meaningful ki_description property."""
    lines = SYMBOL_LIB.read_text(encoding="utf-8").splitlines()
    first_symbol_idx = None
    for idx, line in enumerate(lines):
        if line.lstrip().startswith('(symbol "'):
            first_symbol_idx = idx
            break

    if first_symbol_idx is None:
        return

    header = lines[:first_symbol_idx]
    parent_blocks = _extract_parent_symbol_blocks(lines)
    rebuilt = header[:]

    for block in parent_blocks:
        block_text = "\n".join(block)
        new_desc = _make_meaningful_description(block_text).replace('"', "'")
        found = False

        for i, line in enumerate(block):
            if '(property "ki_description"' in line:
                block[i] = re.sub(
                    r'(\(property\s+"ki_description"\s+")([^"]*)(")',
                    lambda m: m.group(1) + new_desc + m.group(3),
                    line,
                )
                found = True
                break

        if not found:
            indent_match = re.match(r"^(\s*)", block[0])
            indent = (indent_match.group(1) if indent_match else "") + "  "
            insert_at = len(block) - 1
            for i, line in enumerate(block[1:], start=1):
                if line.lstrip().startswith('(symbol "'):
                    insert_at = i
                    break

            desc_lines = [
                '{}(property "ki_description" "{}" (id 97) (at 0 0 0)'.format(indent, new_desc),
                "{}  (effects (font (size 1.27 1.27)) hide)".format(indent),
                "{})".format(indent),
            ]
            block = block[:insert_at] + desc_lines + block[insert_at:]

        rebuilt.extend(block)

    rebuilt.append(")")
    SYMBOL_LIB.write_text("\n".join(rebuilt) + "\n", encoding="utf-8")


def patch_3d_model_paths() -> None:
    """Patch (model "...") lines in .kicad_mod files to use ${ROBOSUB_3D_MODELS}/filename."""
    for mod_path in FP_LIB_DIR.glob("*.kicad_mod"):
        mtext = mod_path.read_text(encoding="utf-8")

        def repl(match):
            orig_path = match.group(2)
            filename = os.path.basename(orig_path)
            return '{}${{{}}}/{}"'.format(match.group(1), MODELS_ENV, filename)

        new_mtext = re.sub(r'(\(model\s+")([^"]+)"', repl, mtext)

        if new_mtext != mtext:
            mod_path.write_text(new_mtext, encoding="utf-8")


def extract_symbol_info(symbol_src: Path, fp_basename: Optional[str]) -> Tuple[str, str, str]:
    """Return (symbol_name, footprint_string, description)."""
    src = symbol_src.read_text(encoding="utf-8")

    m_sym = re.search(r'\(symbol\s+"([^"]+)"', src)
    symbol_name = m_sym.group(1) if m_sym else ""

    m_desc = re.search(r'property\s+"Description"\s+"([^"]*)"', src)
    description = m_desc.group(1) if m_desc else ""

    footprint = ""
    if fp_basename:
        fp_name_noext = os.path.splitext(fp_basename)[0]
        footprint = "{}:{}".format(FP_LIB_NICKNAME, fp_name_noext)

    return symbol_name, footprint, description


def append_log(part: str, symbol_name: str, footprint: str, description: str) -> None:
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([part, symbol_name, footprint, description])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv) -> int:
    if len(argv) != 2:
        print_err("Usage: {} <JLC part-number (e.g. C22383822)>".format(Path(argv[0]).name))
        return 1

    part = argv[1].strip()

    if not part:
        print_err("ERROR: Part number cannot be empty.")
        return 1

    ensure_dirs_and_files()

    if already_processed(part):
        print("Part {} is already recorded in {}; nothing to do.".format(part, LOG_FILE))
        return 0

    symbol_name = ""
    footprint_str = ""
    description = ""

    with tempfile.TemporaryDirectory() as tmp_work_str:
        work_dir = Path(tmp_work_str) / "work"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Run JLC2KiCadLib
        run_jlc2kicadlib(part, work_dir)

        # Locate generated files
        symbol_src, fp_src_file, model_files = find_generated_files(work_dir)

        if symbol_src is None:
            print_err("ERROR: No .kicad_sym file produced by JLC2KiCadLib for {}".format(part))
            return 1

        # Read symbol name early so we can enforce coherent footprint/model basenames.
        symbol_name, _, description = extract_symbol_info(symbol_src, None)
        coherent_base = coherent_basename_from_symbol(symbol_name) if symbol_name else part

        # Copy footprint (if any), renamed to coherent symbol-based basename.
        fp_basename = None
        old_model_name = None
        new_model_name = None
        if fp_src_file is not None:
            FP_LIB_DIR.mkdir(parents=True, exist_ok=True)
            fp_dest = FP_LIB_DIR / "{}.kicad_mod".format(coherent_base)
            shutil.copy2(fp_src_file, fp_dest)
            old_model_name, new_model_name = rewrite_footprint_file_identity(fp_dest, coherent_base)
            fp_basename = fp_dest.name
        else:
            print("WARNING: No .kicad_mod footprint file found for this part.")

        # Copy 3D models (if any)
        for model_path in model_files:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            out_name = model_path.name
            if old_model_name and new_model_name and model_path.name == old_model_name:
                out_name = new_model_name
            dest = MODELS_DIR / out_name
            shutil.copy2(model_path, dest)

        # Ensure symbol footprint property follows the coherent basename.
        if fp_basename is not None:
            patch_symbol_source_footprint(symbol_src, os.path.splitext(fp_basename)[0])

        # Extract info for logging BEFORE temp dir is deleted
        symbol_name, footprint_str, description = extract_symbol_info(symbol_src, fp_basename)

        # Merge symbol(s) into main symbol library
        merge_symbol_into_master(symbol_src)

    # Patch properties and model paths in final libs
    patch_datasheet_for_symbol(symbol_name, part)
    patch_footprint_properties()
    patch_ki_descriptions()
    patch_3d_model_paths()

    # Log entry
    append_log(part, symbol_name, footprint_str, description)

    print("Merged JLC part {} into project-local KiCad libraries:".format(part))
    print("  Symbols   :", SYMBOL_LIB)
    print("  Footprints:", FP_LIB_DIR)
    print("  3D models :", MODELS_DIR)
    print("  Log       :", LOG_FILE)
    print()
    print("Re-run protection: if you call this again with {}".format(part))
    print("the script will detect it in the log and exit without changes.")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))