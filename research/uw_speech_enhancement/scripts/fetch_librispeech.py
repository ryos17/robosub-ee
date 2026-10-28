"""Download and extract LibriSpeech subsets from OpenSLR 12 (RESEARCH_PLAN.md 5.1).

Usage::

    python scripts/fetch_librispeech.py dev-clean test-clean            # into data/librispeech
    python scripts/fetch_librispeech.py train-clean-100 --out D:/corpora/librispeech
    python scripts/fetch_librispeech.py dev-clean --dry-run

Archives land in ``<out>/archives/`` and extract to ``<out>/LibriSpeech/<subset>/``.
MD5 and SHA-256 of every archive are printed; the MD5 is compared with the
value published in OpenSLR's md5sum.txt (copied below; re-check against
https://www.openslr.org/resources/12/md5sum.txt if a mismatch is reported).
Extraction uses tarfile's "data" filter (no absolute paths, no links out of
the target directory). Licence: CC BY 4.0.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

MIRRORS = (
    "https://www.openslr.org/resources/12",
    "https://us.openslr.org/resources/12",
    "https://openslr.elda.org/resources/12",
)

# subset -> (OpenSLR md5, approximate archive size in GB)
SUBSETS: dict[str, tuple[str, float]] = {
    "dev-clean": ("42e2234ba48799c1f50f24a7926300a1", 0.34),
    "test-clean": ("32fa31d27d2e1cad72775fee3f4849a9", 0.35),
    "train-clean-100": ("2a93770f6d5c6c964bc36631d331a522", 6.4),
    "train-clean-360": ("c0e676e450a7ff2f54aeade5171606fa", 23.0),
}

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "librispeech"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("subsets", nargs="+", choices=sorted(SUBSETS), help="subsets to fetch")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="target directory")
    ap.add_argument("--mirror", default=MIRRORS[0], help=f"base URL (known: {', '.join(MIRRORS)})")
    ap.add_argument("--skip-verify", action="store_true", help="extract even if the MD5 differs")
    ap.add_argument("--keep-archive", action="store_true", help="keep the .tar.gz after extraction")
    ap.add_argument("--no-extract", action="store_true", help="download and verify only")
    ap.add_argument("--dry-run", action="store_true", help="print what would be done and exit")
    args = ap.parse_args(argv)
    args.subsets = list(dict.fromkeys(args.subsets))
    parts = args.out.resolve().parts
    if len(parts) > 1 and parts[1].strip("/\\").lower() == "datasets":
        ap.error("refusing to write under /datasets (read-only by policy)")
    return args


def url_for(subset: str, mirror: str = MIRRORS[0]) -> str:
    return f"{mirror.rstrip('/')}/{subset}.tar.gz"


def hashes(path: Path, chunk: int = 1 << 20) -> tuple[str, str]:
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            md5.update(b)
            sha.update(b)
    return md5.hexdigest(), sha.hexdigest()


def download(url: str, dst: Path) -> None:
    from tqdm import tqdm

    tmp = dst.with_suffix(dst.suffix + ".part")
    got = 0
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=dst.name) as bar:
            while b := r.read(1 << 20):
                f.write(b)
                got += len(b)
                bar.update(len(b))
    # A dropped connection ends read() early; never promote a short file to the
    # final name (a later run would skip the download and fail the MD5 forever).
    if total is not None and got != total:
        raise IOError(f"{url}: received {got} of {total} bytes; partial file left at {tmp}")
    tmp.replace(dst)


def extract(archive: Path, out: Path) -> None:
    with tarfile.open(archive, "r:gz") as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(out, filter="data")
        else:  # Python < 3.11.4: reject unsafe members ourselves
            root = out.resolve()
            for m in tf.getmembers():
                if m.issym() or m.islnk() or not (root / m.name).resolve().is_relative_to(root):
                    raise RuntimeError(f"unsafe member in archive: {m.name}")
            tf.extractall(out)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    need_gb = sum(SUBSETS[s][1] for s in args.subsets) * 2.2  # archive + extracted copy
    print(f"target: {args.out}")
    for s in args.subsets:
        print(f"  {s:16s} {url_for(s, args.mirror)}  md5 {SUBSETS[s][0]}  ~{SUBSETS[s][1]} GB")
    if args.dry_run:
        print(f"dry run; about {need_gb:.1f} GB free space needed")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(args.out).free / 1e9
    if free_gb < need_gb:
        print(f"only {free_gb:.1f} GB free, need about {need_gb:.1f} GB", file=sys.stderr)
        return 2
    arch_dir = args.out / "archives"
    arch_dir.mkdir(exist_ok=True)
    status = 0
    for s in args.subsets:
        if (args.out / "LibriSpeech" / s).is_dir():
            print(f"{s}: already extracted, skipping")
            continue
        archive = arch_dir / f"{s}.tar.gz"
        if not archive.exists():
            download(url_for(s, args.mirror), archive)
        md5, sha = hashes(archive)
        ok = md5 == SUBSETS[s][0]
        print(f"{s}: md5 {md5} ({'OK' if ok else 'MISMATCH, expected ' + SUBSETS[s][0]})")
        print(f"{s}: sha256 {sha}")
        if not ok and not args.skip_verify:
            print(f"{s}: not extracted; delete {archive} to re-download", file=sys.stderr)
            status = 1
            continue
        if not args.no_extract:
            extract(archive, args.out)
            print(f"{s}: extracted to {args.out / 'LibriSpeech' / s}")
            if not args.keep_archive:
                archive.unlink()
    return status


if __name__ == "__main__":
    sys.exit(main())
