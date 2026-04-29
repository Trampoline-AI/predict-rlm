"""Extract SpreadsheetBench archives and build the trainset/testset folders.

Extracts both archives under ``data/``, then creates:

* ``data/trainset/`` — the 912-set minus the verified 400-set (512 entries).
  Spreadsheet directories are symlinked into ``all_data_912_v0.1/spreadsheet/``
  to avoid duplicating on-disk data.
* ``data/testset`` — a directory symlink pointing at the verified 400-set,
  so callers can reference ``data/trainset`` and ``data/testset`` uniformly.
"""

from __future__ import annotations

import json
import logging
import shutil
import tarfile
from collections import Counter
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = BENCH_DIR.parent.parent
DATA_DIR = EXAMPLE_DIR / "data"

ARCHIVES: dict[str, str] = {
    "spreadsheetbench_verified_400.tar.gz": "spreadsheetbench_verified_400",
    "spreadsheetbench_912_v0.1.tar.gz": "all_data_912_v0.1",
}

VERIFIED_DIR = DATA_DIR / "spreadsheetbench_verified_400"
FULL_DIR = DATA_DIR / "all_data_912_v0.1"
TRAINSET_DIR = DATA_DIR / "trainset"
TESTSET_LINK = DATA_DIR / "testset"

log = logging.getLogger("spreadbench.dataset")


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TiB"


def extract_archive(archive: Path, expected_top_dir: Path) -> None:
    if expected_top_dir.exists():
        log.info("skip extract: %s already present", expected_top_dir.name)
        return
    if not archive.exists():
        raise FileNotFoundError(f"missing archive: {archive}")
    log.info(
        "extracting %s (%s) -> %s",
        archive.name,
        _human_size(archive.stat().st_size),
        DATA_DIR,
    )
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        tar.extractall(DATA_DIR, filter="data")
    log.info("extracted %d members from %s", len(members), archive.name)


def _summarize(label: str, entries: list[dict]) -> None:
    kinds = Counter(e.get("instruction_type", "<none>") for e in entries)
    log.info("%s: %d entries", label, len(entries))
    for kind, count in sorted(kinds.items(), key=lambda kv: (-kv[1], kv[0])):
        log.info("  %-32s %4d", kind, count)


def build_trainset() -> None:
    log.info("building trainset (912-set minus verified 400-set)")
    verified = json.loads((VERIFIED_DIR / "dataset.json").read_text())
    full = json.loads((FULL_DIR / "dataset.json").read_text())

    verified_ids = {str(entry["id"]) for entry in verified}
    full_ids = {str(entry["id"]) for entry in full}

    missing_from_full = verified_ids - full_ids
    if missing_from_full:
        raise RuntimeError(
            f"verified set contains ids not present in full set: "
            f"{sorted(missing_from_full)[:5]}..."
        )

    train = [entry for entry in full if str(entry["id"]) not in verified_ids]

    _summarize("full 912-set", full)
    _summarize("verified 400-set (held out)", verified)
    _summarize("trainset (filtered)", train)

    filtered_count = len(full) - len(train)
    log.info("filter criteria: drop entry if str(entry['id']) is in the verified 400-set")
    log.info(
        "filter result: %d / %d entries dropped (matched verified id), %d kept",
        filtered_count,
        len(full),
        len(train),
    )
    sample_dropped = sorted(full_ids & verified_ids)[:5]
    sample_kept = [str(e["id"]) for e in train[:5]]
    log.info("  sample dropped ids: %s", sample_dropped)
    log.info("  sample kept ids:    %s", sample_kept)
    assert filtered_count == len(verified), (
        f"expected to drop {len(verified)} entries, dropped {filtered_count}"
    )

    if TRAINSET_DIR.exists():
        log.info("removing stale trainset at %s", TRAINSET_DIR)
        shutil.rmtree(TRAINSET_DIR)
    TRAINSET_DIR.mkdir(parents=True)

    dataset_path = TRAINSET_DIR / "dataset.json"
    dataset_path.write_text(
        json.dumps(train, indent=4, ensure_ascii=False) + "\n"
    )
    log.info(
        "wrote %s (%s)",
        dataset_path.relative_to(EXAMPLE_DIR),
        _human_size(dataset_path.stat().st_size),
    )

    spreadsheet_dir = TRAINSET_DIR / "spreadsheet"
    spreadsheet_dir.mkdir()
    linked = 0
    for entry in train:
        entry_id = str(entry["id"])
        source = FULL_DIR / "spreadsheet" / entry_id
        if not source.is_dir():
            raise FileNotFoundError(f"missing spreadsheet dir: {source}")
        link = spreadsheet_dir / entry_id
        relative_target = Path("..") / ".." / FULL_DIR.name / "spreadsheet" / entry_id
        link.symlink_to(relative_target, target_is_directory=True)
        linked += 1
    log.info(
        "linked %d spreadsheet dirs into %s",
        linked,
        spreadsheet_dir.relative_to(EXAMPLE_DIR),
    )


def build_testset() -> None:
    log.info("building testset (alias for verified 400-set)")
    if TESTSET_LINK.is_symlink() or TESTSET_LINK.exists():
        log.info("removing stale testset link at %s", TESTSET_LINK)
        if TESTSET_LINK.is_symlink() or TESTSET_LINK.is_file():
            TESTSET_LINK.unlink()
        else:
            shutil.rmtree(TESTSET_LINK)
    TESTSET_LINK.symlink_to(
        Path(VERIFIED_DIR.name), target_is_directory=True
    )
    verified = json.loads((VERIFIED_DIR / "dataset.json").read_text())
    log.info(
        "linked %s -> %s (%d entries)",
        TESTSET_LINK.relative_to(EXAMPLE_DIR),
        VERIFIED_DIR.name,
        len(verified),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log.info("data dir: %s", DATA_DIR)
    for archive_name, top_dir in ARCHIVES.items():
        extract_archive(DATA_DIR / archive_name, DATA_DIR / top_dir)
    build_trainset()
    build_testset()
    log.info("done")


if __name__ == "__main__":
    main()
