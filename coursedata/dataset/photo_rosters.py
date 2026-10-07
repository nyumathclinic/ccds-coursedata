"""Photo roster processing: Albert HTML rosters -> JSON + Anki decks.

Commands are wired into:
    python -m coursedata.dataset process photo-rosters

Only the latest version of each roster is kept: each run replaces
``PROCESSED_DATA_DIR/albert/rosters/latest/<class number>/``.
"""

from pathlib import Path
import shutil

from loguru import logger

try:
    from edubag.albert.exports import write_anki_deck
    from edubag.albert.roster import AlbertRoster
    EDUBAG_AVAILABLE = True
except ImportError:
    EDUBAG_AVAILABLE = False

from coursedata.config import ALBERT_CONFIG, PROCESSED_DATA_DIR, RAW_DATA_DIR

PHOTO_ROSTERS_RAW_DIR = RAW_DATA_DIR / "albert" / "rosters" / "latest"
PHOTO_ROSTERS_PROCESSED_DIR = PROCESSED_DATA_DIR / "albert" / "rosters" / "latest"


def _process_one(html_path: Path, output_dir: Path) -> None:
    """Write ``<pathstem>.json`` (plus photos) and ``<pathstem>.apkg`` for one roster."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    roster = AlbertRoster.from_html(html_path)
    json_path = roster.to_json(output_dir / f"{roster.pathstem}.json")
    # Build the deck from the JSON copy so the photos it embeds are the processed ones.
    apkg_path = write_anki_deck(AlbertRoster.from_json(json_path), json_path.with_suffix(".apkg"))
    missing = int(roster.students["photo"].isna().sum())
    logger.info(
        f"{roster.pathstem}: {len(roster.students)} students ({missing} without photo) "
        f"-> {json_path.name}, {apkg_path.name}"
    )


def process_all() -> None:
    """Convert the latest photo roster of each configured class number."""
    if not EDUBAG_AVAILABLE:
        logger.error("edubag module is not available. Cannot process photo rosters.")
        return

    class_numbers = [str(n) for n in ALBERT_CONFIG.get("photo_rosters", [])]
    if not class_numbers:
        logger.info("No [tool.coursedata.albert].photo_rosters configured; skipping.")
        return

    for class_number in class_numbers:
        html_paths = sorted(
            (PHOTO_ROSTERS_RAW_DIR / class_number).glob("*.html"),
            key=lambda p: p.stat().st_mtime,
        )
        if not html_paths:
            logger.warning(
                f"No photo roster HTML for class number {class_number} in "
                f"{PHOTO_ROSTERS_RAW_DIR / class_number}; run `get albert photo-rosters`."
            )
            continue
        _process_one(html_paths[-1], PHOTO_ROSTERS_PROCESSED_DIR / class_number)
    logger.success(f"Photo rosters processed into {PHOTO_ROSTERS_PROCESSED_DIR}")
