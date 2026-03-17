"""Gmail CLI commands for ``dataset process gmail-filters``."""

from pathlib import Path
from typing import Optional

from loguru import logger
import typer

try:
    from edubag.gmail import filter_from_roster_command
    EDUBAG_AVAILABLE = True
except ImportError:
    EDUBAG_AVAILABLE = False

from coursedata.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


def _filters_impl(
    roster_paths: Optional[list[Path]] = None,
    output: Optional[Path] = None,
) -> None:
    """Core implementation for generating Gmail filters."""
    if not EDUBAG_AVAILABLE:
        logger.error("edubag module is not available. Cannot generate Gmail filters.")
        raise typer.Exit(code=1)

    if not roster_paths:
        logger.info("No roster files provided. Using most recently downloaded rosters.")
        rosters_base_dir = RAW_DATA_DIR / "albert" / "rosters"

        date_dirs = sorted([d for d in rosters_base_dir.iterdir() if d.is_dir()])
        if not date_dirs:
            logger.error(f"No roster directories found in {rosters_base_dir}")
            raise typer.Exit(code=1)

        latest_date_dir = date_dirs[-1]
        logger.info(f"Using rosters from {latest_date_dir.name}")

        roster_paths = sorted(latest_date_dir.glob("*.XLS"))

        if not roster_paths:
            logger.error(f"No .XLS files found in {latest_date_dir}")
            raise typer.Exit(code=1)

    if not output:
        output = PROCESSED_DATA_DIR / "gmail" / "gmail_filters.xml"

    filter_from_roster_command(roster_paths, output=output)
    logger.success(f"Gmail filters saved to {output}")


def process_all() -> None:
    """Run all Gmail process commands."""
    _filters_impl()
