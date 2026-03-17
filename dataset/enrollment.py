"""Enrollment CLI commands for ``dataset process enrollment`` and ``dataset report enrollment``."""

from pathlib import Path
from typing import Annotated, Optional

from loguru import logger

from coursedata.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from coursedata.enrollment import find_roster_files, generate_enrollment_report, generate_enrollment_roster


def _rosters_impl(
    rosters_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """Core implementation for generating enrollment rosters."""
    if rosters_dir is None:
        rosters_dir = INTERIM_DATA_DIR / "albert" / "rosters"

    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR / "enrollment"

    logger.info(f"Finding roster files in {rosters_dir}")
    sections = find_roster_files(rosters_dir)

    if not sections:
        logger.warning(f"No roster files found in {rosters_dir}")
        return

    logger.info(f"Found {len(sections)} sections")

    for section_name, roster_files in sections.items():
        logger.info(f"Processing section: {section_name}")
        generate_enrollment_roster(section_name, roster_files, output_dir)

    logger.success("Enrollment roster generation complete")


def _reports_impl(
    rosters_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """Core implementation for generating enrollment reports."""
    if rosters_dir is None:
        rosters_dir = INTERIM_DATA_DIR / "albert" / "rosters"

    if output_dir is None:
        output_dir = REPORTS_DIR / "enrollment"

    logger.info(f"Finding roster files in {rosters_dir}")
    sections = find_roster_files(rosters_dir)

    if not sections:
        logger.warning(f"No roster files found in {rosters_dir}")
        return

    logger.info(f"Found {len(sections)} sections")

    for section_name, roster_files in sections.items():
        logger.info(f"Processing section: {section_name}")
        generate_enrollment_report(section_name, roster_files, output_dir)

    logger.success("Enrollment report generation complete")


def process_all() -> None:
    """Run all enrollment process commands."""
    _rosters_impl()


def report_all() -> None:
    """Run all enrollment report commands."""
    _reports_impl()
