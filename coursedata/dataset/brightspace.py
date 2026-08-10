"""Brightspace CLI commands for ``dataset get brightspace``."""

import os
import shutil
from pathlib import Path
from typing import Annotated, Optional

from loguru import logger
import typer

try:
    from edubag.brightspace.client import BrightspaceClient
    BRIGHTSPACE_AVAILABLE = True
except ImportError:
    BRIGHTSPACE_AVAILABLE = False

from coursedata.config import BRIGHTSPACE_CONFIG, RAW_DATA_DIR
from ._utils import d8, get_password, get_sso_credentials

app = typer.Typer(help="Fetch data from Brightspace.")


def _gradebooks_impl(
    output_dir: Optional[Path] = None,
    clean: bool = False,
    headless: bool = False,
) -> None:
    """Core implementation for fetching Brightspace gradebooks."""
    if not BRIGHTSPACE_AVAILABLE:
        logger.error("edubag brightspace client is not available. Cannot fetch gradebooks.")
        raise typer.Exit(code=1)

    course_ids = BRIGHTSPACE_CONFIG.get("courses", [])
    if not course_ids:
        logger.error("No Brightspace course IDs found in configuration.")
        raise typer.Exit(code=1)

    if output_dir is None:
        output_dir = RAW_DATA_DIR / "brightspace" / "gradebooks" / d8

    if clean and output_dir.exists():
        logger.info(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)

    logger.info(
        f"Fetching Brightspace gradebooks for courses {course_ids} to '{output_dir}'"
    )

    username, password = get_sso_credentials()

    try:
        client = BrightspaceClient()
        client.authenticate(username=username, password=password, headless=headless)
    except Exception as e:
        logger.error(f"Brightspace authentication failed: {e}")
        raise typer.Exit(code=1)

    for course in course_ids:
        try:
            client.save_gradebook(course, save_dir=output_dir, headless=headless)
        except Exception as e:
            logger.error(f"Failed to fetch gradebook for course {course}: {e}")
            raise typer.Exit(code=1)
    logger.success("Brightspace gradebooks fetched successfully.")


def _attendance_impl(
    output_dir: Optional[Path] = None,
    clean: bool = False,
    headless: bool = False,
) -> None:
    """Core implementation for fetching Brightspace attendance files."""
    if not BRIGHTSPACE_AVAILABLE:
        logger.error("edubag brightspace client is not available. Cannot fetch attendance.")
        raise typer.Exit(code=1)

    course_ids = BRIGHTSPACE_CONFIG.get("courses", [])
    if not course_ids:
        logger.error("No Brightspace course IDs found in configuration.")
        raise typer.Exit(code=1)

    if output_dir is None:
        output_dir = RAW_DATA_DIR / "brightspace" / "attendance" / d8

    if clean and output_dir.exists():
        logger.info(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)

    logger.info(
        f"Fetching Brightspace attendance for courses {course_ids} to '{output_dir}'"
    )

    username, password = get_sso_credentials()

    try:
        client = BrightspaceClient()
        client.authenticate(username=username, password=password, headless=headless)
    except Exception as e:
        logger.error(f"Brightspace authentication failed: {e}")
        raise typer.Exit(code=1)

    for course in course_ids:
        try:
            client.save_attendance(course, save_dir=output_dir, headless=headless)
        except Exception as e:
            logger.error(f"Failed to fetch attendance for course {course}: {e}")
            raise typer.Exit(code=1)
    logger.success("Brightspace attendance fetched successfully.")


def run_all(headless: bool = False) -> None:
    """Run all Brightspace fetch commands."""
    _gradebooks_impl(headless=headless)
    _attendance_impl(headless=headless)


@app.callback(invoke_without_command=True)
def brightspace_callback(ctx: typer.Context) -> None:
    """Fetch all Brightspace data. Run without a subcommand to fetch everything."""
    if ctx.invoked_subcommand is None:
        headless = (ctx.obj or {}).get("headless", False)
        run_all(headless=headless)


@app.command()
def gradebooks(
    ctx: typer.Context,
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Output directory for Brightspace gradebooks")
    ] = None,
    clean: Annotated[
        bool,
        typer.Option(help="Remove existing files in output directories before fetching"),
    ] = False,
) -> None:
    """Fetch Brightspace gradebooks for configured courses and save to output_dir."""
    headless = (ctx.obj or {}).get("headless", False)
    _gradebooks_impl(output_dir=output_dir, clean=clean, headless=headless)


@app.command()
def attendance(
    ctx: typer.Context,
    output_dir: Annotated[
        Optional[Path],
        typer.Option(help="Output directory for Brightspace attendance files"),
    ] = None,
    clean: Annotated[
        bool,
        typer.Option(help="Remove existing files in output directories before fetching"),
    ] = False,
) -> None:
    """Fetch Brightspace attendance files for configured courses and save to output_dir."""
    headless = (ctx.obj or {}).get("headless", False)
    _attendance_impl(output_dir=output_dir, clean=clean, headless=headless)
