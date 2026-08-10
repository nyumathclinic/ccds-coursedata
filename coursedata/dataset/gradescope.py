"""Gradescope CLI commands for ``dataset get gradescope``."""

import os
import shutil
from pathlib import Path
from typing import Annotated, Optional

from loguru import logger
import typer

try:
    from edubag.gradescope.client import GradescopeClient
    EDUBAG_AVAILABLE = True
except ImportError:
    EDUBAG_AVAILABLE = False

from coursedata.config import COURSE_NAME, GRADESCOPE_CONFIG, RAW_DATA_DIR, TERM_NAME
from ._utils import d8, get_password

app = typer.Typer(help="Fetch data from Gradescope.")


def _get_gradescope_credentials() -> tuple[Optional[str], Optional[str]]:
    """Get Gradescope username and password from environment and keychain."""
    username = os.getenv("GRADESCOPE_USERNAME")
    if not username:
        logger.warning(
            "GRADESCOPE_USERNAME not found in environment variables. Set it in your .env file."
        )
        username = None

    password = None
    if username:
        password = get_password("gradescope.com", username)
        if not password:
            logger.warning(
                f"Password for user '{username}' not found in macOS Keychain. "
                f"Store it with: security add-generic-password -s gradescope.com -a {username} -w YOUR_PASSWORD"
            )

    return username, password


def _class_details_impl(
    output: Optional[Path] = None,
    headless: bool = False,
) -> None:
    """Core implementation for fetching Gradescope class details."""
    if not EDUBAG_AVAILABLE:
        logger.error(
            "edubag module is not available. Cannot fetch Gradescope class details."
        )
        raise typer.Exit(code=1)

    if output is None:
        output = RAW_DATA_DIR / "gradescope" / "class_details" / "class_details.json"

    logger.info(
        f"Fetching Gradescope class details for course '{COURSE_NAME}' in term '{TERM_NAME}' to '{output}'"
    )

    username, password = _get_gradescope_credentials()

    client = GradescopeClient()
    client.fetch_class_details(
        COURSE_NAME, TERM_NAME, output=output, username=username, password=password
    )
    logger.success("Gradescope class details fetched successfully.")


def _rosters_impl(
    output_dir: Optional[Path] = None,
    clean: bool = False,
    headless: bool = False,
) -> None:
    """Core implementation for fetching Gradescope rosters."""
    if not EDUBAG_AVAILABLE:
        logger.error("edubag module is not available. Cannot fetch Gradescope rosters.")
        raise typer.Exit(code=1)

    course_ids = GRADESCOPE_CONFIG.get("courses", [])
    if not course_ids:
        logger.error("No Gradescope course IDs found in configuration.")
        raise typer.Exit(code=1)

    if output_dir is None:
        output_dir = RAW_DATA_DIR / "gradescope" / "rosters" / d8

    if clean and output_dir.exists():
        logger.info(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)

    logger.info(
        f"Fetching Gradescope rosters for courses {course_ids} to '{output_dir}'"
    )

    username, password = _get_gradescope_credentials()

    try:
        client = GradescopeClient()
        client.authenticate(username=username, password=password, headless=headless)
    except Exception as e:
        logger.error(f"Gradescope authentication failed: {e}")
        raise typer.Exit(code=1)

    for course in course_ids:
        try:
            client.save_roster(course, save_dir=output_dir, headless=headless)
        except Exception as e:
            logger.error(f"Failed to fetch roster for course {course}: {e}")
            raise typer.Exit(code=1)
    logger.success("Gradescope rosters fetched successfully.")


def run_all(headless: bool = False) -> None:
    """Run all Gradescope fetch commands."""
    _class_details_impl(headless=headless)
    _rosters_impl(headless=headless)


@app.callback(invoke_without_command=True)
def gradescope_callback(ctx: typer.Context) -> None:
    """Fetch all Gradescope data. Run without a subcommand to fetch everything."""
    if ctx.invoked_subcommand is None:
        headless = (ctx.obj or {}).get("headless", False)
        run_all(headless=headless)


@app.command("class-details")
def class_details(
    ctx: typer.Context,
    output: Annotated[
        Optional[Path], typer.Option(help="Output path for the class details file")
    ] = None,
) -> None:
    """Fetch all Gradescope class details for the specified course and term, and save to output."""
    headless = (ctx.obj or {}).get("headless", False)
    _class_details_impl(output=output, headless=headless)


@app.command()
def rosters(
    ctx: typer.Context,
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Output directory for Gradescope rosters")
    ] = None,
    clean: Annotated[
        bool,
        typer.Option(help="Remove existing files in output directories before fetching"),
    ] = False,
) -> None:
    """Fetch Gradescope rosters for configured courses and save to output_dir."""
    headless = (ctx.obj or {}).get("headless", False)
    _rosters_impl(output_dir=output_dir, clean=clean, headless=headless)
