"""EdStem CLI commands for ``dataset get edstem``."""

import os
import shutil
from pathlib import Path
from typing import Annotated, Optional

from loguru import logger
import typer

try:
    from edubag.edstem.client import EdstemClient
    EDSTEM_AVAILABLE = True
except ImportError:
    EDSTEM_AVAILABLE = False

from coursedata.config import EDSTEM_CONFIG, RAW_DATA_DIR
from ._utils import d8, get_password

app = typer.Typer(help="Fetch data from EdStem.")


def _get_edstem_credentials() -> tuple[Optional[str], Optional[str]]:
    """Get EdStem username and password from environment and keychain."""
    username = os.getenv("EDSTEM_USERNAME")
    if not username:
        logger.warning(
            "EDSTEM_USERNAME not found in environment variables. Set it in your .env file."
        )
        username = None

    password = None
    if username:
        password = get_password("edstem.org", username)
        if not password:
            logger.warning(
                f"Password for user '{username}' not found in macOS Keychain. "
                f"Store it with: security add-generic-password -s edstem.org -a {username} -w YOUR_PASSWORD"
            )

    return username, password


def _analytics_impl(
    output_dir: Optional[Path] = None,
    clean: bool = False,
    headless: bool = False,
) -> None:
    """Core implementation for fetching EdStem analytics."""
    if not EDSTEM_AVAILABLE:
        logger.error("edubag edstem client is not available. Cannot fetch EdStem analytics.")
        raise typer.Exit(code=1)

    course_ids = EDSTEM_CONFIG.get("courses", [])
    if not course_ids:
        logger.error("No EdStem course IDs found in configuration.")
        raise typer.Exit(code=1)

    if output_dir is None:
        output_dir = RAW_DATA_DIR / "edstem" / "analytics" / d8

    if clean and output_dir.exists():
        logger.info(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)

    logger.info(
        f"Fetching EdStem analytics for courses {course_ids} to '{output_dir}'"
    )

    username, password = _get_edstem_credentials()

    try:
        client = EdstemClient()
        client.authenticate(username=username, password=password, headless=headless)
    except Exception as e:
        logger.error(f"EdStem authentication failed: {e}")
        raise typer.Exit(code=1)

    for course in course_ids:
        try:
            client.save_analytics(
                course,
                save_dir=output_dir,
                headless=headless,
            )
        except Exception as e:
            logger.error(f"Failed to fetch analytics for course {course}: {e}")
            raise typer.Exit(code=1)
    logger.success("EdStem analytics fetched successfully.")


def run_all(headless: bool = False) -> None:
    """Run all EdStem fetch commands."""
    _analytics_impl(headless=headless)


@app.callback(invoke_without_command=True)
def edstem_callback(ctx: typer.Context) -> None:
    """Fetch all EdStem data. Run without a subcommand to fetch everything."""
    if ctx.invoked_subcommand is None:
        headless = (ctx.obj or {}).get("headless", False)
        run_all(headless=headless)


@app.command()
def analytics(
    ctx: typer.Context,
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Output directory for EdStem analytics")
    ] = None,
    clean: Annotated[
        bool,
        typer.Option(help="Remove existing files in output directories before fetching"),
    ] = False,
) -> None:
    """Fetch EdStem analytics for configured courses and save to output_dir."""
    headless = (ctx.obj or {}).get("headless", False)
    _analytics_impl(output_dir=output_dir, clean=clean, headless=headless)
