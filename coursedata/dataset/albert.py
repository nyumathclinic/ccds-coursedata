"""Albert CLI commands for ``dataset get albert``."""

import json
import os
import shutil
from pathlib import Path
from typing import Annotated, Optional

from loguru import logger
from tqdm import tqdm
import typer

try:
    from edubag.albert import xls2csv
    from edubag.albert.client import AlbertClient
    EDUBAG_AVAILABLE = True
except ImportError:
    EDUBAG_AVAILABLE = False

from coursedata.config import ALBERT_CONFIG, COURSE_NAME, INTERIM_DATA_DIR, RAW_DATA_DIR, TERM_NAME
from ._utils import d8, get_password, get_sso_credentials

app = typer.Typer(help="Fetch data from Albert.")


def _albert_course_ids_and_term() -> tuple[Optional[list], str]:
    """Read per-course config from [tool.coursedata.albert] in pyproject.toml."""
    course_ids = ALBERT_CONFIG.get("courses")
    term = ALBERT_CONFIG.get("term_number") or TERM_NAME
    return course_ids, term


def _require_instructor_id() -> str:
    instructor_id = os.environ.get("ALBERT_INSTRUCTOR_ID")
    if not instructor_id:
        logger.error(
            "ALBERT_INSTRUCTOR_ID environment variable must be set to fetch by individual course number."
        )
        raise typer.Exit(code=1)
    return instructor_id


def _rosters_impl(
    output_dir: Optional[Path] = None,
    convert_to_csv: bool = True,
    csv_output_dir: Optional[Path] = None,
    clean: bool = False,
    headless: bool = False,
) -> None:
    """Core implementation for fetching Albert rosters."""
    if not EDUBAG_AVAILABLE:
        logger.error("edubag module is not available. Cannot fetch rosters.")
        raise typer.Exit(code=1)

    if output_dir is None:
        output_dir = RAW_DATA_DIR / "albert" / "rosters" / d8

    if csv_output_dir is None:
        csv_output_dir = INTERIM_DATA_DIR / "albert" / "rosters" / d8

    if clean:
        if output_dir.exists():
            logger.info(f"Cleaning output directory: {output_dir}")
            shutil.rmtree(output_dir)
        if convert_to_csv and csv_output_dir.exists():
            logger.info(f"Cleaning CSV output directory: {csv_output_dir}")
            shutil.rmtree(csv_output_dir)

    username, password = get_sso_credentials()
    client = AlbertClient()

    course_ids, term = _albert_course_ids_and_term()
    if course_ids:
        instructor_id = _require_instructor_id()
        logger.info(
            f"Fetching rosters for courses {course_ids} in term '{term}' to '{output_dir}'"
        )
        xls_path_list = [
            client.fetch_roster(
                class_number,
                term,
                instructor_id=instructor_id,
                save_dir=output_dir,
                username=username,
                password=password,
                headless=headless,
            )
            for class_number in tqdm(course_ids, desc="Fetching rosters")
        ]
    else:
        logger.info(
            f"Fetching rosters for course '{COURSE_NAME}' in term '{TERM_NAME}' to '{output_dir}'"
        )
        xls_path_list = client.fetch_and_save_rosters(
            COURSE_NAME,
            TERM_NAME,
            output_dir,
            username=username,
            password=password,
            headless=headless,
        )
    logger.success("Rosters fetched successfully.")
    if convert_to_csv:
        csv_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Converting Excel files to CSV in '{csv_output_dir}'")
        for xls_path in tqdm(xls_path_list, desc="Converting to CSV"):
            xls2csv([xls_path], csv_output_dir)
        logger.success("Conversion to CSV complete.")


def _class_details_impl(
    output: Optional[Path] = None,
    headless: bool = False,
) -> None:
    """Core implementation for fetching Albert class details."""
    if not EDUBAG_AVAILABLE:
        logger.error("edubag module is not available. Cannot fetch class details.")
        raise typer.Exit(code=1)

    if output is None:
        output = RAW_DATA_DIR / "albert" / "class_details" / d8 / "class_details.json"

    username, password = get_sso_credentials()
    client = AlbertClient()

    course_ids, term = _albert_course_ids_and_term()
    if course_ids:
        instructor_id = _require_instructor_id()
        logger.info(
            f"Fetching class details for courses {course_ids} in term '{term}' to '{output}'"
        )
        details = [
            client.fetch_course_details(
                class_number,
                term,
                instructor_id=instructor_id,
                username=username,
                password=password,
                headless=headless,
            )
            for class_number in tqdm(course_ids, desc="Fetching class details")
        ]
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(details, f, indent=2)
    else:
        logger.info(
            f"Fetching class details for course '{COURSE_NAME}' in term '{TERM_NAME}' to '{output}'"
        )
        client.fetch_class_details(
            COURSE_NAME,
            TERM_NAME,
            output=output,
            username=username,
            password=password,
            headless=headless,
        )
    logger.success("Class details fetched successfully.")


def run_all(headless: bool = False) -> None:
    """Run all Albert fetch commands."""
    _rosters_impl(headless=headless)
    _class_details_impl(headless=headless)


@app.callback(invoke_without_command=True)
def albert_callback(ctx: typer.Context) -> None:
    """Fetch all Albert data. Run without a subcommand to fetch everything."""
    if ctx.invoked_subcommand is None:
        headless = (ctx.obj or {}).get("headless", False)
        run_all(headless=headless)


@app.command()
def rosters(
    ctx: typer.Context,
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Output directory for the rosters file")
    ] = None,
    convert_to_csv: Annotated[
        bool, typer.Option(help="Convert the fetched Excel files to CSV format")
    ] = True,
    csv_output_dir: Annotated[
        Optional[Path], typer.Option(help="Output directory for CSV files")
    ] = None,
    clean: Annotated[
        bool,
        typer.Option(help="Remove existing files in output directories before fetching"),
    ] = False,
) -> None:
    """Fetch all rosters for the specified course and term, and save to output_dir."""
    headless = (ctx.obj or {}).get("headless", False)
    _rosters_impl(
        output_dir=output_dir,
        convert_to_csv=convert_to_csv,
        csv_output_dir=csv_output_dir,
        clean=clean,
        headless=headless,
    )


@app.command("class-details")
def class_details(
    ctx: typer.Context,
    output: Annotated[
        Optional[Path], typer.Option(help="Output path for the class details file")
    ] = None,
) -> None:
    """Fetch all class details for the specified course and term, and save to output."""
    headless = (ctx.obj or {}).get("headless", False)
    _class_details_impl(output=output, headless=headless)
