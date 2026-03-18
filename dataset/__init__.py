"""Course dataset CLI.

Command groups:
  get      Fetch raw data from external sources into RAW_DATA_DIR.
  process  Process raw/interim data into PROCESSED_DATA_DIR.
  report   Generate reports and save to REPORTS_DIR.

Use ``dataset <group> --help`` for details on each group.
"""

from pathlib import Path
from typing import Annotated, Optional

from loguru import logger
from rich.progress import Progress, SpinnerColumn, TextColumn
import typer

from . import albert as _albert
from . import brightspace as _brightspace
from . import edstem as _edstem
from . import engagement as _engagement
from . import enrollment as _enrollment
from . import gmail as _gmail
from . import progress as _progress
from . import gradescope as _gradescope

app = typer.Typer(
    help="Course data management CLI.",
    no_args_is_help=True,
)

# ---------------------------------------------------------------------------
# get group
# ---------------------------------------------------------------------------

get_app = typer.Typer(
    help="Fetch raw data from external sources into RAW_DATA_DIR.",
    no_args_is_help=False,
)


@get_app.callback(invoke_without_command=True)
def get_callback(
    ctx: typer.Context,
    headless: Annotated[
        bool,
        typer.Option(
            "--headless/--headed",
            help="Run browser headless (for automation) or headed (for debugging).",
        ),
    ] = False,
) -> None:
    """Fetch raw data from all sources. Run without a subcommand to fetch everything."""
    ctx.ensure_object(dict)
    ctx.obj["headless"] = headless
    if ctx.invoked_subcommand is None:
        sources = [
            ("albert", _albert.run_all),
            ("brightspace", _brightspace.run_all),
            ("gradescope", _gradescope.run_all),
            ("edstem", _edstem.run_all),
        ]
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            for name, run_fn in sources:
                task = progress.add_task(f"Fetching {name} data…", total=None)
                run_fn(headless=headless)
                progress.remove_task(task)


get_app.add_typer(_albert.app, name="albert")
get_app.add_typer(_brightspace.app, name="brightspace")
get_app.add_typer(_gradescope.app, name="gradescope")
get_app.add_typer(_edstem.app, name="edstem")

app.add_typer(get_app, name="get")

# ---------------------------------------------------------------------------
# process group
# ---------------------------------------------------------------------------

process_app = typer.Typer(
    help="Process raw/interim data into PROCESSED_DATA_DIR.",
    no_args_is_help=False,
)


@process_app.callback(invoke_without_command=True)
def process_callback(ctx: typer.Context) -> None:
    """Process all data. Run without a subcommand to run everything."""
    if ctx.invoked_subcommand is None:
        commands = [
            ("enrollment rosters", _enrollment.process_all),
            ("gmail filters", _gmail.process_all),
        ]
        commands.append(("engagement scores", _engagement.process_all))
        commands.append(("midterm progress report", _progress.process_all))
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            for name, run_fn in commands:
                task = progress.add_task(f"Processing {name}…", total=None)
                run_fn()
                progress.remove_task(task)


@process_app.command("enrollment")
def process_enrollment(
    rosters_dir: Annotated[
        Optional[Path],
        typer.Option(help="Directory containing dated roster subdirectories"),
    ] = None,
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Output directory for enrollment rosters")
    ] = None,
) -> None:
    """Generate enrollment rosters for all sections."""
    _enrollment._rosters_impl(rosters_dir=rosters_dir, output_dir=output_dir)


@process_app.command("gmail-filters")
def process_gmail_filters(
    roster_paths: Annotated[
        Optional[list[Path]],
        typer.Option(
            help="One or more Albert roster XLS files. If not set, the most recently downloaded rosters will be used."
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to save the Gmail filter XML file. If not set, save to processed data directory."
        ),
    ] = None,
) -> None:
    """Generate Gmail filters XML from Albert roster XLS files."""
    _gmail._filters_impl(roster_paths=roster_paths, output=output)


app.add_typer(process_app, name="process")


@process_app.command("engagement")
def process_engagement_cmd(
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output CSV path (overrides configured output_path)."),
    ] = None,
    keep_source_columns: Annotated[
        Optional[bool],
        typer.Option("--keep-source-cols/--no-source-cols", help="Include per-source columns in output."),
    ] = None,
    report: Annotated[
        bool,
        typer.Option("--report/--no-report", help="Print and save a validation report."),
    ] = True,
    report_dir: Annotated[
        Optional[Path],
        typer.Option(help="Directory for report files (overrides configured report_path)."),
    ] = None,
) -> None:
    """Compute engagement scores and save to PROCESSED_DATA_DIR."""
    _engagement._process_impl(
        output_path=output,
        keep_source_columns=keep_source_columns,
        with_report=report,
        report_dir=report_dir,
    )


@process_app.command("midterm-progress")
def process_midterm_progress_cmd(
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output CSV path override."),
    ] = None,
) -> None:
    """Generate a syllabus-aligned midterm progress report CSV."""
    _progress._build_midterm_progress_report(output_path=output)

# ---------------------------------------------------------------------------
# report group
# ---------------------------------------------------------------------------

report_app = typer.Typer(
    help="Generate reports and save to REPORTS_DIR.",
    no_args_is_help=False,
)


@report_app.callback(invoke_without_command=True)
def report_callback(ctx: typer.Context) -> None:
    """Generate all reports. Run without a subcommand to run everything."""
    if ctx.invoked_subcommand is None:
        commands = [
            ("enrollment reports", _enrollment.report_all),
        ]
        commands.append(("engagement report", _engagement.report_all))
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            for name, run_fn in commands:
                task = progress.add_task(f"Generating {name}…", total=None)
                run_fn()
                progress.remove_task(task)


@report_app.command("enrollment")
def report_enrollment(
    rosters_dir: Annotated[
        Optional[Path],
        typer.Option(help="Directory containing dated roster subdirectories"),
    ] = None,
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Output directory for enrollment reports")
    ] = None,
) -> None:
    """Generate enrollment reports for all sections."""
    _enrollment._reports_impl(rosters_dir=rosters_dir, output_dir=output_dir)


app.add_typer(report_app, name="report")


@report_app.command("engagement")
def report_engagement_cmd(
    report_dir: Annotated[
        Optional[Path],
        typer.Option(help="Directory for report files (overrides configured report_path)."),
    ] = None,
) -> None:
    """Re-compute engagement and emit a validation report without saving a CSV."""
    _engagement._report_impl(report_dir=report_dir)

# ---------------------------------------------------------------------------
# daily command
# ---------------------------------------------------------------------------


@app.command()
def daily(
    headless: Annotated[
        bool,
        typer.Option(
            "--headless/--headed",
            help="Run browser headless (for automation) or headed (for debugging).",
        ),
    ] = False,
) -> None:
    """Run all daily data processing steps (get + process + report)."""
    logger.info("Running daily pipeline: get phase")
    for name, run_fn in [
        ("albert", _albert.run_all),
        ("brightspace", _brightspace.run_all),
        ("gradescope", _gradescope.run_all),
        ("edstem", _edstem.run_all),
    ]:
        logger.info(f"  → get {name}")
        run_fn(headless=headless)

    logger.info("Running daily pipeline: process phase")
    _enrollment.process_all()
    _gmail.process_all()
    _engagement.process_all()
    _progress.process_all()

    logger.info("Running daily pipeline: report phase")
    _enrollment.report_all()
    _engagement.report_all()

    logger.success("Daily pipeline complete.")


# ---------------------------------------------------------------------------
# Backward-compatible deprecated commands
# ---------------------------------------------------------------------------

_DEPRECATION_FMT = (
    "The '{old}' command is deprecated and will be removed in a future version. "
    "Use '{new}' instead."
)


@app.command("brightspace-gradebooks", hidden=True)
def _compat_brightspace_gradebooks(
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Output directory for Brightspace gradebooks")
    ] = None,
    clean: Annotated[
        bool,
        typer.Option(help="Remove existing files in output directories before fetching"),
    ] = False,
    headless: Annotated[
        bool,
        typer.Option(
            "--headless/--headed",
            help="Run browser headless (for automation) or headed (for debugging).",
        ),
    ] = False,
) -> None:
    """[Deprecated] Use 'get brightspace gradebooks' instead."""
    typer.echo(
        _DEPRECATION_FMT.format(
            old="brightspace-gradebooks", new="get brightspace gradebooks"
        ),
        err=True,
    )
    _brightspace._gradebooks_impl(output_dir=output_dir, clean=clean, headless=headless)


@app.command("brightspace-attendance", hidden=True)
def _compat_brightspace_attendance(
    output_dir: Annotated[
        Optional[Path],
        typer.Option(help="Output directory for Brightspace attendance files"),
    ] = None,
    clean: Annotated[
        bool,
        typer.Option(help="Remove existing files in output directories before fetching"),
    ] = False,
    headless: Annotated[
        bool,
        typer.Option(
            "--headless/--headed",
            help="Run browser headless (for automation) or headed (for debugging).",
        ),
    ] = False,
) -> None:
    """[Deprecated] Use 'get brightspace attendance' instead."""
    typer.echo(
        _DEPRECATION_FMT.format(
            old="brightspace-attendance", new="get brightspace attendance"
        ),
        err=True,
    )
    _brightspace._attendance_impl(output_dir=output_dir, clean=clean, headless=headless)


@app.command("albert-rosters", hidden=True)
def _compat_albert_rosters(
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
    """[Deprecated] Use 'get albert rosters' instead."""
    typer.echo(
        _DEPRECATION_FMT.format(old="albert-rosters", new="get albert rosters"),
        err=True,
    )
    _albert._rosters_impl(
        output_dir=output_dir,
        convert_to_csv=convert_to_csv,
        csv_output_dir=csv_output_dir,
        clean=clean,
    )


@app.command("albert-class-details", hidden=True)
def _compat_albert_class_details(
    output: Annotated[
        Optional[Path], typer.Option(help="Output path for the class details file")
    ] = None,
) -> None:
    """[Deprecated] Use 'get albert class-details' instead."""
    typer.echo(
        _DEPRECATION_FMT.format(
            old="albert-class-details", new="get albert class-details"
        ),
        err=True,
    )
    _albert._class_details_impl(output=output)


@app.command("gradescope-class-details", hidden=True)
def _compat_gradescope_class_details(
    output: Annotated[
        Optional[Path], typer.Option(help="Output path for the class details file")
    ] = None,
) -> None:
    """[Deprecated] Use 'get gradescope class-details' instead."""
    typer.echo(
        _DEPRECATION_FMT.format(
            old="gradescope-class-details", new="get gradescope class-details"
        ),
        err=True,
    )
    _gradescope._class_details_impl(output=output)


@app.command("gradescope-rosters", hidden=True)
def _compat_gradescope_rosters(
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Output directory for Gradescope rosters")
    ] = None,
    clean: Annotated[
        bool,
        typer.Option(help="Remove existing files in output directories before fetching"),
    ] = False,
) -> None:
    """[Deprecated] Use 'get gradescope rosters' instead."""
    typer.echo(
        _DEPRECATION_FMT.format(
            old="gradescope-rosters", new="get gradescope rosters"
        ),
        err=True,
    )
    _gradescope._rosters_impl(output_dir=output_dir, clean=clean)


@app.command("edstem-analytics", hidden=True)
def _compat_edstem_analytics(
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Output directory for EdStem analytics")
    ] = None,
    clean: Annotated[
        bool,
        typer.Option(help="Remove existing files in output directories before fetching"),
    ] = False,
) -> None:
    """[Deprecated] Use 'get edstem analytics' instead."""
    typer.echo(
        _DEPRECATION_FMT.format(old="edstem-analytics", new="get edstem analytics"),
        err=True,
    )
    _edstem._analytics_impl(output_dir=output_dir, clean=clean)


@app.command("enrollment-rosters", hidden=True)
def _compat_enrollment_rosters(
    rosters_dir: Annotated[
        Optional[Path],
        typer.Option(help="Directory containing dated roster subdirectories"),
    ] = None,
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Output directory for enrollment rosters")
    ] = None,
) -> None:
    """[Deprecated] Use 'process enrollment' instead."""
    typer.echo(
        _DEPRECATION_FMT.format(
            old="enrollment-rosters", new="process enrollment"
        ),
        err=True,
    )
    _enrollment._rosters_impl(rosters_dir=rosters_dir, output_dir=output_dir)


@app.command("enrollment-reports", hidden=True)
def _compat_enrollment_reports(
    rosters_dir: Annotated[
        Optional[Path],
        typer.Option(help="Directory containing dated roster subdirectories"),
    ] = None,
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Output directory for enrollment reports")
    ] = None,
) -> None:
    """[Deprecated] Use 'report enrollment' instead."""
    typer.echo(
        _DEPRECATION_FMT.format(
            old="enrollment-reports", new="report enrollment"
        ),
        err=True,
    )
    _enrollment._reports_impl(rosters_dir=rosters_dir, output_dir=output_dir)


@app.command("gmail-filters", hidden=True)
def _compat_gmail_filters(
    roster_paths: Annotated[
        Optional[list[Path]],
        typer.Option(
            help="One or more Albert roster XLS files. If not set, the most recently downloaded rosters will be used."
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to save the Gmail filter XML file. If not set, save to processed data directory."
        ),
    ] = None,
) -> None:
    """[Deprecated] Use 'process gmail-filters' instead."""
    typer.echo(
        _DEPRECATION_FMT.format(old="gmail-filters", new="process gmail-filters"),
        err=True,
    )
    _gmail._filters_impl(roster_paths=roster_paths, output=output)
