"""Brightspace CLI commands for ``dataset get brightspace``."""

from datetime import datetime, timezone
import json
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


def _quiz_courses() -> list[str]:
    """Course IDs to fetch quiz reports for.

    ``[tool.coursedata.brightspace.quizzes].courses`` if set (it must be a
    subset of ``[tool.coursedata.brightspace].courses``), else every course.
    """
    course_ids = [str(c) for c in BRIGHTSPACE_CONFIG.get("courses", [])]
    configured = BRIGHTSPACE_CONFIG.get("quizzes", {}).get("courses")
    if configured is None:
        return course_ids
    configured = [str(c) for c in configured]
    unknown = sorted(set(configured) - set(course_ids))
    if unknown:
        logger.error(
            f"[tool.coursedata.brightspace.quizzes].courses has {unknown}, "
            "which are not in [tool.coursedata.brightspace].courses."
        )
        raise typer.Exit(code=1)
    return configured


def _parse_time(value: Optional[str]) -> Optional[datetime]:
    """An aware datetime from a Brightspace/manifest ISO string (or None)."""
    return datetime.fromisoformat(value) if value else None


def _quiz_deadline(quiz: dict) -> Optional[datetime]:
    """When a quiz's reports are final enough to fetch: its due date, else its end date."""
    return _parse_time(quiz.get("due")) or _parse_time(quiz.get("end"))


def _quiz_fetch_reason(
    quiz: dict, entry: Optional[dict], reports: list[str], now: datetime, force: bool
) -> Optional[str]:
    """Why a quiz's reports should be (re)fetched, or None if they're up to date.

    Reports are fetched once when first seen, and once more after the
    quiz's deadline if the last fetch came before it. After that, only
    ``force`` fetches them again.
    """
    from edubag.brightspace.quizzes import report_matches

    if force:
        return "forced"
    if not entry or "fetched_at" not in entry:
        return "new"
    if entry.get("failed"):
        return "retry"
    if reports and not any(report_matches(r, reports) for r in entry.get("reports", {}).values()):
        return "new"
    deadline = _quiz_deadline(quiz)
    if deadline is not None and deadline <= now and _parse_time(entry["fetched_at"]) < deadline:
        return "past due"
    return None


def _write_json(path: Path, data) -> None:
    """Write JSON atomically, so an interrupted run never leaves a truncated file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    os.replace(tmp, path)


def _quizzes_impl(
    output_dir: Optional[Path] = None,
    quiz: Optional[list[str]] = None,
    report: Optional[list[str]] = None,
    due: bool = False,
    force: bool = False,
    list_only: bool = False,
    headless: bool = False,
) -> None:
    """Core implementation for fetching Brightspace quiz reports.

    Files go to ``<output_dir>/<course>/``, overwriting earlier copies, with
    ``quizzes.json`` (the quiz list) and ``manifest.json`` (what was fetched
    when) beside them.
    """
    if not BRIGHTSPACE_AVAILABLE:
        logger.error("edubag brightspace client is not available. Cannot fetch quiz reports.")
        raise typer.Exit(code=1)

    course_ids = _quiz_courses()
    if not course_ids:
        logger.error("No Brightspace course IDs found in configuration.")
        raise typer.Exit(code=1)

    if output_dir is None:
        output_dir = RAW_DATA_DIR / "brightspace" / "quizzes"
    reports = list(report or BRIGHTSPACE_CONFIG.get("quizzes", {}).get("reports", []))
    selectors = [q.casefold() for q in quiz or []]

    logger.info(f"Fetching Brightspace quiz reports for courses {course_ids} to '{output_dir}'")

    username, password = get_sso_credentials()

    try:
        client = BrightspaceClient()
        client.authenticate(username=username, password=password, headless=headless)
    except Exception as e:
        logger.error(f"Brightspace authentication failed: {e}")
        raise typer.Exit(code=1)

    failed = False
    for course in course_ids:
        course_dir = output_dir / course
        manifest_path = course_dir / "manifest.json"
        try:
            quiz_list = client.list_quizzes(course, headless=headless)
        except Exception as e:
            logger.error(f"Failed to list quizzes for course {course}: {e}")
            raise typer.Exit(code=1)
        if not quiz_list:
            logger.info(f"Course {course} has no quizzes.")
            continue
        _write_json(course_dir / "quizzes.json", quiz_list)
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

        now = datetime.now(timezone.utc)
        to_fetch = []
        for q in quiz_list:
            if selectors and not any(s == q["id"] or s in q["name"].casefold() for s in selectors):
                continue
            deadline = _quiz_deadline(q)
            start = _parse_time(q.get("start"))
            if not force and start is not None and start > now:
                skip = "not open"
            elif due and (deadline is None or deadline > now):
                skip = "not due"
            else:
                skip = None
            if skip:
                if list_only:
                    typer.echo(f"{course}  {skip:>10}  {q['name']}")
                continue
            reason = _quiz_fetch_reason(q, manifest.get(q["id"]), reports, now, force)
            if list_only:
                typer.echo(f"{course}  {reason or 'up to date':>10}  {q['name']}")
            if reason:
                to_fetch.append(q)

        if list_only:
            continue
        if not to_fetch:
            logger.info(f"Course {course}: quiz reports are up to date.")
            continue

        logger.info(f"Course {course}: fetching reports for {len(to_fetch)} quiz(zes)")
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            results = client.save_quiz_reports(
                course,
                quiz_ids=[q["id"] for q in to_fetch],
                reports=reports,
                save_dir=course_dir,
                headless=headless,
            )
        except Exception as e:
            logger.error(f"Failed to fetch quiz reports for course {course}: {e}")
            raise typer.Exit(code=1)

        for q in to_fetch:
            entry = manifest.setdefault(q["id"], {})
            entry.update(name=q["name"], due=q["due"], end=q["end"], fetched_at=fetched_at)
            quiz_results = [r for r in results if r["quiz_id"] == q["id"]]
            entry["failed"] = any(r["status"] == "failed" for r in quiz_results)
            for r in quiz_results:
                if r["status"] == "saved":
                    entry.setdefault("reports", {})[r["report_id"]] = {
                        "name": r["report"],
                        "type": r["report_type"],
                        "file": Path(r["path"]).name,
                        "fetched_at": fetched_at,
                    }
        _write_json(manifest_path, manifest)

        saved = sum(r["status"] == "saved" for r in results)
        failures = [r for r in results if r["status"] == "failed"]
        logger.info(f"Course {course}: saved {saved} report(s)")
        for r in failures:
            logger.error(f"Failed: {r['quiz_name']} - {r['report']}: {r['error']}")
        failed = failed or bool(failures)

    if failed:
        raise typer.Exit(code=1)
    if not list_only:
        logger.success("Brightspace quiz reports fetched successfully.")


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


@app.command()
def quizzes(
    ctx: typer.Context,
    quiz: Annotated[
        Optional[list[str]],
        typer.Option(help="Only this quiz ID or part of its name, e.g. '8.2' (repeatable)"),
    ] = None,
    report: Annotated[
        Optional[list[str]],
        typer.Option(help="Only reports with this name or type (repeatable; overrides pyproject)"),
    ] = None,
    due: Annotated[
        bool, typer.Option(help="Only quizzes whose due date (else end date) has passed")
    ] = False,
    force: Annotated[
        bool, typer.Option(help="Fetch even reports that are already up to date")
    ] = False,
    list_only: Annotated[
        bool, typer.Option("--list", help="Show what would be fetched, without downloading")
    ] = False,
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Output directory for quiz reports")
    ] = None,
) -> None:
    """Fetch quiz reports for configured courses into data/raw/brightspace/quizzes/<course>.

    A quiz's reports are fetched when first seen (once it has opened) and once
    more after its due date; later runs skip them unless --force.
    """
    headless = (ctx.obj or {}).get("headless", False)
    _quizzes_impl(
        output_dir=output_dir,
        quiz=quiz,
        report=report,
        due=due,
        force=force,
        list_only=list_only,
        headless=headless,
    )
