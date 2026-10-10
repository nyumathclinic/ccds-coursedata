"""WebAssign CLI commands for ``dataset get webassign``."""

from pathlib import Path
import re
from typing import Annotated, Optional

from loguru import logger
import typer

try:
    from edubag.webassign.client import WebAssignClient
    from edubag.webassign.export import WebAssignRoster, WebAssignScores
    WEBASSIGN_AVAILABLE = True
except ImportError:
    WEBASSIGN_AVAILABLE = False

from coursedata.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, WEBASSIGN_CONFIG
from ._utils import d8

app = typer.Typer(help="Fetch data from WebAssign.")

# Download kind -> folder name under data/{raw,processed}/webassign.
KIND_DIRS = {"roster": "rosters", "scores": "scores"}


def _section_ids() -> list[str]:
    """WebAssign section IDs from [tool.coursedata.webassign].courses."""
    return [str(s) for s in WEBASSIGN_CONFIG.get("courses", [])]


def _fetch_impl(
    kind: str,
    output_dir: Optional[Path] = None,
    headless: bool = False,
) -> None:
    """Save a dated download ("roster" or "scores") for each configured section."""
    if not WEBASSIGN_AVAILABLE:
        logger.error("edubag webassign client is not available. Cannot fetch WebAssign data.")
        raise typer.Exit(code=1)

    sections = _section_ids()
    if not sections:
        logger.info("No [tool.coursedata.webassign].courses configured; skipping WebAssign.")
        return

    if output_dir is None:
        output_dir = RAW_DATA_DIR / "webassign" / KIND_DIRS[kind] / d8

    # Sign-in uses the saved Cengage session (`edubag webassign client authenticate`),
    # or WEBASSIGN_USERNAME / WEBASSIGN_PASSWORD from .env.
    client = WebAssignClient()
    save = client.save_roster if kind == "roster" else client.save_scores
    for section in sections:
        try:
            path = save([section], save_dir=output_dir, headless=headless)
        except Exception as e:
            logger.error(f"Failed to fetch WebAssign {kind} for section {section}: {e}")
            raise typer.Exit(code=1)
        # WebAssign names downloads with an unrelated serial number, and the
        # file itself doesn't say which section it is; record that in the name.
        path = path.replace(output_dir / f"{kind}_{section}{path.suffix}")
        logger.info(f"Saved WebAssign {kind} for section {section} to {path}")
    logger.success(f"WebAssign {kind} fetched.")


def _process_impl(date: Optional[str] = None, kinds: tuple[str, ...] = ("roster", "scores")) -> None:
    """Convert the newest (or given date's) downloads to plain CSV and JSON."""
    if not WEBASSIGN_AVAILABLE:
        logger.error("edubag webassign module is not available. Cannot process WebAssign data.")
        raise typer.Exit(code=1)

    parsers = {"roster": WebAssignRoster, "scores": WebAssignScores}
    for kind in kinds:
        base = RAW_DATA_DIR / "webassign" / KIND_DIRS[kind]
        dated = sorted(
            d for d in base.glob("*")
            if d.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d.name)
            and (date is None or d.name == date)
        ) if base.exists() else []
        if not dated:
            logger.info(f"No WebAssign {kind} downloads in {base} for {date or 'any date'}; skipping.")
            continue
        date_dir = dated[-1]
        output_dir = PROCESSED_DATA_DIR / "webassign" / KIND_DIRS[kind] / date_dir.name
        for path in sorted(date_dir.glob(f"{kind}_*.csv")):
            if path.stem.removeprefix(f"{kind}_") not in _section_ids():
                logger.warning(
                    f"Skipping {path}: name doesn't carry a configured section ID "
                    f"(expected {kind}_<section>.csv; re-run `get webassign`)."
                )
                continue
            data = parsers[kind].from_csv(path)
            data.to_csv(output_dir / f"{path.stem}.csv")
            data.to_json(output_dir / f"{path.stem}.json")
            logger.info(f"{path.name}: {len(data.students)} students -> {output_dir / path.stem}.{{csv,json}}")
    logger.success("WebAssign downloads processed.")


def process_all() -> None:
    """Process the newest WebAssign roster and scores downloads."""
    _process_impl()


def run_all(headless: bool = False) -> None:
    """Run all WebAssign fetch commands."""
    _fetch_impl("roster", headless=headless)
    _fetch_impl("scores", headless=headless)


@app.callback(invoke_without_command=True)
def webassign_callback(ctx: typer.Context) -> None:
    """Fetch all WebAssign data. Run without a subcommand to fetch everything."""
    if ctx.invoked_subcommand is None:
        headless = (ctx.obj or {}).get("headless", False)
        run_all(headless=headless)


@app.command()
def rosters(
    ctx: typer.Context,
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Output directory for WebAssign rosters")
    ] = None,
) -> None:
    """Fetch the roster for each configured WebAssign section."""
    headless = (ctx.obj or {}).get("headless", False)
    _fetch_impl("roster", output_dir=output_dir, headless=headless)


@app.command()
def scores(
    ctx: typer.Context,
    output_dir: Annotated[
        Optional[Path], typer.Option(help="Output directory for WebAssign scores")
    ] = None,
) -> None:
    """Fetch assignment scores for each configured WebAssign section."""
    headless = (ctx.obj or {}).get("headless", False)
    _fetch_impl("scores", output_dir=output_dir, headless=headless)
