"""WebAssign CLI commands for ``dataset get webassign``."""

from pathlib import Path
from typing import Annotated, Optional

from loguru import logger
import typer

try:
    from edubag.webassign.client import WebAssignClient
    WEBASSIGN_AVAILABLE = True
except ImportError:
    WEBASSIGN_AVAILABLE = False

from coursedata.config import RAW_DATA_DIR, WEBASSIGN_CONFIG
from ._utils import d8

app = typer.Typer(help="Fetch data from WebAssign.")


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
        output_dir = RAW_DATA_DIR / "webassign" / ("rosters" if kind == "roster" else kind) / d8

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
        logger.info(f"Saved WebAssign {kind} for section {section} to {path}")
    logger.success(f"WebAssign {kind} fetched.")


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
