"""``dataset get aim``: testing-center exam scans from the AIM instructor portal.

Each exam to fetch is a table in pyproject.toml describing its rows on AIM's Testing > Completed
page; the key names the exam folder under ``RAW_DATA_DIR/artifacts`` that ``process artifacts``
reads::

    [tool.coursedata.aim.exams.ch7]
    crn   = ["10488", "10493"]   # default: [tool.coursedata.albert].courses
    type  = ["Exam"]             # optional; also subject, course, section
    date  = 2026-11-05           # required
    until = 2026-11-13           # or grace_days = 8; with neither, only `date`
    # artifacts  = "ch7"         # exam folder (default: the key)
    # fetch_days = 7             # keep checking this long after the last exam date
    # term       = "Fall 2026"   # fail unless AIM shows this term

``get aim ch7`` downloads that exam's new scans into ``RAW_DATA_DIR/artifacts/ch7/``, keeping
the ZIPs in ``_zips/`` there and a record of what was fetched in ``aim-manifest.json``. A row is
fetched again only if its "Uploaded On" date changes (a rescan), or with ``--force``.
``get aim`` alone does this for every configured exam that is still active: from ``date`` to
``fetch_days`` after its last date, since scans are uploaded after the exam. ``get aim --all``
does it for every configured exam, active or not.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
from pathlib import Path
from typing import Annotated, Optional

from loguru import logger
import typer

try:
    from edubag.aim.client import AimClient, AimError
    from edubag.aim.portal import ExamFilter, ExamRow

    AIM_AVAILABLE = True
except ImportError:
    AIM_AVAILABLE = False

from coursedata.config import AIM_CONFIG, ALBERT_CONFIG, RAW_DATA_DIR

from ._utils import get_sso_credentials

MANIFEST = "aim-manifest.json"
DEFAULT_FETCH_DAYS = 7
FILTER_KEYS = ("crn", "type", "subject", "course", "section", "date", "until", "grace_days")
KNOWN_KEYS = {*FILTER_KEYS, "artifacts", "fetch_days", "term"}


class ConfigError(ValueError):
    """A bad [tool.coursedata.aim] entry or command-line filter."""


@dataclass
class ExamSpec:
    name: str
    filter: "ExamFilter"
    artifacts: str
    fetch_days: int = DEFAULT_FETCH_DAYS
    term: Optional[str] = None

    @property
    def save_dir(self) -> Path:
        return RAW_DATA_DIR / "artifacts" / self.artifacts

    def active(self, today: date) -> bool:
        """Whether scans may still be arriving: from ``date`` to ``fetch_days`` past the end."""
        f = self.filter
        return f.date <= today <= f.end + timedelta(days=self.fetch_days)


def configured_exams() -> dict[str, dict]:
    return dict(AIM_CONFIG.get("exams", {}))


def exam_spec(name: str, cfg: dict) -> ExamSpec:
    """Validate one exam's settings (config merged with command-line overrides)."""
    unknown = set(cfg) - KNOWN_KEYS
    if unknown:
        raise ConfigError(f"AIM exam {name!r}: unknown keys {sorted(unknown)} (known: {sorted(KNOWN_KEYS)})")
    crn = cfg.get("crn") or ALBERT_CONFIG.get("courses")
    if not crn:
        raise ConfigError(
            f"AIM exam {name!r}: no CRNs. Set crn = [...] for the exam, or "
            "[tool.coursedata.albert].courses for all of them."
        )
    if not cfg.get("date"):
        raise ConfigError(f"AIM exam {name!r}: date is required")
    try:
        flt = ExamFilter.build(**{k: cfg.get(k) for k in FILTER_KEYS} | {"crn": [str(c) for c in crn]})
    except ValueError as e:
        raise ConfigError(f"AIM exam {name!r}: {e}") from e
    return ExamSpec(
        name=name,
        filter=flt,
        artifacts=str(cfg.get("artifacts") or name),
        fetch_days=int(cfg.get("fetch_days", DEFAULT_FETCH_DAYS)),
        term=cfg.get("term"),
    )


def select_specs(
    names: list[str], overrides: dict, today: date, include_inactive: bool = False
) -> list[ExamSpec]:
    """The exams to fetch: those named (config + overrides), or else all active configured ones
    (all configured ones with ``include_inactive``)."""
    exams = configured_exams()
    if include_inactive and (names or overrides):
        raise ConfigError("--all takes no exam names or filters")
    if overrides and len(names) != 1:
        raise ConfigError("Filter options apply to exactly one exam name (the artifacts folder)")
    if names:
        missing = [n for n in names if n not in exams and not overrides]
        if missing:
            raise ConfigError(
                f"No [tool.coursedata.aim.exams.<name>] for {missing}; configured: {sorted(exams) or 'none'}"
            )
        return [exam_spec(n, {**exams.get(n, {}), **overrides}) for n in names]
    specs = [exam_spec(n, cfg) for n, cfg in exams.items()]
    if include_inactive:
        return specs
    active = [s for s in specs if s.active(today)]
    for s in specs:
        if s not in active:
            logger.debug(f"AIM exam {s.name}: not active today ({s.filter.describe()})")
    return active


# ---------------------------------------------------------------------------- manifest


def load_manifest(save_dir: Path) -> dict:
    path = save_dir / MANIFEST
    if not path.exists():
        return {"exams": {}}
    data = json.loads(path.read_text())
    data.setdefault("exams", {})
    return data


def write_manifest(save_dir: Path, manifest: dict) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / MANIFEST
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def fetch_reason(row: "ExamRow", entry: Optional[dict], force: bool) -> Optional[str]:
    """Why to (re)download a row, or None to skip it."""
    if force:
        return "forced"
    if entry is None:
        return "new"
    uploaded = row.uploaded_on.isoformat() if row.uploaded_on else None
    if entry.get("uploaded_on") != uploaded:
        return "re-uploaded"
    return None


def record_batch(manifest: dict, batch, save_dir: Path, now: datetime) -> None:
    files = [str(f.relative_to(save_dir)) for f in batch.files]
    for row in batch.rows:
        info = row.to_json()
        manifest["exams"][row.key] = {
            **{k: info[k] for k in ("crn", "subject", "course", "section", "name", "email", "type")},
            "exam_date": info["exam_date"],
            "time_range": info["time_range"],
            "uploaded_on": info["uploaded_on"],
            "fetched_at": now.isoformat(timespec="seconds"),
            "zip": str(batch.zip_path.relative_to(save_dir)),
            "batch_files": files,
        }


# ---------------------------------------------------------------------------- command


def _describe(row: "ExamRow") -> str:
    return f"{row.crn} {row.type} {row.exam_date} {row.time_range} {row.name}"


def fetch_exam(client, spec: ExamSpec, *, list_only: bool, force: bool, headless: bool, credentials) -> int:
    """Fetch one exam's new rows; returns how many were downloaded (or would be)."""
    save_dir = spec.save_dir
    manifest = load_manifest(save_dir)
    username, password = credentials
    logger.info(f"AIM exam {spec.name}: {spec.filter.describe()} -> {save_dir}")

    if list_only:
        rows = client.list_exams(spec.filter, headless=headless, username=username, password=password, term=spec.term)
        todo = 0
        for row in rows:
            reason = fetch_reason(row, manifest["exams"].get(row.key), force)
            todo += bool(reason)
            typer.echo(f"  {(reason or 'have'):<12} {_describe(row)}")
        typer.echo(f"  {len(rows)} matching, {todo} to fetch")
        return todo

    reasons: dict[str, str] = {}

    def want(row) -> bool:
        reason = fetch_reason(row, manifest["exams"].get(row.key), force)
        if reason:
            reasons[row.key] = reason
        return reason is not None

    batches = client.save_exams(
        save_dir,
        spec.filter,
        want=want,
        headless=headless,
        username=username,
        password=password,
        term=spec.term,
    )
    now = datetime.now().astimezone()
    for batch in batches:
        for row in batch.rows:
            logger.info(f"  {reasons.get(row.key, 'fetched')}: {_describe(row)}")
        record_batch(manifest, batch, save_dir, now)
    manifest["last_checked"] = now.isoformat(timespec="seconds")
    write_manifest(save_dir, manifest)
    n = sum(len(b.rows) for b in batches)
    files = sum(len(b.files) for b in batches)
    if n:
        logger.success(f"AIM exam {spec.name}: {n} exam(s), {files} file(s) into {save_dir}")
    else:
        logger.info(f"AIM exam {spec.name}: nothing new")
    return n


def aim_cmd(
    ctx: typer.Context,
    exams: Annotated[
        Optional[list[str]],
        typer.Argument(help="Exam names from [tool.coursedata.aim.exams] (default: all active ones)"),
    ] = None,
    all_exams: Annotated[bool, typer.Option("--all", help="Every configured exam, active or not")] = False,
    list_only: Annotated[bool, typer.Option("--list", help="Show matching rows; download nothing")] = False,
    force: Annotated[bool, typer.Option(help="Download rows already fetched, too")] = False,
    crn: Annotated[Optional[list[str]], typer.Option(help="CRN filter (repeatable); overrides config")] = None,
    exam_type: Annotated[
        Optional[list[str]], typer.Option("--type", help="Type filter, e.g. Exam (repeatable); overrides config")
    ] = None,
    exam_date: Annotated[Optional[str], typer.Option("--date", help="Exam date YYYY-MM-DD; overrides config")] = None,
    until: Annotated[Optional[str], typer.Option(help="Last exam date YYYY-MM-DD; overrides config")] = None,
    grace_days: Annotated[Optional[int], typer.Option(help="Instead of --until: DATE plus this many days")] = None,
) -> None:
    """Download testing-center exam scans from AIM into RAW_DATA_DIR/artifacts/<exam>/.

    With filter options and one exam name, fetch ad hoc into that exam folder (merged over any
    configured settings for it).
    """
    if not AIM_AVAILABLE:
        logger.error("edubag.aim is not available; upgrade python-edubag.")
        raise typer.Exit(code=1)
    headless = (ctx.obj or {}).get("headless", True)
    overrides = {
        k: v
        for k, v in {"crn": crn, "type": exam_type, "date": exam_date, "until": until, "grace_days": grace_days}.items()
        if v
    }
    if "until" in overrides or "grace_days" in overrides:
        # An override of one end of the range replaces the other from config.
        overrides.setdefault("until", None)
        overrides.setdefault("grace_days", None)
    try:
        specs = select_specs(list(exams or []), overrides, date.today(), include_inactive=all_exams)
    except ConfigError as e:
        logger.error(str(e))
        raise typer.Exit(code=2) from e
    if not specs:
        logger.info("No AIM exams are configured." if all_exams else "No AIM exams are active today.")
        return

    client = AimClient()
    credentials = get_sso_credentials()
    failed = False
    for spec in specs:
        try:
            fetch_exam(client, spec, list_only=list_only, force=force, headless=headless, credentials=credentials)
        except AimError as e:
            logger.error(f"AIM exam {spec.name}: {e}")
            failed = True
    if failed:
        raise typer.Exit(code=1)
