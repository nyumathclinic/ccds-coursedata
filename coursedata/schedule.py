"""Run jobs from a weekly schedule in ``[[tool.coursedata.schedule]]``.

Each entry in ``pyproject.toml`` looks like::

    [[tool.coursedata.schedule]]
    name  = "quiz-reports"
    days  = ["Tue", "Thu"]        # or "daily"
    time  = "08:05"               # local wall-clock time, 24-hour
    from  = 2026-09-01            # optional, inclusive
    until = 2026-12-14            # optional, inclusive
    run   = ["python -m coursedata.dataset get brightspace quizzes --due"]

``python -m coursedata.schedule run`` is meant to be invoked often (every few
minutes, by a LaunchAgent). An entry is *due* when today is one of its days,
inside its date range, the current time is at or past ``time``, and it has not
already run today. Missed times therefore catch up later the same day (e.g.
after the Mac wakes from sleep), but never on a later day.

An entry is recorded as run for the day whether its commands succeed or fail,
so a broken login is not retried every few minutes (which would mean repeated
Duo pushes). Failures raise a macOS notification instead.

Run items are command lines executed in the project root. A leading ``python``
is replaced by the current interpreter, so ``python -m coursedata...`` uses the
project's environment. The items of one entry run in order; a failing item
does not stop the rest.
"""

from __future__ import annotations

import datetime as dt
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Optional

from loguru import logger
import typer

from coursedata.config import INTERIM_DATA_DIR, PROJ_ROOT, _config

app = typer.Typer(help="Run jobs from [[tool.coursedata.schedule]] when they are due.")

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
STATE_PATH = INTERIM_DATA_DIR / "schedule-state.json"


class ScheduleConfigError(ValueError):
    """Raised for a malformed [[tool.coursedata.schedule]] entry."""


@dataclass(frozen=True)
class Entry:
    name: str
    days: frozenset[int]  # 0 = Monday, as in date.weekday()
    time: dt.time
    run: tuple[str, ...]
    start: Optional[dt.date] = None
    until: Optional[dt.date] = None

    def is_due(self, now: dt.datetime, last_run: Optional[str]) -> bool:
        today = now.date()
        if today.weekday() not in self.days:
            return False
        if self.start is not None and today < self.start:
            return False
        if self.until is not None and today > self.until:
            return False
        if now.time() < self.time:
            return False
        return last_run != today.isoformat()


def _parse_days(value: Any, name: str) -> frozenset[int]:
    if value == "daily":
        return frozenset(range(7))
    if not isinstance(value, list) or not value:
        raise ScheduleConfigError(f"{name}: days must be \"daily\" or a list like [\"Tue\", \"Thu\"]")
    days = set()
    for day in value:
        key = str(day).strip()[:3].title()
        if key not in DAY_NAMES:
            raise ScheduleConfigError(f"{name}: unknown day {day!r}; use {DAY_NAMES}")
        days.add(DAY_NAMES.index(key))
    return frozenset(days)


def _parse_time(value: Any, name: str) -> dt.time:
    if isinstance(value, dt.time):
        return value
    try:
        hour, minute = str(value).split(":")
        return dt.time(int(hour), int(minute))
    except ValueError as e:
        raise ScheduleConfigError(f"{name}: time must be \"HH:MM\", got {value!r}") from e


def _parse_date(value: Any, name: str, key: str) -> Optional[dt.date]:
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError as e:
        raise ScheduleConfigError(f"{name}: {key} must be a date like 2026-12-14") from e


def parse_entries(raw: Any) -> list[Entry]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ScheduleConfigError("[tool.coursedata.schedule] must be an array of tables ([[...]])")
    entries = []
    seen = set()
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ScheduleConfigError(f"schedule entry {i} is not a table")
        name = item.get("name")
        if not name or not isinstance(name, str):
            raise ScheduleConfigError(f"schedule entry {i} needs a name")
        if name in seen:
            raise ScheduleConfigError(f"duplicate schedule entry name {name!r}")
        seen.add(name)
        run = item.get("run")
        if isinstance(run, str):
            run = [run]
        if not run or not all(isinstance(cmd, str) and cmd.strip() for cmd in run):
            raise ScheduleConfigError(f"{name}: run must be a non-empty list of command lines")
        unknown = set(item) - {"name", "days", "time", "from", "until", "run"}
        if unknown:
            raise ScheduleConfigError(f"{name}: unknown keys {sorted(unknown)}")
        entries.append(
            Entry(
                name=name,
                days=_parse_days(item.get("days"), name),
                time=_parse_time(item.get("time"), name),
                run=tuple(run),
                start=_parse_date(item.get("from"), name, "from"),
                until=_parse_date(item.get("until"), name, "until"),
            )
        )
    return entries


def load_state(path: Optional[Path] = None) -> dict[str, str]:
    path = path or STATE_PATH
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        logger.warning(f"Ignoring unreadable schedule state {path}")
        return {}


def save_state(state: dict[str, str], path: Optional[Path] = None) -> None:
    path = path or STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _argv(command: str) -> list[str]:
    argv = shlex.split(command)
    if argv and argv[0] == "python":
        argv[0] = sys.executable
    return argv


def _notify(message: str) -> None:
    script = f'display notification {json.dumps(message)} with title "Course data schedule"'
    try:
        subprocess.run(["osascript", "-e", script], check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        pass


def run_entry(entry: Entry, cwd: Optional[Path] = None) -> bool:
    cwd = cwd or PROJ_ROOT
    ok = True
    for command in entry.run:
        logger.info(f"[{entry.name}] $ {command}")
        try:
            result = subprocess.run(_argv(command), cwd=cwd, check=False)
            returncode = result.returncode
        except OSError as e:
            logger.error(f"[{entry.name}] could not start {command!r}: {e}")
            returncode = -1
        if returncode != 0:
            logger.error(f"[{entry.name}] {command!r} exited with status {returncode}")
            ok = False
    return ok


def _entries_or_exit() -> list[Entry]:
    try:
        return parse_entries(_config.get("schedule"))
    except ScheduleConfigError as e:
        logger.error(f"Invalid [[tool.coursedata.schedule]] config: {e}")
        _notify(f"{PROJ_ROOT.name}: invalid schedule config")
        raise typer.Exit(code=1)


@app.command("run")
def run(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show which entries are due without running them.")
    ] = False,
) -> None:
    """Run every schedule entry that is due now."""
    entries = _entries_or_exit()
    state = load_state()
    now = dt.datetime.now()
    due = [entry for entry in entries if entry.is_due(now, state.get(entry.name))]
    if not due:
        logger.debug("No schedule entries due.")
        return
    for entry in due:
        if dry_run:
            logger.info(f"[{entry.name}] due; would run: {list(entry.run)}")
            continue
        logger.info(f"[{entry.name}] starting")
        # Record before running so a crash or kill doesn't cause a retry storm.
        state[entry.name] = now.date().isoformat()
        save_state(state)
        if run_entry(entry):
            logger.success(f"[{entry.name}] finished")
        else:
            _notify(f"{PROJ_ROOT.name}: {entry.name} failed")


@app.command("list")
def list_entries() -> None:
    """Show each schedule entry and when it last ran."""
    entries = _entries_or_exit()
    state = load_state()
    now = dt.datetime.now()
    for entry in entries:
        days = "daily" if len(entry.days) == 7 else ",".join(DAY_NAMES[d] for d in sorted(entry.days))
        window = f"{entry.start or '…'} – {entry.until or '…'}"
        flag = "  [due now]" if entry.is_due(now, state.get(entry.name)) else ""
        typer.echo(
            f"{entry.name:20} {days:16} {entry.time:%H:%M}  {window:25} "
            f"last run: {state.get(entry.name, 'never')}{flag}"
        )
        for command in entry.run:
            typer.echo(f"{'':22}$ {command}")


if __name__ == "__main__":
    app()
