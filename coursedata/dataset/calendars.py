"""Per-section .ics calendar generation.

Cross-references the canonical sections table (``sections.json``) with the
course calendar CSV (the ``calendar`` Drive resource) to build one iCalendar
file per section: lecture sections get Lecture/Calendar/Exam events, and
recitation sections get Recitation events using their own meeting weekday
(which may differ from the CSV's listed date -- see ``_build_recitation_events``).

Commands are wired into:
    python -m coursedata.dataset process calendars
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Optional, Union
from zoneinfo import ZoneInfo

from icalendar import Calendar, Event
from loguru import logger
import pandas as pd

from coursedata.config import (
    CALENDARS_CONFIG,
    COURSE_NAME,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RAW_DATA_DIR,
    TERM_CODE,
)

DAY_TOKENS = {"Mo": 0, "Tu": 1, "We": 2, "Th": 3, "Fr": 4, "Sa": 5, "Su": 6}

_MEETING_TIMES_RE = re.compile(
    r"^(?P<days>(?:Mo|Tu|We|Th|Fr|Sa|Su)+)\s+"
    r"(?P<start>\d{1,2}:\d{2}[AP]M)-(?P<end>\d{1,2}:\d{2}[AP]M)$"
)

LECTURE_TYPES = {"Lecture", "Calendar", "Exam"}


@dataclass
class EventSpec:
    summary: str
    start: Union[date, datetime]
    end: Union[date, datetime]
    all_day: bool
    location: Optional[str] = None


def _parse_meeting_times(value: str) -> tuple[list[int], time, time]:
    match = _MEETING_TIMES_RE.match(value.strip())
    if not match:
        raise ValueError(f"Could not parse meeting_times value: {value!r}")

    day_str = match.group("days")
    weekdays = [DAY_TOKENS[day_str[i : i + 2]] for i in range(0, len(day_str), 2)]
    start_time = datetime.strptime(match.group("start"), "%I:%M%p").time()
    end_time = datetime.strptime(match.group("end"), "%I:%M%p").time()
    return weekdays, start_time, end_time


def _group_for_weekdays(weekdays: list[int]) -> str:
    day_set = set(weekdays)
    if day_set == {0, 2}:
        return "MW"
    if day_set == {1, 3}:
        return "TR"
    raise ValueError(f"Unrecognized lecture meeting pattern for weekdays {sorted(day_set)}")


def _assign_groups(sections: list[dict]) -> dict[str, str]:
    groups: dict[str, str] = {}
    current_group: Optional[str] = None

    for row in sections:
        number = str(row.get("section", "")).strip()
        section_type = str(row.get("section_type", "")).strip().lower()

        if section_type == "lecture":
            weekdays, _, _ = _parse_meeting_times(str(row.get("meeting_times", "")))
            current_group = _group_for_weekdays(weekdays)
            groups[number] = current_group
        elif section_type == "recitation":
            if current_group is None:
                raise ValueError(
                    f"Recitation section {number} precedes any lecture section in sections.json"
                )
            groups[number] = current_group

    return groups


def _resolve_sections_path(input_path: Optional[Path]) -> Path:
    if input_path is not None:
        return input_path if input_path.is_absolute() else (PROJ_ROOT / input_path)

    configured = str(
        CALENDARS_CONFIG.get("sections_path", "data/processed/sections/latest/sections.json")
    ).strip()
    candidate = Path(configured)
    return candidate if candidate.is_absolute() else (PROJ_ROOT / candidate)


def _resolve_calendar_csv_path(input_path: Optional[Path]) -> Path:
    if input_path is not None:
        return input_path if input_path.is_absolute() else (RAW_DATA_DIR / input_path)

    configured = str(CALENDARS_CONFIG.get("calendar_csv_path", "drive/calendar")).strip()
    stem = Path(configured)
    base_dir = RAW_DATA_DIR / stem.parent
    filename = f"{stem.name}.csv"

    candidates = list(base_dir.glob(f"*/{filename}"))
    direct_file = base_dir / filename
    if direct_file.exists():
        candidates.append(direct_file)

    if not candidates:
        raise FileNotFoundError(
            f"No {filename} files found under {base_dir}. Run "
            "'python -m coursedata.dataset get drive --resource calendar' first."
        )

    def _sort_key(path: Path) -> tuple[str, float]:
        match = re.search(r"(\d{4}-\d{2}-\d{2})", path.parent.name)
        snap = match.group(1) if match else ""
        return snap, path.stat().st_mtime

    return max(candidates, key=_sort_key)


def _resolve_class_details_path(input_path: Optional[Path]) -> Optional[Path]:
    if input_path is not None:
        base_dir = input_path if input_path.is_absolute() else (RAW_DATA_DIR / input_path)
    else:
        configured = str(
            CALENDARS_CONFIG.get("class_details_path", "albert/class_details")
        ).strip()
        base = Path(configured)
        base_dir = base if base.is_absolute() else (RAW_DATA_DIR / base)

    if base_dir.is_file():
        return base_dir

    candidates = list(base_dir.glob("*/class_details.json"))
    direct_file = base_dir / "class_details.json"
    if direct_file.exists():
        candidates.append(direct_file)

    if not candidates:
        return None

    def _sort_key(path: Path) -> tuple[str, float]:
        match = re.search(r"(\d{4}-\d{2}-\d{2})", path.parent.name)
        snap = match.group(1) if match else ""
        return snap, path.stat().st_mtime

    return max(candidates, key=_sort_key)


def _load_course_codes(class_details_path: Optional[Path]) -> dict[str, str]:
    """Map section number -> "SUBJECT_CATALOG" (e.g. "MATH-UA_122") from raw class details.

    Read from raw Albert class_details.json rather than a config value so it can't
    drift out of sync with the actual course/section data, and so it works out of
    the box for cross-listed or renumbered sections.
    """
    if class_details_path is None or not class_details_path.exists():
        return {}

    with open(class_details_path, "r", encoding="utf-8") as f:
        raw_rows = json.load(f)

    codes: dict[str, str] = {}
    for row in raw_rows:
        section = str(row.get("section", "")).strip()
        subject = str(row.get("subject", "")).strip()
        catalog_number = row.get("catalog_number")
        if section and subject and catalog_number is not None:
            codes[section] = f"{subject}_{catalog_number}"
    return codes


def _snapshot_from_path(path: Path) -> str:
    date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})")
    for candidate in [path.parent.name, path.stem, path.name]:
        match = date_pattern.search(candidate)
        if not match:
            continue
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d").date().isoformat()
        except ValueError:
            continue
    return date.today().isoformat()


def _load_calendar_rows(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y").dt.date
    df["Type"] = df["Type"].fillna("").str.strip()
    df["Topic"] = df["Topic"].fillna("").str.strip()
    df["group_tokens"] = (
        df["Group"]
        .fillna("")
        .apply(lambda value: {token.strip() for token in value.split(",") if token.strip()})
    )
    return df


def _build_lecture_events(
    group: str,
    start_time: time,
    end_time: time,
    location: str,
    df: pd.DataFrame,
) -> list[EventSpec]:
    mask = df["Type"].isin(LECTURE_TYPES) & df["group_tokens"].apply(
        lambda tokens: group in tokens
    )
    events: list[EventSpec] = []

    for _, row in df.loc[mask].sort_values("Date").iterrows():
        row_date: date = row["Date"]
        topic = row["Topic"]

        if row["Type"] == "Calendar":
            events.append(
                EventSpec(
                    summary=topic,
                    start=row_date,
                    end=row_date + timedelta(days=1),
                    all_day=True,
                    location=None,
                )
            )
        else:
            events.append(
                EventSpec(
                    summary=topic,
                    start=datetime.combine(row_date, start_time),
                    end=datetime.combine(row_date, end_time),
                    all_day=False,
                    location=location or None,
                )
            )

    return events


def _build_recitation_events(
    group: str,
    weekdays: list[int],
    start_time: time,
    end_time: time,
    location: str,
    df: pd.DataFrame,
) -> list[EventSpec]:
    if len(weekdays) != 1:
        raise ValueError(f"Recitation sections must meet exactly one day, got weekdays={weekdays}")

    target_iso_weekday = weekdays[0] + 1  # our Monday=0 -> ISO Monday=1
    mask = (df["Type"] == "Recitation") & df["group_tokens"].apply(lambda tokens: group in tokens)
    events: list[EventSpec] = []

    for _, row in df.loc[mask].sort_values("Date").iterrows():
        row_date: date = row["Date"]
        iso_year, iso_week, _ = row_date.isocalendar()
        actual_date = date.fromisocalendar(iso_year, iso_week, target_iso_weekday)

        events.append(
            EventSpec(
                summary=row["Topic"],
                start=datetime.combine(actual_date, start_time),
                end=datetime.combine(actual_date, end_time),
                all_day=False,
                location=location or None,
            )
        )

    return events


def _build_calendar(section: str, events: list[EventSpec], tz: ZoneInfo) -> Calendar:
    cal = Calendar()
    cal.add("prodid", "-//coursedata//calendars//EN")
    cal.add("version", "2.0")
    cal.add("calscale", "GREGORIAN")
    cal.add("x-wr-calname", f"{COURSE_NAME} Section {section}")

    for event_spec in events:
        vevent = Event()
        vevent.add("summary", event_spec.summary)

        uid_source = f"{section}|{event_spec.start}|{event_spec.summary}"
        uid = hashlib.sha1(uid_source.encode("utf-8")).hexdigest()
        vevent.add("uid", f"{uid}@coursedata")
        vevent.add("dtstamp", datetime.now(timezone.utc))

        if event_spec.all_day:
            vevent.add("dtstart", event_spec.start)
            vevent.add("dtend", event_spec.end)
        else:
            vevent.add("dtstart", event_spec.start.replace(tzinfo=tz).astimezone(timezone.utc))
            vevent.add("dtend", event_spec.end.replace(tzinfo=tz).astimezone(timezone.utc))

        if event_spec.location:
            vevent.add("location", event_spec.location)

        cal.add_component(vevent)

    return cal


def _calendar_filename(section: str, course_code: Optional[str]) -> str:
    if course_code and TERM_CODE:
        return f"{course_code}_{section}_{TERM_CODE}_calendar.ics"
    return f"{section}.ics"


def _process_impl(
    sections_path: Optional[Path] = None,
    calendar_csv_path: Optional[Path] = None,
    class_details_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> list[Path]:
    sections_file = _resolve_sections_path(sections_path)
    csv_file = _resolve_calendar_csv_path(calendar_csv_path)

    logger.info(f"Building section calendars from {sections_file} and {csv_file}")
    with open(sections_file, "r", encoding="utf-8") as f:
        sections = json.load(f)

    groups = _assign_groups(sections)
    course_codes = _load_course_codes(_resolve_class_details_path(class_details_path))
    df = _load_calendar_rows(csv_file)
    tz = ZoneInfo(str(CALENDARS_CONFIG.get("timezone", "America/New_York")))

    snapshot_date = _snapshot_from_path(csv_file)
    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR / "calendars" / snapshot_date
    output_dir.mkdir(parents=True, exist_ok=True)

    latest_dir = PROCESSED_DATA_DIR / "calendars" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for row in sections:
        number = str(row.get("section", "")).strip()
        section_type = str(row.get("section_type", "")).strip().lower()
        group = groups.get(number)
        if group is None:
            logger.warning(f"Skipping section {number}: could not determine MW/TR group")
            continue

        weekdays, start_time, end_time = _parse_meeting_times(str(row.get("meeting_times", "")))
        location = str(row.get("location", "")).strip()

        if section_type == "lecture":
            events = _build_lecture_events(group, start_time, end_time, location, df)
        elif section_type == "recitation":
            events = _build_recitation_events(group, weekdays, start_time, end_time, location, df)
        else:
            continue

        cal = _build_calendar(number, events, tz)
        ics_bytes = cal.to_ical()

        filename = _calendar_filename(number, course_codes.get(number))
        out_path = output_dir / filename
        out_path.write_bytes(ics_bytes)
        (latest_dir / filename).write_bytes(ics_bytes)
        written.append(out_path)

    logger.success(f"Wrote {len(written)} section calendars to {output_dir}")
    return written


def process_all() -> None:
    """Run all calendar processing commands."""
    _process_impl()
