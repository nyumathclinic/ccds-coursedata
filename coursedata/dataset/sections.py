"""Sections dashboard processing and reporting.

Commands are wired into:
    python -m coursedata.dataset process sections
    python -m coursedata.dataset report sections
"""

from __future__ import annotations

from datetime import date, datetime
import json
from pathlib import Path
import re
import shutil
from typing import Optional

from loguru import logger
from nyu_colors import DEEP_VIOLET, LIGHT_GRAY, LIGHT_VIOLET2, MEDIUM_GRAY2, NYU_VIOLET, WHITE
import pandas as pd

from coursedata.config import (
    PROJ_ROOT,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SECTIONS_DASHBOARD_CONFIG,
)


DEFAULT_FIELD_MAP = {
    "section": ["section", "section_number", "class_section"],
    "section_type": ["class_type", "section_type", "type"],
    "instructor": ["instructor", "instructors", "instructor_name"],
    "location": ["location", "room", "building_and_room", "facility"],
    "meeting_times": ["days_and_times", "meeting_times", "meeting_time"],
    "capacity": ["class_capacity", "capacity", "enrollment_cap", "max_enrollment", "max_capacity"],
    "enrolled": ["enrollment_total", "enrolled", "enrollment", "current_enrollment", "enrolled_count"],
    "waitlist_count": ["wait_list_total", "waitlist_count", "waitlist", "waitlisted", "waitlist_total"],
    "waitlist_capacity": ["wait_list_capacity", "waitlist_capacity"],
    "status": ["status", "class_status"],
}


# Backward-compatible config key aliases.
FIELD_MAP_ALIASES = {
    "instructor": ["instructors"],
    "waitlist_count": ["waitlist"],
}


def _resolve_input_path(input_path: Optional[Path]) -> Path:
    if input_path is not None:
        return input_path if input_path.is_absolute() else (RAW_DATA_DIR / input_path)

    configured = str(SECTIONS_DASHBOARD_CONFIG.get("input_path", "albert/class_details")).strip()
    candidate = Path(configured)
    if candidate.is_absolute():
        return candidate

    normalized = configured.replace("\\", "/")
    if normalized.startswith("data/raw/"):
        return Path(normalized)
    if normalized.startswith("raw/"):
        return RAW_DATA_DIR / normalized.removeprefix("raw/")
    return RAW_DATA_DIR / normalized


def _latest_snapshot_file(base: Path) -> Path:
    if base.is_file():
        return base

    candidates = list(base.glob("*/class_details.json"))
    candidates.extend(base.glob("*.json"))

    direct_file = base / "class_details.json"
    if direct_file.exists():
        candidates.append(direct_file)

    if not candidates:
        raise FileNotFoundError(f"No class_details.json files found under {base}")

    def _sort_key(path: Path) -> tuple[str, float]:
        snap = _snapshot_from_path(path)
        return snap, path.stat().st_mtime

    return max(candidates, key=_sort_key)


def _coalesce_field_map() -> dict[str, list[str]]:
    configured = SECTIONS_DASHBOARD_CONFIG.get("field_map", {})
    field_map: dict[str, list[str]] = {}
    for canonical_key, defaults in DEFAULT_FIELD_MAP.items():
        value = configured.get(canonical_key)
        if value is None:
            for alias in FIELD_MAP_ALIASES.get(canonical_key, []):
                if alias in configured:
                    value = configured[alias]
                    break
        if value is None:
            value = defaults
        if isinstance(value, str):
            field_map[canonical_key] = [value]
        else:
            field_map[canonical_key] = [str(v) for v in value]
    return field_map


def _extract_nested(row: dict, dotted_key: str):
    current = row
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _extract_value(row: dict, keys: list[str]):
    for key in keys:
        value = _extract_nested(row, key)
        if value is not None and value != "":
            return value
    return None


# Albert's class-detail scraper reuses one authenticated session across
# sequential fetches; the roster header component it reads for `section`
# can lag one fetch behind (still showing the previous course's value).
# `full_course_name` (e.g. "MATH-UA 120 - 020 Discrete Mathematics") comes
# from a part of the page that consistently reflects the class actually
# requested, so prefer parsing the section out of it when available.
_SECTION_FROM_NAME_RE = re.compile(r"\s-\s*(\d{2,4})\s")


def _section_from_full_course_name(row: dict) -> Optional[str]:
    full_name = row.get("full_course_name")
    if not isinstance(full_name, str):
        return None
    match = _SECTION_FROM_NAME_RE.search(full_name)
    return match.group(1) if match else None


def _normalize_instructors(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        names: list[str] = []
        for item in value:
            if isinstance(item, dict):
                names.append(str(item.get("name", item.get("full_name", "")).strip()))
            else:
                names.append(str(item).strip())
        return ", ".join([n for n in names if n])
    if isinstance(value, dict):
        return str(value.get("name", value.get("full_name", "")).strip())
    return str(value)


def _normalize_number(value):
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def _snapshot_from_path(path: Path) -> str:
    date_pattern = re.compile(r"(\d{4}-\d{2}-\d{2})")

    for candidate in [path.parent.name, path.stem, path.name]:
        match = date_pattern.search(candidate)
        if not match:
            continue
        try:
            parsed = datetime.strptime(match.group(1), "%Y-%m-%d")
            return parsed.date().isoformat()
        except ValueError:
            continue

    return date.today().isoformat()


def _normalize_rows(raw_rows: list[dict], snapshot_date: str) -> pd.DataFrame:
    field_map = _coalesce_field_map()
    records = []
    for row in raw_rows:
        header_section = _extract_value(row, field_map["section"])
        derived_section = _section_from_full_course_name(row)
        if derived_section is not None:
            section_value = derived_section
            if header_section is not None and str(header_section).strip() != derived_section:
                logger.warning(
                    f"Section field mismatch for class_number={row.get('class_number')}: "
                    f"raw section field says '{header_section}' but full_course_name says "
                    f"'{derived_section}' (Albert scraper staleness); using '{derived_section}'."
                )
        else:
            section_value = header_section
        if section_value is None:
            continue

        record = {
            "section": str(section_value).strip(),
            "section_type": str(_extract_value(row, field_map["section_type"]) or "").strip(),
            "instructor": _normalize_instructors(_extract_value(row, field_map["instructor"])),
            "location": str(_extract_value(row, field_map["location"]) or "").strip(),
            "meeting_times": str(_extract_value(row, field_map["meeting_times"]) or "").strip(),
            "capacity": _normalize_number(_extract_value(row, field_map["capacity"])),
            "enrolled": _normalize_number(_extract_value(row, field_map["enrolled"])),
            "waitlist_count": _normalize_number(_extract_value(row, field_map["waitlist_count"])),
            "waitlist_capacity": _normalize_number(_extract_value(row, field_map["waitlist_capacity"])),
            "status": str(_extract_value(row, field_map["status"]) or "").strip(),
            "snapshot_date": snapshot_date,
        }
        records.append(record)

    if not records:
        return pd.DataFrame(
            columns=[
                "section",
                "section_type",
                "instructor",
                "location",
                "meeting_times",
                "capacity",
                "enrolled",
                "waitlist_count",
                "waitlist_capacity",
                "status",
                "snapshot_date",
            ]
        )

    df = pd.DataFrame(records)
    df.drop_duplicates(subset=["section"], keep="last", inplace=True)
    df.sort_values(by=["section"], inplace=True)
    return df


def _process_impl(
    input_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> tuple[Path, Path]:
    source_base = _resolve_input_path(input_path)
    source_file = _latest_snapshot_file(source_base)
    snapshot_date = _snapshot_from_path(source_file)

    if output_dir is None:
        output_dir = PROCESSED_DATA_DIR / "sections" / snapshot_date

    logger.info(f"Building sections dataset from {source_file}")
    with open(source_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError("Expected class_details JSON top-level object to be a list")

    df = _normalize_rows(raw_data, snapshot_date=snapshot_date)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sections.csv"
    json_path = output_dir / "sections.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    latest_dir = PROCESSED_DATA_DIR / "sections" / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(latest_dir / "sections.csv", index=False)
    df.to_json(latest_dir / "sections.json", orient="records", indent=2)

    logger.success(f"Wrote canonical sections table to {csv_path}")
    logger.success(f"Wrote canonical sections JSON to {json_path}")
    return csv_path, json_path


def _render_dashboard_html(df: pd.DataFrame, title: str) -> str:
    palette = {
        "bg": LIGHT_GRAY,
        "ink": DEEP_VIOLET,
        "accent": NYU_VIOLET,
        "accent_dark": DEEP_VIOLET,
        "panel": WHITE,
        "line": MEDIUM_GRAY2,
        "header": LIGHT_VIOLET2,
    }
    rows_json = df.to_json(orient="records")
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<html lang=\"en\"> 
<head>
  <meta charset=\"utf-8\"> 
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"> 
  <title>{title}</title>
  <style>
    :root {{
            --bg: {palette['bg']};
            --ink: {palette['ink']};
            --accent: {palette['accent']};
            --accent-dark: {palette['accent_dark']};
            --panel: {palette['panel']};
            --line: {palette['line']};
            --header: {palette['header']};
    }}
        body {{ margin: 0; font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--ink); }}
        header {{ padding: 24px; background: linear-gradient(120deg, var(--accent), var(--accent-dark)); color: #ffffff; border-bottom: 1px solid var(--accent-dark); }}
        h1 {{ margin: 0 0 6px; letter-spacing: 0.01em; }}
        main {{ padding: 20px; max-width: 1200px; margin: 0 auto; }}
    .controls {{ margin-bottom: 12px; display: flex; gap: 12px; flex-wrap: wrap; }}
        input {{ padding: 8px 10px; border: 1px solid var(--line); border-radius: 8px; min-width: 240px; background: var(--panel); color: var(--ink); }}
        input:focus {{ outline: 2px solid var(--accent); outline-offset: 1px; }}
        table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); box-shadow: 0 10px 28px rgba(51, 6, 98, 0.10); }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; }}
        th {{ background: var(--header); color: var(--accent-dark); position: sticky; top: 0; }}
        tbody tr:nth-child(even) {{ background: #faf8fc; }}
        tbody tr:hover {{ background: #f3ecf8; }}
        .meta {{ font-size: 0.9rem; color: #f2eefe; }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <p class=\"meta\">Generated {generated}</p>
  </header>
  <main>
    <div class=\"controls\">
      <input id=\"q\" type=\"search\" placeholder=\"Filter by section, instructor, location...\">
    </div>
    <table>
      <thead>
        <tr>
                    <th>Section</th><th>Type</th><th>Instructor</th><th>Location</th><th>Meeting Times</th>
                    <th>Capacity</th><th>Enrolled</th><th>Waitlist</th><th>Waitlist Cap</th><th>Status</th>
        </tr>
      </thead>
      <tbody id=\"rows\"></tbody>
    </table>
  </main>
  <script>
    const rows = {rows_json};
    const tbody = document.getElementById('rows');
    const search = document.getElementById('q');

    function render(items) {{
      tbody.innerHTML = items.map(r => `
        <tr>
          <td>${{r.section ?? ''}}</td>
                    <td>${{r.section_type ?? ''}}</td>
                    <td>${{r.instructor ?? ''}}</td>
          <td>${{r.location ?? ''}}</td>
          <td>${{r.meeting_times ?? ''}}</td>
          <td>${{r.capacity ?? ''}}</td>
          <td>${{r.enrolled ?? ''}}</td>
                    <td>${{r.waitlist_count ?? ''}}</td>
                    <td>${{r.waitlist_capacity ?? ''}}</td>
          <td>${{r.status ?? ''}}</td>
        </tr>
      `).join('');
    }}

    search.addEventListener('input', () => {{
      const q = search.value.toLowerCase().trim();
      if (!q) {{ render(rows); return; }}
      render(rows.filter(r =>
        String(r.section ?? '').toLowerCase().includes(q) ||
                String(r.section_type ?? '').toLowerCase().includes(q) ||
                String(r.instructor ?? '').toLowerCase().includes(q) ||
        String(r.location ?? '').toLowerCase().includes(q) ||
        String(r.meeting_times ?? '').toLowerCase().includes(q)
      ));
    }});

    render(rows);
  </script>
</body>
</html>
"""


def _reports_impl(
    processed_json_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> tuple[Path, Path]:
    if processed_json_path is None:
        processed_json_path = PROCESSED_DATA_DIR / "sections" / "latest" / "sections.json"
        if not processed_json_path.exists():
            _, processed_json_path = _process_impl()

    with open(processed_json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)

    # Backward compatibility for artifacts generated with older column names.
    if "instructor" not in df.columns and "instructors" in df.columns:
        df["instructor"] = df["instructors"]
    if "waitlist_count" not in df.columns and "waitlist" in df.columns:
        df["waitlist_count"] = df["waitlist"]
    if "section_type" not in df.columns:
        df["section_type"] = ""
    if "waitlist_capacity" not in df.columns:
        df["waitlist_capacity"] = pd.NA

    snapshot = "latest"
    if "snapshot_date" in df.columns and not df.empty:
        snapshot = str(df["snapshot_date"].iloc[0])

    if output_dir is None:
        output_dir = REPORTS_DIR / "sections" / snapshot
    output_dir.mkdir(parents=True, exist_ok=True)

    title = str(SECTIONS_DASHBOARD_CONFIG.get("title", "Course Sections Dashboard")).strip()
    html_path = output_dir / "index.html"
    summary_path = output_dir / "summary.md"

    docs_publish_dir_raw = str(
        SECTIONS_DASHBOARD_CONFIG.get("docs_publish_dir", "docs/docs/sections")
    ).strip()
    docs_publish_dir = Path(docs_publish_dir_raw)
    if not docs_publish_dir.is_absolute():
        docs_publish_dir = (PROJ_ROOT / docs_publish_dir).resolve()
    docs_snapshot_dir = docs_publish_dir / "snapshots" / snapshot
    docs_latest_html = docs_publish_dir / "index.html"
    docs_latest_summary = docs_publish_dir / "summary.md"
    docs_snapshot_html = docs_snapshot_dir / "index.html"
    docs_snapshot_summary = docs_snapshot_dir / "summary.md"

    html_path.write_text(_render_dashboard_html(df, title=title), encoding="utf-8")

    total_sections = int(df["section"].nunique()) if "section" in df.columns else 0
    total_capacity = int(df["capacity"].fillna(0).sum()) if "capacity" in df.columns else 0
    lecture_mask = (
        df["section_type"].astype(str).str.strip().str.casefold().eq("lecture")
        if "section_type" in df.columns
        else pd.Series(False, index=df.index)
    )
    total_lecture_enrolled = (
        int(df.loc[lecture_mask, "enrolled"].fillna(0).sum()) if "enrolled" in df.columns else 0
    )
    total_waitlist = (
        int(df["waitlist_count"].fillna(0).sum()) if "waitlist_count" in df.columns else 0
    )
    total_waitlist_capacity = (
        int(df["waitlist_capacity"].fillna(0).sum()) if "waitlist_capacity" in df.columns else 0
    )

    summary = [
        f"# {title}",
        "",
        f"Snapshot: {snapshot}",
        "",
        "## Totals",
        f"- Sections: {total_sections}",
        f"- Capacity: {total_capacity}",
        f"- Lecture Enrolled: {total_lecture_enrolled}",
        f"- Waitlist: {total_waitlist}",
        f"- Waitlist Capacity: {total_waitlist_capacity}",
        "",
        "## Artifacts",
        f"- Dashboard HTML: {html_path}",
        f"- Canonical JSON: {processed_json_path}",
        f"- Docs Latest HTML: {docs_latest_html}",
        f"- Docs Latest Summary: {docs_latest_summary}",
        f"- Docs Snapshot HTML: {docs_snapshot_html}",
        f"- Docs Snapshot Summary: {docs_snapshot_summary}",
    ]
    summary_path.write_text("\n".join(summary), encoding="utf-8")

    docs_publish_dir.mkdir(parents=True, exist_ok=True)
    docs_snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(html_path, docs_latest_html)
    shutil.copy2(summary_path, docs_latest_summary)
    shutil.copy2(html_path, docs_snapshot_html)
    shutil.copy2(summary_path, docs_snapshot_summary)

    logger.success(f"Wrote sections dashboard report to {html_path}")
    logger.success(f"Wrote sections summary report to {summary_path}")
    logger.success(f"Published sections dashboard to docs at {docs_latest_html}")
    return html_path, summary_path


def process_all() -> None:
    """Run all sections processing commands."""
    _process_impl()


def report_all() -> None:
    """Run all sections report commands."""
    _reports_impl()