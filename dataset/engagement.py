"""Engagement score computation: ``dataset process engagement`` and ``dataset report engagement``.

This module aggregates multiple data sources—Brightspace gradebook, attendance,
EdStem analytics, and office-hours logs—into a single engagement score per
student, then writes the results to ``PROCESSED_DATA_DIR`` and optionally saves
a validation report to ``REPORTS_DIR``.

CLI entry points
----------------
::

    python -m coursedata.dataset process engagement
    python -m coursedata.dataset report engagement

Both commands read all configuration from ``[tool.coursedata.engagement]`` in
``pyproject.toml``.  Every option can be overridden from the command line; run
``--help`` for details.

Configuration schema (pyproject.toml)
--------------------------------------
::

    [tool.coursedata.engagement]
    # Path for the output CSV.  {date} is replaced with today's ISO date.
    # Relative to PROCESSED_DATA_DIR by default:
    output_path    = "brightspace/grades/{date}/engagement.csv"
    # Directory for warning/validation report files.
    # Relative to REPORTS_DIR by default:
    report_path    = "engagement"
    # When true, per-source columns are included in the output CSV.
    keep_source_columns = false

    [tool.coursedata.engagement.data_sources.<name>]
    type       = "brightspace_gradebook" | "attendance" |
                 "edstem_analytics"     | "office_hours_html"
    # Relative to RAW_DATA_DIR by default (or absolute).
    path       = "brightspace/gradebooks"
    # Optional: when choosing one file from a directory snapshot, require this
    # substring to appear in the file stem.
    stem_contains = "Recitation"
    # If path is a *directory* the module automatically selects the newest file
    # inside the most-recently-dated subdirectory (YYYY-MM-DD format).
    # For brightspace_gradebook sources only:
    categories = ["Cat1", "Cat2"]

    # Column specifications – one table per output column:
    [[tool.coursedata.engagement.columns]]
    name        = "My Ratio Column"
    # Ratio formula from a list of numerator terms and a denominator expression.
    numerator   = ["term1", "term2", "2*term3"]
    denominator = "denom_a + denom_b + 3"
    scale       = 100.0   # multiply result by this factor
    max_cap     = 100.0   # clip result to this maximum

    [[tool.coursedata.engagement.columns]]
    name          = "My Piecewise Column"
    # Piecewise linear mapping from an already-computed column.
    piecewise_base = "Name of previously computed column"
    piecewise = [
        { condition = "<= 20",            formula = "x" },
        { condition = "> 20 and < 80",    formula = "20 + (x - 20) * (4/3)" },
        { condition = ">= 80",            formula = "100" },
    ]
    scale   = 1.0
    max_cap = 100.0

Auto-computed denominator columns
-----------------------------------
After all sources are merged the module adds helper columns that formulas can
reference:

``{source}_{sanitized_category}_denominator``
    For every category listed under a ``brightspace_gradebook`` source.
    Value = total items − per-student exemptions for that category.
    *Sanitized* means hyphens and spaces in the category name are replaced by
    underscores, so e.g. category ``"Pre-Quizzes"`` becomes the column
    ``gradebook_Pre_Quizzes_denominator``.

``{source}_denominator``
    For every ``attendance`` source.
    Value = P + R + A (sessions where attendance was actively tracked,
    excluding excused absences).

Numerator terms and the denominator expression are passed through
:class:`edubag.aggregator.EngagementAggregator`, which automatically prefixes
bare column names with their source name.  Terms containing hyphens are
auto-wrapped in backticks so pandas eval resolves them correctly.

Credentials
-----------
This module does *not* fetch raw data; it processes files already present on
disk.  Use ``dataset get brightspace``, ``dataset get edstem``, etc. to
download fresh data first.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Optional

from loguru import logger
import pandas as pd
import typer

# --------------------------------------------------------------------------- #
# Optional edubag imports – module loads without them; errors at runtime only. #
# --------------------------------------------------------------------------- #
try:
    from edubag.aggregator import EngagementAggregator
    from edubag.brightspace.attendance import AttendanceData
    from edubag.brightspace.gradebook import Gradebook
    from edubag.edstem.analytics import EdstemAnalytics
    from edubag.sources import DataSource, OfficeHoursData
    from edubag.transformers import GradebookTransformer

    _EDUBAG_AVAILABLE = True
except ImportError:
    _EDUBAG_AVAILABLE = False

from coursedata.config import (
    ENGAGEMENT_CONFIG,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
)
from ._utils import d8


app = typer.Typer(help="Compute and report on student engagement scores.")


# =========================================================================== #
# Path resolution                                                              #
# =========================================================================== #


def _is_date_dir(name: str) -> bool:
    """Return ``True`` if *name* looks like an ISO-8601 date (``YYYY-MM-DD``)."""
    return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", name))


def resolve_newest_path(path: Path) -> Path:
    """Resolve *path* to a single file, auto-detecting the newest snapshot.

    If *path* is already an existing file it is returned unchanged.

    If *path* is a directory the function first looks for sub-directories whose
    names are ISO-8601 dates (``YYYY-MM-DD``), picks the lexicographically
    largest (i.e. newest date), and then returns the most-recently-modified
    regular file inside that sub-directory.  If no dated sub-directories are
    found the search falls back to the directory itself.

    Args:
        path: File or directory path to resolve.

    Returns:
        Path to a single file.

    Raises:
        FileNotFoundError: When *path* does not exist or no files can be found.
    """
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if path.is_file():
        return path

    dated_dirs = sorted(
        (d for d in path.iterdir() if d.is_dir() and _is_date_dir(d.name)),
        reverse=True,
    )
    search_dir = dated_dirs[0] if dated_dirs else path
    if dated_dirs:
        logger.info(f"  Auto-selected dated snapshot: {search_dir.name} in {path}")

    candidates = sorted(
        (f for f in search_dir.iterdir() if f.is_file()),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No files found in {search_dir}")
    if len(candidates) > 1:
        logger.warning(
            f"  Multiple files in {search_dir}; using most-recently-modified:"
            f" {candidates[0].name}"
        )
    return candidates[0]


def resolve_newest_matching_path(path: Path, stem_contains: str | None = None) -> Path:
    """Resolve *path* to one file, optionally filtering by a stem substring.

    If *path* is a directory, the newest dated snapshot is selected first.
    Within that snapshot, files are optionally filtered to those whose stem
    contains *stem_contains*, case-insensitively. The most-recently-modified
    matching file is returned.
    """
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if path.is_file():
        if stem_contains and stem_contains.lower() not in path.stem.lower():
            raise FileNotFoundError(
                f"File {path} does not match required stem substring {stem_contains!r}"
            )
        return path

    snapshot_dir = resolve_newest_snapshot_dir(path)
    candidates = [f for f in snapshot_dir.iterdir() if f.is_file()]
    if stem_contains:
        candidates = [
            f for f in candidates if stem_contains.lower() in f.stem.lower()
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No files in {snapshot_dir} matched stem substring {stem_contains!r}"
            )

    candidates = sorted(candidates, key=lambda f: f.stat().st_mtime, reverse=True)
    if len(candidates) > 1:
        logger.warning(
            f"  Multiple matching files in {snapshot_dir}; using most-recently-modified: "
            f"{candidates[0].name}"
        )
    return candidates[0]


def resolve_newest_snapshot_dir(path: Path) -> Path:
    """Resolve *path* to the newest snapshot directory.

    If *path* is already a file, returns its parent directory. If *path* is a
    directory with dated subdirectories, returns the newest dated subdirectory.
    Otherwise returns *path* itself.
    """
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if path.is_file():
        return path.parent

    dated_dirs = sorted(
        (d for d in path.iterdir() if d.is_dir() and _is_date_dir(d.name)),
        reverse=True,
    )
    snapshot_dir = dated_dirs[0] if dated_dirs else path
    if dated_dirs:
        logger.info(f"  Auto-selected dated snapshot: {snapshot_dir.name} in {path}")
    return snapshot_dir


def _resolve_config_path(raw: str, default_base: Path) -> Path:
    """Resolve a config path string to an absolute path.

    Resolution rules, in order:
    1) Absolute paths are used as-is.
    2) Paths starting with ``data/raw/`` are rebased to ``RAW_DATA_DIR``.
    3) Paths starting with ``data/processed/`` are rebased to
       ``PROCESSED_DATA_DIR``.
    4) Paths starting with ``reports/`` are rebased to ``REPORTS_DIR``.
    5) Any other relative path is resolved under *default_base*.
    """
    p = Path(raw)
    if p.is_absolute():
        return p

    normalized = raw.strip().lstrip("./")
    if normalized.startswith("data/raw/"):
        return RAW_DATA_DIR / normalized.removeprefix("data/raw/")
    if normalized.startswith("data/processed/"):
        return PROCESSED_DATA_DIR / normalized.removeprefix("data/processed/")
    if normalized.startswith("reports/"):
        return REPORTS_DIR / normalized.removeprefix("reports/")

    return default_base / normalized


# =========================================================================== #
# Data-source loading                                                          #
# =========================================================================== #


def _load_source(
    name: str,
    source_cfg: dict[str, Any],
) -> tuple["DataSource", list[str], dict[str, Any]]:
    """Load one data source as described by *source_cfg*.

    Args:
        name: Logical name for the source (used as the merge prefix).
        source_cfg: Sub-dictionary from ``[tool.coursedata.engagement.data_sources.<name>]``.

    Returns:
        A 3-tuple of ``(source, categories, category_metadata)``.
        *categories* and *category_metadata* are only populated for
        ``brightspace_gradebook`` sources.
    """
    raw_path = _resolve_config_path(source_cfg["path"], default_base=RAW_DATA_DIR)
    stem_contains = source_cfg.get("stem_contains")
    path = resolve_newest_matching_path(raw_path, stem_contains=stem_contains)
    source_type: str = source_cfg["type"]
    categories: list[str] = source_cfg.get("categories", [])
    category_metadata: dict[str, Any] = {}

    logger.info(f"Loading '{name}' ({source_type}) from {path}")

    if source_type == "brightspace_gradebook":
        source = Gradebook.from_csv(path)
        source.resolve_identity()
        # Drop any stale engagement columns so freshly computed values are used.
        stale = [
            c
            for c in source.grades.columns
            if c.startswith("Engagement Raw Score Points")
            or c.startswith("Engagement Adjusted Score Points")
        ]
        if stale:
            source.grades = source.grades.drop(columns=stale)
            source.data = source.grades.copy()
        if categories:
            transformer = GradebookTransformer(source)
            transformer.add_category_metrics(categories)
            category_metadata = transformer.get_metadata()
            for cat, meta in category_metadata.items():
                category_cols = meta.get("columns", [])
                active_cols = []
                for col in category_cols:
                    series = source.grades[col]
                    normalized = series.astype(str).str.strip().str.lower()
                    is_blank = normalized.isin({"", "nan", "none", "-"})
                    if not is_blank.all():
                        active_cols.append(col)

                meta["active_items"] = len(active_cols)
                meta["excluded_blank_columns"] = [
                    col for col in category_cols if col not in active_cols
                ]

                if meta["excluded_blank_columns"]:
                    logger.info(
                        f"  {cat}: using {meta['active_items']}/{meta['total_items']} items "
                        f"(excluded {len(meta['excluded_blank_columns'])} blank columns)"
                    )
                else:
                    logger.info(f"  {cat}: {meta['active_items']} items")
        return source, categories, category_metadata

    elif source_type == "attendance":
        source = AttendanceData.from_file(path)
        source.resolve_identity()
        _drop_unrecorded_attendance_sessions(source)
        return source, [], {}

    elif source_type == "edstem_analytics":
        source = EdstemAnalytics.from_file(path)
        source.resolve_identity()
        return source, [], {}

    elif source_type == "office_hours_html":
        if raw_path.is_dir():
            snapshot_dir = resolve_newest_snapshot_dir(raw_path)
            logger.info(
                f"Loading all office hours files from snapshot directory {snapshot_dir}"
            )
            source = _load_office_hours_sources_from_dir(snapshot_dir)
        else:
            source = OfficeHoursData.from_file(path)
            source.resolve_identity()
        return source, [], {}

    else:
        raise ValueError(f"Unknown source type '{source_type}' for source '{name}'")


def _load_office_hours_sources_from_dir(dir_path: Path) -> "OfficeHoursData":
    """Load and combine all office-hours files in a snapshot directory.

    Supported file types are whatever ``OfficeHoursData.from_file`` supports,
    such as zip, html, and csv. Files are combined by Username and summed over
    ``visit_count`` when present.
    """
    files = sorted(f for f in dir_path.iterdir() if f.is_file())
    if not files:
        raise FileNotFoundError(f"No files found in {dir_path}")

    loaded_sources: list[OfficeHoursData] = []
    for file_path in files:
        try:
            source = OfficeHoursData.from_file(file_path)
            source.resolve_identity()
            loaded_sources.append(source)
        except Exception as exc:
            logger.warning(f"Skipping office hours file {file_path}: {exc}")

    if not loaded_sources:
        raise FileNotFoundError(f"No readable office hours files found in {dir_path}")

    combined = pd.concat(
        [source.data for source in loaded_sources],
        axis=0,
        ignore_index=True,
        sort=False,
    )

    if "visit_count" in combined.columns and "Username" in combined.columns:
        combined = (
            combined.groupby("Username", as_index=False)
            .agg(visit_count=("visit_count", "sum"))
            .sort_values("Username")
            .reset_index(drop=True)
        )

    merged_source = OfficeHoursData()
    merged_source.data = combined
    merged_source.metadata = {
        "source": str(dir_path),
        "type": "office_hours_combined",
        "files_loaded": len(loaded_sources),
        "files": [str(file_path) for file_path in files],
    }
    return merged_source


def _drop_unrecorded_attendance_sessions(source: "AttendanceData") -> None:
    """Remove attendance session columns that are entirely unrecorded.

    This is a defensive post-load cleanup. ``AttendanceData.from_file()``
    already drops all-``-`` columns, but we repeat the check here so the
    denominator logic is insulated from source-format changes.
    """
    identifier_cols = {"First Name", "Last Name", "Username", "% Attendance"}
    status_cols = set(getattr(AttendanceData, "statuses", []))
    sessions = list(source.metadata.get("sessions", []))

    if not sessions:
        sessions = [
            col
            for col in source.data.columns
            if col not in identifier_cols and col not in status_cols
        ]

    kept_sessions: list[str] = []
    dropped_sessions: list[str] = []
    for col in sessions:
        if col not in source.data.columns:
            continue
        normalized = source.data[col].astype(str).str.strip()
        if normalized.eq("-").all():
            dropped_sessions.append(col)
            continue
        kept_sessions.append(col)

    if not dropped_sessions:
        source.metadata["sessions"] = kept_sessions
        return

    logger.info(
        f"Dropping {len(dropped_sessions)} unrecorded attendance session columns: "
        f"{', '.join(dropped_sessions)}"
    )
    source.data = source.data.drop(columns=dropped_sessions)

    for session_col in kept_sessions:
        source.data[session_col] = source.data[session_col].replace("-", "A")

    for status in AttendanceData.statuses:
        source.data[status] = source.data[kept_sessions].apply(
            lambda row, status=status: sum(1 for value in row if value == status),
            axis=1,
        )

    total = source.data["P"] + source.data["R"] + source.data["A"]
    source.data["% Attendance"] = (
        (source.data["P"] + 0.5 * source.data["R"]).where(total != 0, 0.0) / total.where(total != 0, 1)
    ).fillna(0.0)
    source.metadata["sessions"] = kept_sessions


# =========================================================================== #
# Auto-computed denominator columns                                            #
# =========================================================================== #


def _sanitize(name: str) -> str:
    """Replace non-identifier characters with underscores for column naming."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _add_denominator_columns(
    merged_df: "pd.DataFrame",  # type: ignore[name-defined]  # noqa: F821
    source_cfgs: dict[str, dict],
    category_metadatas: dict[str, dict],
) -> None:
    """Add auto-computed denominator helper columns to *merged_df* in-place.

    For each ``brightspace_gradebook`` source, adds::

        {source}_{sanitized_category}_denominator = total_items - exemptions

    For each ``attendance`` source, adds::

        {source}_denominator = P + R + A

    These columns can be referenced directly in column denominator formulas.
    """
    for src_name, src_cfg in source_cfgs.items():
        stype = src_cfg.get("type", "")

        if stype == "brightspace_gradebook":
            meta = category_metadatas.get(src_name, {})
            for category, cat_meta in meta.items():
                total = cat_meta.get("active_items", cat_meta.get("total_items", 0))
                safe_cat = _sanitize(category)
                exemptions_col = f"{src_name}_{category}_exemptions"
                denom_col = f"{src_name}_{safe_cat}_denominator"
                exemptions = (
                    merged_df[exemptions_col]
                    if exemptions_col in merged_df.columns
                    else pd.Series(0, index=merged_df.index)
                )
                merged_df[denom_col] = (total - exemptions).clip(lower=0)
                logger.debug(f"  {denom_col} = {total} - exemptions")

        elif stype == "attendance":
            denom_col = f"{src_name}_denominator"
            total = pd.Series(0, index=merged_df.index)
            for stat in ("P", "R", "A"):
                col = f"{src_name}_{stat}"
                if col in merged_df.columns:
                    total = total + merged_df[col]
            merged_df[denom_col] = total
            logger.debug(f"  {denom_col} = P + R + A")


def _add_configured_denominators(
    merged_df: "pd.DataFrame",  # type: ignore[name-defined]  # noqa: F821
    denominators_cfg: dict[str, dict],
) -> None:
    """Add denominator columns from config formulas.

    Each key in ``denominators_cfg`` becomes a column in ``merged_df`` with
    value computed via ``DataFrame.eval`` on the provided ``formula`` string.
    """
    if not denominators_cfg:
        return

    logger.info("Computing configured denominator formulas…")
    for denom_name, denom_cfg in denominators_cfg.items():
        formula = denom_cfg.get("formula")
        if not formula:
            logger.warning(f"  Denominator '{denom_name}' has no formula; skipped.")
            continue
        try:
            merged_df[denom_name] = merged_df.eval(formula, engine="python")
            logger.info(f"  Computed {denom_name}")
        except Exception as exc:
            logger.error(f"  Failed to compute {denom_name}: {exc}")


# =========================================================================== #
# Column-config building                                                       #
# =========================================================================== #


def _as_formula_term(term: str) -> str:
    """Wrap a numerator term in backticks when it contains hyphens.

    The :class:`~edubag.aggregator.EngagementAggregator` uses backtick
    notation to handle column names that are not valid Python identifiers.
    This function auto-wraps bare column names (e.g. ``Pre-Quizzes_positive``)
    while leaving arithmetic expressions (e.g. ``2*Answers``) unchanged.
    """
    term = term.strip()
    if term.startswith("`"):  # already quoted
        return term
    # Expressions with operators (other than a single leading coefficient)
    if re.search(r"[+/()\[\]]", term):
        return term
    # "2*ColName" — only quote the column part if it has hyphens
    m = re.match(r"^([\d.]+\s*\*\s*)(.+)$", term)
    if m:
        coeff, col = m.group(1), m.group(2).strip()
        if "-" in col:
            return f"{coeff}`{col}`"
        return term
    # Plain column name with hyphens
    if "-" in term:
        return f"`{term}`"
    return term


def _build_piecewise_formula(base_col: str, pieces: list[dict]) -> str:
    """Return a piecewise formula string using boolean-mask multiplication.

    Each *piece* must supply ``condition`` and ``formula`` keys.  In
    ``formula`` the letter ``x`` is a placeholder for *base_col*.

    Example input::

        pieces = [
            {"condition": "<= 20",           "formula": "x"},
            {"condition": "> 20 and < 80",   "formula": "20 + (x - 20) * (4/3)"},
            {"condition": ">= 80",           "formula": "100"},
        ]

    The generated formula exploits the fact that Python booleans behave as 0/1
    in arithmetic::

        ((`base` <= 20) * (`base`))
        + ((`base` > 20) * (`base` < 80) * (20 + (`base` - 20) * (4/3)))
        + ((`base` >= 80) * (100))
    """
    parts = []
    for piece in pieces:
        condition: str = piece["condition"]
        expr: str = piece["formula"].replace("x", f"`{base_col}`")
        if " and " in condition:
            sub_conds = condition.split(" and ")
            mask = " * ".join(f"(`{base_col}` {c.strip()})" for c in sub_conds)
        else:
            mask = f"(`{base_col}` {condition})"
        parts.append(f"({mask} * ({expr}))")
    return " + ".join(parts)


def _build_aggregator_config(columns_cfg: list[dict]) -> dict[str, dict]:
    """Convert the ``[[tool.coursedata.engagement.columns]]`` list to aggregator config.

    Supports two column-specification styles:

    **Ratio** (``numerator`` + ``denominator``)
        ``numerator`` is a list of formula terms that are *summed*.  Terms
        containing hyphens are auto-wrapped in backticks.
        ``denominator`` is an expression string.

    **Piecewise** (``piecewise_base`` + ``piecewise``)
        Builds a piecewise linear expression using boolean masking.

    Both styles accept optional ``scale`` (multiply result) and ``max_cap``
    (clip ceiling).
    """
    config: dict[str, dict] = {}
    for col_cfg in columns_cfg:
        name: str = col_cfg["name"]

        if "numerator" in col_cfg:
            numerator_terms = [_as_formula_term(t) for t in col_cfg["numerator"]]
            numerator_str = " + ".join(numerator_terms)
            denominator_str: str = col_cfg["denominator"]
            formula = f"({numerator_str}) / ({denominator_str})"

        elif "piecewise_base" in col_cfg:
            formula = _build_piecewise_formula(
                col_cfg["piecewise_base"], col_cfg["piecewise"]
            )

        else:
            logger.warning(f"Column '{name}' has no formula specification; skipped.")
            continue

        config[name] = {
            "formula": formula,
            "scale": col_cfg.get("scale", 1.0),
            "clip_upper": col_cfg.get("max_cap"),
        }
    return config


# =========================================================================== #
# Core implementations                                                         #
# =========================================================================== #


def _process_impl(
    output_path: Optional[Path] = None,
    keep_source_columns: Optional[bool] = None,
    save_csv: bool = True,
    with_report: bool = True,
    report_dir: Optional[Path] = None,
) -> None:
    """Compute engagement scores and write them to *output_path*.

    This is the primary implementation function shared by both the Typer
    command and the :func:`run_all` pipeline hook.

    Args:
        output_path: Override the configured ``output_path``.  When
            ``save_csv`` is ``False`` this parameter is ignored.
        keep_source_columns: Include per-source columns in the output CSV.
            Defaults to the value in config (``keep_source_columns = false``).
        save_csv: Write the output CSV.  Set to ``False`` when running for
            the report only.
        with_report: Print a validation report and save warnings to
            *report_dir*.
        report_dir: Override the configured ``report_path``.
    """
    if not _EDUBAG_AVAILABLE:
        logger.error(
            "The edubag package is required for engagement computation. "
            "Install it and try again."
        )
        raise typer.Exit(code=1)

    if not ENGAGEMENT_CONFIG:
        logger.error(
            "No [tool.coursedata.engagement] section found in pyproject.toml. "
            "Add configuration before running this command."
        )
        raise typer.Exit(code=1)

    source_cfgs: dict[str, dict] = ENGAGEMENT_CONFIG.get("data_sources", {})
    columns_cfg: list[dict] = ENGAGEMENT_CONFIG.get("columns", [])
    denominators_cfg: dict[str, dict] = ENGAGEMENT_CONFIG.get("denominators", {})
    validation_cfg: dict[str, Any] = ENGAGEMENT_CONFIG.get("validation", {})
    display_cfg: dict[str, Any] = ENGAGEMENT_CONFIG.get("display", {})

    if not source_cfgs:
        logger.error(
            "No data_sources configured under [tool.coursedata.engagement.data_sources]."
        )
        raise typer.Exit(code=1)
    if not columns_cfg:
        logger.error("No columns configured under [[tool.coursedata.engagement.columns]].")
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------ #
    # Resolve output path                                                  #
    # ------------------------------------------------------------------ #
    if save_csv:
        if output_path is None:
            raw_out: str = ENGAGEMENT_CONFIG.get(
                "output_path",
                "brightspace/grades/{date}/engagement.csv",
            )
            output_path = _resolve_config_path(
                raw_out.replace("{date}", d8),
                default_base=PROCESSED_DATA_DIR,
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if keep_source_columns is None:
        keep_source_columns = ENGAGEMENT_CONFIG.get("keep_source_columns", False)

    # ------------------------------------------------------------------ #
    # Load sources                                                         #
    # ------------------------------------------------------------------ #
    logger.info("Loading data sources…")
    sources: dict[str, DataSource] = {}
    gradebook_source: Optional[Gradebook] = None
    category_metadatas: dict[str, dict] = {}

    for src_name, src_cfg in source_cfgs.items():
        try:
            source, _, cat_meta = _load_source(src_name, src_cfg)
        except FileNotFoundError as exc:
            logger.warning(f"Skipping source '{src_name}': {exc}")
            continue
        sources[src_name] = source
        category_metadatas[src_name] = cat_meta
        if src_cfg.get("type") == "brightspace_gradebook":
            gradebook_source = source  # type: ignore[assignment]

    if not sources:
        logger.error("No data sources could be loaded.")
        raise typer.Exit(code=1)

    # ------------------------------------------------------------------ #
    # Aggregate                                                            #
    # ------------------------------------------------------------------ #
    logger.info("Merging sources…")
    aggregator = EngagementAggregator(base_gradebook=gradebook_source)
    for src_name, source in sources.items():
        aggregator.add_source(src_name, source)
    aggregator.merge_sources()
    assert aggregator.merged_data is not None

    # ------------------------------------------------------------------ #
    # Denominator columns (added to merged_data before compute_columns)   #
    # ------------------------------------------------------------------ #
    logger.info("Computing denominator columns…")
    _add_denominator_columns(
        aggregator.merged_data, source_cfgs, category_metadatas
    )
    _add_configured_denominators(aggregator.merged_data, denominators_cfg)

    # ------------------------------------------------------------------ #
    # Compute engagement columns                                           #
    # ------------------------------------------------------------------ #
    logger.info("Building column configuration…")
    aggregator.config = _build_aggregator_config(columns_cfg)
    logger.info("Computing engagement columns…")
    aggregator.compute_columns()

    # ------------------------------------------------------------------ #
    # Validate & report                                                    #
    # ------------------------------------------------------------------ #
    show_report = display_cfg.get("show_report", True)
    if with_report and show_report:
        _emit_report(
            aggregator,
            report_dir=report_dir,
            validation_cfg=validation_cfg,
        )

    # ------------------------------------------------------------------ #
    # Export                                                               #
    # ------------------------------------------------------------------ #
    if save_csv:
        assert output_path is not None
        full_output_gb = aggregator.to_gradebook(keep_source_columns=True)

        try:
            rel_output_path = output_path.relative_to(PROCESSED_DATA_DIR)
        except ValueError:
            rel_output_path = Path(output_path.name)

        interim_output_path = INTERIM_DATA_DIR / rel_output_path
        interim_output_path.parent.mkdir(parents=True, exist_ok=True)
        full_output_gb.to_csv(interim_output_path)
        logger.info(f"Wrote full engagement dataframe → {interim_output_path}")

        output_gb = (
            full_output_gb
            if bool(keep_source_columns)
            else aggregator.to_gradebook(keep_source_columns=False)
        )

        keep_only = ENGAGEMENT_CONFIG.get("keep_only_engagement_columns", False)
        if keep_only:
            engagement_col_names = [c["name"] for c in columns_cfg]
            keep_cols = ["Username"] + [
                c for c in engagement_col_names if c in output_gb.grades.columns
            ]
            output_gb.grades = output_gb.grades[keep_cols]

        output_gb.to_csv(output_path)
        logger.success(f"Wrote engagement scores → {output_path}")

        sample_rows = int(display_cfg.get("sample_rows", 0) or 0)
        if sample_rows > 0:
            logger.info(f"Sample engagement scores (first {sample_rows} rows):")
            print(output_gb.grades.head(sample_rows).to_string(index=False))


def _emit_report(
    aggregator: "EngagementAggregator",
    report_dir: Optional[Path] = None,
    validation_cfg: Optional[dict[str, Any]] = None,
) -> None:
    """Validate, print, and persist an engagement report.

    Args:
        aggregator: A fully computed :class:`~edubag.aggregator.EngagementAggregator`.
        report_dir: Directory where the warnings text file is saved.  Defaults
            to the value of ``report_path`` in config, or ``REPORTS_DIR /
            "engagement"``.
    """
    report = aggregator.validate()

    if validation_cfg is None:
        validation_cfg = {}
    zero_threshold = validation_cfg.get("warn_zero_percent_threshold", 50)
    for col_name, stats in report.get("column_stats", {}).items():
        if stats.get("count", 0) > 0:
            zero_pct = (stats["zeros"] / stats["count"]) * 100
            if zero_pct > zero_threshold:
                report.setdefault("warnings", []).append(
                    f"{col_name}: {zero_pct:.1f}% zeros (threshold: {zero_threshold}%)"
                )

    aggregator.print_report()

    if report_dir is None:
        raw_rdir: str = ENGAGEMENT_CONFIG.get("report_path", "")
        report_dir = (
            _resolve_config_path(raw_rdir, default_base=REPORTS_DIR)
            if raw_rdir
            else REPORTS_DIR / "engagement"
        )
    report_dir.mkdir(parents=True, exist_ok=True)

    warnings = report.get("warnings", [])
    if warnings:
        warn_path = report_dir / f"engagement_warnings_{d8}.txt"
        warn_path.write_text("\n".join(warnings) + "\n")
        logger.info(f"Warnings saved → {warn_path}")


def _report_impl(report_dir: Optional[Path] = None) -> None:
    """Re-run engagement computation and emit a report without saving the CSV."""
    _process_impl(save_csv=False, with_report=True, report_dir=report_dir)


# =========================================================================== #
# Typer CLI                                                                    #
# =========================================================================== #


@app.callback(invoke_without_command=True)
def engagement_callback(ctx: typer.Context) -> None:
    """Compute and report on student engagement scores.

    Running without a sub-command is equivalent to ``process``.
    """
    if ctx.invoked_subcommand is None:
        _process_impl()


@app.command("process")
def process_command(
    output: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output CSV path (overrides the configured output_path).",
        ),
    ] = None,
    keep_source_columns: Annotated[
        Optional[bool],
        typer.Option(
            "--keep-source-cols/--no-source-cols",
            help="Include per-source columns in the output CSV.",
        ),
    ] = None,
    report: Annotated[
        bool,
        typer.Option(
            "--report/--no-report",
            help="Print a validation report and save warnings after computing.",
        ),
    ] = True,
    report_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Directory for report files (overrides the configured report_path)."
        ),
    ] = None,
) -> None:
    """Compute engagement scores and save to PROCESSED_DATA_DIR."""
    _process_impl(
        output_path=output,
        keep_source_columns=keep_source_columns,
        with_report=report,
        report_dir=report_dir,
    )


@app.command("report")
def report_command(
    report_dir: Annotated[
        Optional[Path],
        typer.Option(
            help="Directory for report files (overrides the configured report_path)."
        ),
    ] = None,
) -> None:
    """Re-compute engagement and emit a validation report without saving a CSV.

    Useful when you only want to inspect statistics or refresh the warnings
    file without overwriting the processed output.
    """
    _report_impl(report_dir=report_dir)


# =========================================================================== #
# Pipeline hook                                                                #
# =========================================================================== #


def run_all() -> None:
    """Run the full engagement pipeline (compute + report).

    Called by the ``dataset process`` and ``dataset daily`` pipeline hooks.
    """
    _process_impl(with_report=True)


def process_all() -> None:
    """Alias for :func:`run_all` consistent with other dataset sub-modules."""
    run_all()


def report_all() -> None:
    """Emit an engagement report without saving the output CSV."""
    _report_impl()
