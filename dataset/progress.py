"""Midterm progress report generation.

Creates a syllabus-aligned progress CSV with columns:
- Username
- Name (Last, First)
- Progress Indicator (Strong, Satisfactory, Concerns)
- Comments

CLI:
    python -m coursedata.dataset process midterm-progress
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Optional

from loguru import logger
import pandas as pd
import typer

from coursedata.config import (
    ENGAGEMENT_CONFIG,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    PROGRESS_REPORT_CONFIG,
    RAW_DATA_DIR,
)
from edubag.brightspace.attendance import AttendanceData
from edubag.brightspace.gradebook import Gradebook

from ._utils import d8
from .engagement import _resolve_config_path, resolve_newest_matching_path


app = typer.Typer(help="Generate syllabus-aligned midterm progress report.")


def _coalesce_source_config(source_name: str, default_type: str, default_path: str) -> dict:
    report_sources = PROGRESS_REPORT_CONFIG.get("data_sources", {})
    if source_name in report_sources:
        cfg = dict(report_sources[source_name])
    else:
        cfg = dict(ENGAGEMENT_CONFIG.get("data_sources", {}).get(source_name, {}))

    if "type" not in cfg:
        cfg["type"] = default_type
    if "path" not in cfg:
        cfg["path"] = default_path
    return cfg


def _resolve_source_file(source_cfg: dict) -> Path:
    raw_path = _resolve_config_path(source_cfg["path"], default_base=RAW_DATA_DIR)
    return resolve_newest_matching_path(raw_path, stem_contains=source_cfg.get("stem_contains"))


def _active_category_columns(df: pd.DataFrame, category: str) -> list[str]:
    category_cols: list[str] = []
    for col in df.columns:
        match = re.search(r"Category:(\w+[\w\s-]*?)\s+(?:CategoryWeight|>)", col)
        if match and match.group(1).strip() == category:
            normalized = df[col].astype(str).str.strip().str.lower()
            is_blank = normalized.isin({"", "nan", "none", "-"})
            if not is_blank.all():
                category_cols.append(col)
    return category_cols


def _subtotal_ratio(df: pd.DataFrame, category: str) -> pd.Series:
    numerator_col = f"{category} Subtotal Numerator"
    denominator_col = f"{category} Subtotal Denominator"

    if numerator_col not in df.columns or denominator_col not in df.columns:
        return pd.Series(float("nan"), index=df.index)

    numerator = pd.to_numeric(df[numerator_col], errors="coerce")
    denominator = pd.to_numeric(df[denominator_col], errors="coerce")
    has_positive_denominator = denominator > 0

    ratio = pd.Series(float("nan"), index=df.index, dtype=float)
    ratio.loc[has_positive_denominator] = (
        numerator.loc[has_positive_denominator] / denominator.loc[has_positive_denominator]
    )
    return ratio


def _subtotal_source(df: pd.DataFrame, category: str) -> pd.Series:
    numerator_col = f"{category} Subtotal Numerator"
    denominator_col = f"{category} Subtotal Denominator"

    if numerator_col not in df.columns or denominator_col not in df.columns:
        return pd.Series("fallback", index=df.index, dtype="string")

    denominator = pd.to_numeric(df[denominator_col], errors="coerce")
    return pd.Series("fallback", index=df.index, dtype="string").where(
        ~(denominator > 0), "subtotal"
    )


def _completion_ratio(df: pd.DataFrame, category: str) -> pd.Series:
    subtotal_ratio = _subtotal_ratio(df, category)

    cols = _active_category_columns(df, category)
    if not cols:
        return subtotal_ratio

    positive = pd.Series(0, index=df.index, dtype=float)
    exemptions = pd.Series(0, index=df.index, dtype=float)

    for col in cols:
        values = df[col].astype(str).str.strip()
        lower = values.str.lower()
        exemptions += lower.eq("exempt").astype(float)

        numeric = pd.to_numeric(values, errors="coerce")
        positive += numeric.fillna(0).gt(0).astype(float)

    denominator = (len(cols) - exemptions).clip(lower=0)
    safe_denominator = denominator.where(denominator > 0, 1.0)
    ratio = (positive / safe_denominator).astype(float)
    ratio = ratio.where(denominator > 0, 0.0)
    ratio = ratio.fillna(0.0)
    return subtotal_ratio.where(subtotal_ratio.notna(), ratio)


def _max_points_from_header(col_name: str) -> float | None:
    match = re.search(r"MaxPoints:([0-9]+(?:\.[0-9]+)?)", col_name)
    if not match:
        return None
    return float(match.group(1))


def _extract_lecture_section(sections_value: object) -> str:
    """Extract lecture section identifier from Brightspace ``Sections`` value.

    The gradebook field typically contains two section values, where the first
    is lecture and the second is recitation. This function extracts the first
    value and then returns a compact section identifier when possible.
    """
    raw = "" if sections_value is None else str(sections_value).strip()
    if not raw:
        return "UNKNOWN"

    first = re.split(r"\s*[;,\n]\s*", raw)[0].strip()
    if not first:
        return "UNKNOWN"

    code_match = re.search(r"(?:^|[.\s_-])(\d{3})(?:[.\s_-]|$)", first)
    if code_match:
        return code_match.group(1)

    return re.sub(r"[^A-Za-z0-9._-]+", "_", first)


def _performance_ratio(df: pd.DataFrame, category: str) -> pd.Series:
    subtotal_ratio = _subtotal_ratio(df, category)

    cols = _active_category_columns(df, category)
    if not cols:
        return subtotal_ratio

    earned = pd.Series(0.0, index=df.index)
    possible = pd.Series(0.0, index=df.index)

    for col in cols:
        max_points = _max_points_from_header(col)
        if max_points is None:
            continue

        values = df[col].astype(str).str.strip()
        lower = values.str.lower()
        is_exempt = lower.eq("exempt")

        numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
        earned += numeric.where(~is_exempt, 0.0)
        possible += (~is_exempt).astype(float) * max_points

    safe_possible = possible.where(possible > 0, 1.0)
    ratio = (earned / safe_possible).astype(float)
    ratio = ratio.where(possible > 0, 0.0)
    ratio = ratio.fillna(0.0)
    return subtotal_ratio.where(subtotal_ratio.notna(), ratio)


def _weighted_row_score(row: pd.Series, weights: dict[str, float], component_cols: list[str]) -> float:
    numerator = 0.0
    denom = 0.0
    for name in component_cols:
        value = row[name]
        if pd.isna(value):
            continue
        weight = float(weights.get(name, 0.0))
        numerator += weight * float(value)
        denom += weight
    if denom == 0:
        return 0.0
    return numerator / denom


def _indicator_and_comment(
    row: pd.Series,
    strong_threshold: float,
    satisfactory_threshold: float,
    min_component_for_strong: float,
    concerns_below_any: float,
) -> tuple[str, str, str]:
    components = {
        "attendance/participation": float(row["attendance_participation"]),
        "pre-class assignments": float(row["preclass_assignments"]),
        "problem sets": float(row["problem_sets"]),
        "quizzes": float(row["quizzes"]),
        "midterm I": float(row["midterm_i"]),
    }
    suggestion_labels = {
        "attendance/participation": "attendance and participation",
        "pre-class assignments": "pre-class assignment completion",
        "problem sets": "problem set performance",
        "quizzes": "quiz performance",
        "midterm I": "exam performance",
    }
    overall = float(row["overall_progress"])

    low_components = [k for k, v in components.items() if v < concerns_below_any]

    if overall >= strong_threshold:
        if low_components:
            focus = ", ".join(suggestion_labels.get(k, k) for k in low_components)
            reason_code = "STRONG_WITH_LOW_" + "_".join(
                k.upper().replace("/", "_").replace(" ", "_") for k in low_components
            )
            return (
                "Strong",
                f"Strong overall progress; consider also improving {focus}.",
                reason_code,
            )
        return (
            "Strong",
            "Strong progress: consistent attendance/participation, pre-class completion, and strong performance on problem sets, quizzes, and Midterm I.",
            "STRONG_ALL_COMPONENTS",
        )

    if low_components:
        indicator = "Concerns"
        reason_code = "LOW_COMPONENTS_" + "_".join(
            comp.upper().replace("/", "_").replace(" ", "_") for comp in low_components
        )
        comments = (
            "Concerns about progress due to low or missing performance in "
            + ", ".join(low_components)
            + "."
        )
        return indicator, comments, reason_code

    if overall >= satisfactory_threshold:
        weakest = sorted(components.items(), key=lambda kv: kv[1])[:2]
        focus = ", ".join(suggestion_labels.get(k, k) for k, _ in weakest)
        reason_code = "SATISFACTORY_FOCUS_" + "_".join(
            k.upper().replace("/", "_").replace(" ", "_") for k, _ in weakest
        )
        return (
            "Satisfactory",
            f"Satisfactory progress overall; strengthen {focus} to see improvement.",
            reason_code,
        )

    weakest = sorted(components.items(), key=lambda kv: kv[1])[:2]
    focus = ", ".join(suggestion_labels.get(k, k) for k, _ in weakest)
    reason_code = "CONCERNS_OVERALL_" + "_".join(
        k.upper().replace("/", "_").replace(" ", "_") for k, _ in weakest
    )
    return (
        "Concerns",
        f"Concerns about progress; immediate improvement needed in {focus}.",
        reason_code,
    )


def _build_midterm_progress_report(output_path: Optional[Path] = None) -> Path:
    gradebook_cfg = _coalesce_source_config(
        "gradebook", default_type="brightspace_gradebook", default_path="brightspace/gradebooks"
    )
    attendance_cfg = _coalesce_source_config(
        "attendance", default_type="attendance", default_path="brightspace/attendance"
    )

    gradebook_path = _resolve_source_file(gradebook_cfg)
    attendance_path = _resolve_source_file(attendance_cfg)

    logger.info(f"Using gradebook file: {gradebook_path}")
    logger.info(f"Using attendance file: {attendance_path}")

    gb = Gradebook.from_csv(gradebook_path)
    gb.resolve_identity()
    att = AttendanceData.from_file(attendance_path)
    att.resolve_identity()

    grades_df = gb.grades.copy()
    attendance_df = att.data.copy()

    pre_quiz_ratio = _completion_ratio(grades_df, "Pre-Quizzes")
    pre_survey_ratio = _completion_ratio(grades_df, "Pre-Surveys")
    polls_ratio = _completion_ratio(grades_df, "Polls")

    problem_sets_ratio = _performance_ratio(grades_df, "Problem Sets")
    quizzes_ratio = _performance_ratio(grades_df, "Recitation Quizzes")
    midterm_ratio = _performance_ratio(grades_df, "Midterm I")

    attendance_df["recitation_attendance"] = (
        attendance_df["P"] + 0.5 * attendance_df["R"]
    ) / (attendance_df["P"] + attendance_df["R"] + attendance_df["A"]).replace(0, pd.NA)
    attendance_df["recitation_attendance"] = attendance_df["recitation_attendance"].fillna(0.0)

    progress_df = pd.DataFrame(
        {
            "Username": grades_df["Username"],
            "Last Name": grades_df.get("Last Name", ""),
            "First Name": grades_df.get("First Name", ""),
            "Sections": grades_df.get("Sections", ""),
            "pre_quiz_ratio": pre_quiz_ratio,
            "pre_survey_ratio": pre_survey_ratio,
            "polls_ratio": polls_ratio,
            "problem_sets": problem_sets_ratio,
            "quizzes": quizzes_ratio,
            "midterm_i": midterm_ratio,
        }
    )

    progress_df = progress_df.merge(
        attendance_df[["Username", "recitation_attendance"]], on="Username", how="left"
    )
    progress_df["recitation_attendance"] = progress_df["recitation_attendance"].fillna(0.0)

    progress_df["attendance_participation"] = (
        progress_df["recitation_attendance"] + progress_df["polls_ratio"]
    ) / 2.0
    progress_df["preclass_assignments"] = (
        progress_df["pre_quiz_ratio"] + progress_df["pre_survey_ratio"]
    ) / 2.0

    weights_cfg = PROGRESS_REPORT_CONFIG.get(
        "weights",
        {
            "attendance_participation": 0.20,
            "preclass_assignments": 0.20,
            "problem_sets": 0.20,
            "quizzes": 0.20,
            "midterm_i": 0.20,
        },
    )

    component_cols = [
        "attendance_participation",
        "preclass_assignments",
        "problem_sets",
        "quizzes",
        "midterm_i",
    ]
    progress_df["overall_progress"] = progress_df.apply(
        lambda row: _weighted_row_score(row, weights_cfg, component_cols), axis=1
    )

    strong_threshold = float(PROGRESS_REPORT_CONFIG.get("strong_threshold", 0.85))
    satisfactory_threshold = float(PROGRESS_REPORT_CONFIG.get("satisfactory_threshold", 0.65))
    min_component_for_strong = float(PROGRESS_REPORT_CONFIG.get("min_component_for_strong", 0.70))
    concerns_below_any = float(PROGRESS_REPORT_CONFIG.get("concerns_below_any", 0.40))

    indicators = progress_df.apply(
        lambda row: _indicator_and_comment(
            row,
            strong_threshold=strong_threshold,
            satisfactory_threshold=satisfactory_threshold,
            min_component_for_strong=min_component_for_strong,
            concerns_below_any=concerns_below_any,
        ),
        axis=1,
    )
    progress_df[["Progress Indicator", "Comments", "Reason Code"]] = pd.DataFrame(
        indicators.tolist(), index=progress_df.index
    )

    progress_df["Name (Last, First)"] = (
        progress_df["Last Name"].fillna("").astype(str)
        + ", "
        + progress_df["First Name"].fillna("").astype(str)
    ).str.strip(", ")
    progress_df["Lecture Section"] = progress_df["Sections"].map(_extract_lecture_section)

    # Dump full progress_df to interim directory for troubleshooting
    if output_path is None:
        raw_out = PROGRESS_REPORT_CONFIG.get(
            "output_path", "progress/midterm/{date}/midterm_progress_report.csv"
        )
        output_path = _resolve_config_path(
            str(raw_out).replace("{date}", d8),
            default_base=PROCESSED_DATA_DIR,
        )
    interim_path = _resolve_config_path(
        str(output_path.parent.relative_to(PROCESSED_DATA_DIR)) + "/progress_df.csv",
        default_base=INTERIM_DATA_DIR,
    )
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    progress_df.to_csv(interim_path, index=False)
    logger.info(f"Dumped full progress_df to interim: {interim_path}")

    output_cols = [
        "Username",
        "Name (Last, First)",
        "Progress Indicator",
        "Comments",
        "Reason Code",
    ]
    output_df = progress_df[output_cols].copy()
    output_df = output_df.sort_values("Name (Last, First)", kind="stable")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    logger.success(f"Wrote midterm progress report: {output_path}")

    # Also emit one CSV per lecture section.
    per_lecture = progress_df.copy()
    for lecture_section, lecture_df in per_lecture.groupby("Lecture Section", dropna=False):
        lecture_output = lecture_df[output_cols].copy().sort_values(
            "Name (Last, First)", kind="stable"
        )
        lecture_suffix = str(lecture_section) if lecture_section else "UNKNOWN"
        lecture_path = output_path.with_name(
            f"{output_path.stem}__lecture_{lecture_suffix}{output_path.suffix}"
        )
        lecture_output.to_csv(lecture_path, index=False)
        logger.success(
            f"Wrote lecture-section progress report ({lecture_suffix}): {lecture_path}"
        )

    return output_path


@app.command("midterm-progress")
def midterm_progress(
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output CSV path override."),
    ] = None,
) -> None:
    """Generate a syllabus-aligned midterm progress report CSV."""
    _build_midterm_progress_report(output_path=output)


def process_all() -> None:
    """Run default progress report processing tasks."""
    _build_midterm_progress_report()
