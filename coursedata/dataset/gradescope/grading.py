"""Grading dashboard for a multi-part, multi-version Gradescope exam.

An exam is configured in ``pyproject.toml`` as one or more *parts* (e.g. multiple choice and
free response), each made of one or more Gradescope assignment containers (version sets)::

    [tool.coursedata.gradescope.grading.ch5-6]
    title = "Chapter 5 and 6 Test"
    course = "1350932"                                   # Gradescope course of every part
    roster = "MATH-UA_122.016_021_*_roster.csv"          # in RAW_DATA_DIR/gradescope/rosters/<date>/
    deadline = "2026-10-20T17:00"                        # optional: flags problems falling behind

    [[tool.coursedata.gradescope.grading.ch5-6.parts]]
    name = "MC"
    containers = ["419376", "419377", "419378"]

    [[tool.coursedata.gradescope.grading.ch5-6.parts]]
    name = "FR"
    containers = ["419372"]
    # course = "..."                                     # optional per-part override
    # question_map = { "8876144" = { "3" = "4", "4" = "3" } }   # renumber one version's questions

    [tool.coursedata.gradescope.grading.ch5-6.graders]   # problem (or whole part) -> who to ping
    "FR.1" = ["Alice", "Bob"]
    "FR" = ["Carol"]

Questions are matched across versions by outline number (``FR.1`` is question 1 of every FR
version), unless a version's ``question_map`` renumbers them.

Steps (all keyed by exam, which may be omitted when only one exam is configured):

* ``dataset get gradescope grading [EXAM]``: one Gradescope session that reads every version's
  Grading Dashboard and scores CSV into ``RAW_DATA_DIR/gradescope/grading/<exam>/<timestamp>/``.
* ``dataset process gradescope-grading [EXAM]``: from the newest snapshot, write
  ``students.csv``, ``progress.csv`` and ``stats.csv`` to
  ``PROCESSED_DATA_DIR/gradescope/grading/<exam>/`` and append to ``history.csv`` there.
* ``dataset report gradescope-grading [EXAM]``: write ``dashboard.html`` next to them.
* ``dataset watch gradescope-grading [EXAM]``: get, process and report in a loop.

Everything stays under ``data/`` (git-ignored): the dashboard lists students by name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import html
import json
from pathlib import Path
import re
import shutil
import time
from typing import Optional
import webbrowser

from loguru import logger
from nyu_colors import (
    DEEP_VIOLET,
    LIGHT_GRAY,
    LIGHT_VIOLET1,
    LIGHT_VIOLET2,
    MEDIUM_GRAY1,
    MEDIUM_GRAY2,
    NYU_VIOLET,
    WHITE,
)
import pandas as pd

from coursedata.config import GRADESCOPE_CONFIG, PROCESSED_DATA_DIR, RAW_DATA_DIR

GRADING_CONFIG = GRADESCOPE_CONFIG.get("grading", {})
RAW_GRADING_DIR = RAW_DATA_DIR / "gradescope" / "grading"
PROCESSED_GRADING_DIR = PROCESSED_DATA_DIR / "gradescope" / "grading"
ROSTERS_DIR = RAW_DATA_DIR / "gradescope" / "rosters"

SNAPSHOT_MARKER = "snapshot.json"  # written last, so only complete snapshots are processed
RATE_WINDOW = timedelta(minutes=30)  # grading rate is measured over this much history
MIN_RATE_SPAN = timedelta(minutes=5)  # ...but not over less than this


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class Part:
    name: str
    course: str
    containers: list[str]
    question_map: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass
class Exam:
    key: str
    title: str
    roster: str
    parts: list[Part]
    graders: dict[str, list[str]]
    deadline: Optional[datetime] = None

    @property
    def raw_dir(self) -> Path:
        return RAW_GRADING_DIR / self.key

    @property
    def processed_dir(self) -> Path:
        return PROCESSED_GRADING_DIR / self.key

    def responsible(self, part: str, problem: str) -> list[str]:
        """Who is responsible for a problem: its own graders entry, else its part's."""
        return list(self.graders.get(problem) or self.graders.get(part) or [])


def exam_keys() -> list[str]:
    return [key for key, value in GRADING_CONFIG.items() if isinstance(value, dict)]


def load_exam(key: Optional[str] = None) -> Exam:
    """Read an exam's configuration.

    Raises:
        ValueError: If the exam is not configured (or ``key`` is omitted and there is not
            exactly one exam), or its configuration is incomplete.
    """
    keys = exam_keys()
    if key is None:
        if len(keys) != 1:
            raise ValueError(
                f"Name the exam to use; configured exams: {', '.join(keys) or 'none'} "
                "(see [tool.coursedata.gradescope.grading] in pyproject.toml)"
            )
        key = keys[0]
    if key not in keys:
        raise ValueError(f"No exam '{key}' in [tool.coursedata.gradescope.grading]; configured: {keys}")
    cfg = GRADING_CONFIG[key]

    def require(table: dict, name: str, where: str):
        if not table.get(name):
            raise ValueError(f"Missing '{name}' in {where}")
        return table[name]

    where = f"[tool.coursedata.gradescope.grading.{key}]"
    parts = []
    for i, part_cfg in enumerate(require(cfg, "parts", where)):
        part_where = f"{where} parts[{i}]"
        course = part_cfg.get("course") or require(cfg, "course", where)
        containers = [str(c) for c in require(part_cfg, "containers", part_where)]
        question_map = {
            str(aid): {str(k): str(v) for k, v in mapping.items()}
            for aid, mapping in (part_cfg.get("question_map") or {}).items()
        }
        parts.append(Part(str(require(part_cfg, "name", part_where)), str(course), containers, question_map))
    names = [p.name for p in parts]
    if len(set(names)) != len(names):
        raise ValueError(f"Part names must be distinct in {where}: {names}")

    deadline = cfg.get("deadline")
    if isinstance(deadline, str):
        deadline = datetime.fromisoformat(deadline)
    if isinstance(deadline, datetime) and deadline.tzinfo is None:
        deadline = deadline.astimezone()

    graders = {
        str(k): [v] if isinstance(v, str) else [str(x) for x in v] for k, v in (cfg.get("graders") or {}).items()
    }
    return Exam(
        key=key,
        title=str(cfg.get("title") or key),
        roster=str(require(cfg, "roster", where)),
        parts=parts,
        graders=graders,
        deadline=deadline,
    )


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


def _get_impl(exam_key: Optional[str] = None, headless: bool = True, scores: bool = True) -> Path:
    """Save a snapshot of the exam's grading from Gradescope; return the snapshot directory."""
    from edubag.gradescope.client import GradescopeClient

    from . import _get_gradescope_credentials

    exam = load_exam(exam_key)
    started = datetime.now().astimezone()
    snapshot_dir = exam.raw_dir / started.strftime("%Y-%m-%dT%H%M%S")

    containers_by_course: dict[str, list[str]] = {}
    for part in exam.parts:
        containers_by_course.setdefault(part.course, []).extend(part.containers)

    username, password = _get_gradescope_credentials()
    client = GradescopeClient()
    for course, containers in containers_by_course.items():
        client.save_grading_snapshot(
            course,
            containers,
            snapshot_dir / course,
            scores=scores,
            headless=headless,
            username=username,
            password=password,
        )
    (snapshot_dir / SNAPSHOT_MARKER).write_text(
        json.dumps({"exam": exam.key, "started": started.isoformat(timespec="seconds"), "scores": scores})
    )
    logger.success(f"Saved grading snapshot of '{exam.title}' to {snapshot_dir}")
    return snapshot_dir


def snapshots(exam: Exam) -> list[Path]:
    """Complete snapshot directories of an exam, oldest first."""
    if not exam.raw_dir.exists():
        return []
    return sorted(d for d in exam.raw_dir.iterdir() if (d / SNAPSHOT_MARKER).exists())


def prune_snapshots(exam: Exam, keep: int) -> None:
    """Delete all but the newest ``keep`` snapshots (progress history survives in history.csv)."""
    if keep <= 0:
        return
    for old in snapshots(exam)[:-keep]:
        shutil.rmtree(old)
        logger.debug(f"Pruned old grading snapshot {old}")


# ---------------------------------------------------------------------------
# process
# ---------------------------------------------------------------------------


def _short_labels(labels: list[str]) -> list[str]:
    """Drop the words that all labels share, wherever they are.

    "Chapter 5 and 6 Test Section 16 Free Response Version A" -> "Section 16 A".
    """
    words = [label.split() for label in labels]
    if len(set(labels)) < 2:
        return labels
    shared = set(words[0]).intersection(*words[1:])
    short = [[w for w in ws if w not in shared] for ws in words]
    if any(not ws for ws in short) or len({" ".join(ws) for ws in short}) < len(set(labels)):
        return labels
    return [" ".join(ws) for ws in short]


def _number_key(number: str) -> tuple:
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"\.", str(number)))


@dataclass
class Snapshot:
    fetched_at: datetime
    versions: pd.DataFrame  # part, course, container, assignment_id, label, version
    progress: pd.DataFrame  # one row per leaf question per version
    scores: pd.DataFrame  # one row per submission (Status != Missing) per version


def load_snapshot(exam: Exam, snapshot_dir: Path) -> Snapshot:
    """Read a snapshot saved by ``_get_impl``.

    Raises:
        FileNotFoundError / ValueError: If a configured container is missing from it.
    """
    from edubag.gradescope.grading import dashboard_progress
    from edubag.gradescope.scoresheet import Scoresheet

    dashboards: dict[str, dict] = {}
    version_rows, progress_frames, score_frames = [], [], []
    fetched = []
    for part in exam.parts:
        if part.course not in dashboards:
            path = snapshot_dir / part.course / "dashboards.json"
            dashboards[part.course] = json.loads(path.read_text())
            fetched.append(datetime.fromisoformat(dashboards[part.course]["fetched_at"]))
        data = dashboards[part.course]
        containers = {str(c["id"]): c for c in data["containers"]}
        for cid in part.containers:
            if cid not in containers:
                raise ValueError(f"Container {cid} of part {part.name} is not in snapshot {snapshot_dir}")
            container = containers[cid]
            for v in container["versions"]:
                aid = str(v["id"])
                label = v.get("title") or f"{container['title']} {v.get('version_name') or aid}"
                version_rows.append(
                    {"part": part.name, "course": part.course, "container": cid, "assignment_id": aid, "label": label}
                )
                prog = dashboard_progress(v["dashboard"])
                prog = prog[prog["leaf"]].copy()
                renumber = part.question_map.get(aid, {})
                prog["number"] = prog["number"].map(lambda n: renumber.get(n, n))
                prog["part"] = part.name
                progress_frames.append(prog)
                if v.get("scores_csv"):
                    sc = Scoresheet.from_csv(snapshot_dir / part.course / v["scores_csv"], drop_missing=True).scores
                    if not sc.empty:
                        sc = sc.dropna(axis=1, how="all").copy()
                        sc["part"] = part.name
                        sc["assignment_id"] = aid
                        score_frames.append(sc)

    versions = pd.DataFrame(version_rows)
    versions["version"] = versions["label"]
    for part_name, idx in versions.groupby("part").groups.items():
        versions.loc[idx, "version"] = _short_labels(versions.loc[idx, "label"].tolist())
    version_of = versions.set_index("assignment_id")["version"]

    progress = pd.concat(progress_frames, ignore_index=True)
    progress["version"] = progress["assignment_id"].map(version_of)
    scores = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame(
        columns=["Email", "Status", "Total Score", "Max Points", "part", "assignment_id"]
    )
    scores["version"] = scores["assignment_id"].map(version_of)
    return Snapshot(fetched_at=min(fetched), versions=versions, progress=progress, scores=scores)


def load_roster(exam: Exam) -> pd.DataFrame:
    """Students of the newest Gradescope roster matching the exam's ``roster`` glob."""
    dated = sorted(d for d in ROSTERS_DIR.iterdir() if d.is_dir()) if ROSTERS_DIR.exists() else []
    for d in reversed(dated):
        matches = sorted(d.glob(exam.roster))
        if len(matches) > 1:
            raise ValueError(f"Roster glob '{exam.roster}' matches several files in {d}: {matches}")
        if matches:
            roster = pd.read_csv(matches[0], dtype=str)
            logger.info(f"Using roster {matches[0]}")
            if "Role" in roster.columns:
                roster = roster[roster["Role"].str.casefold() == "student"]
            return roster.reset_index(drop=True)
    raise FileNotFoundError(f"No roster matching '{exam.roster}' in {ROSTERS_DIR}/<date>/")


def build_students(exam: Exam, snap: Snapshot, roster: pd.DataFrame) -> pd.DataFrame:
    """One row per student on the roster or with a submission, with each part's submission."""
    roster = roster.copy()
    roster["key"] = roster["Email"].str.strip().str.casefold()
    people = roster.reindex(columns=["key", "First Name", "Last Name", "SID", "Email", "Section"])  # single-section courses have no Section column
    people["On Roster"] = True

    scores = snap.scores[snap.scores["Email"].notna()].copy()  # see build_unmatched
    scores["key"] = scores["Email"].str.strip().str.casefold()
    extra = scores[~scores["key"].isin(people["key"])].drop_duplicates("key")
    if not extra.empty:
        extra = extra.reindex(columns=["key", "First Name", "Last Name", "SID", "Email", "Sections"])
        extra = extra.rename(columns={"Sections": "Section"})
        extra["On Roster"] = False
        people = pd.concat([people, extra], ignore_index=True)

    students = people.set_index("key")
    for part in exam.parts:
        p = part.name
        sub = scores[scores["part"] == p]
        if "Submission Time" in sub.columns:
            sub = sub.sort_values("Submission Time")
        counts = sub.groupby("key").size()
        last = sub.drop_duplicates("key", keep="last").set_index("key")
        students[f"{p} Version"] = last["version"]
        students[f"{p} Status"] = last["Status"].astype("string")
        students[f"{p} Score"] = pd.to_numeric(last["Total Score"], errors="coerce")
        students[f"{p} Max"] = pd.to_numeric(last["Max Points"], errors="coerce")
        students[f"{p} Submission ID"] = last.get("Submission ID")
        students[f"{p} Submissions"] = counts.reindex(students.index).fillna(0).astype(int)

    names = [p.name for p in exam.parts]
    has = pd.DataFrame({p: students[f"{p} Submissions"] > 0 for p in names})

    def completeness(row: pd.Series) -> str:
        missing = [p for p in names if not row[p]]
        if not missing:
            return "complete"
        if len(missing) == len(names):
            return "none"
        return "missing " + ", ".join(missing)

    students["Completeness"] = has.apply(completeness, axis=1)
    graded = pd.concat([students[f"{p} Status"] == "Graded" for p in names], axis=1).all(axis=1)
    total = sum(students[f"{p} Score"] for p in names)
    total_max = sum(students[f"{p} Max"] for p in names)
    students["Total Score"] = total.where(graded)
    students["Total Max"] = total_max.where(graded)
    students["Total Percent"] = (students["Total Score"] / students["Total Max"] * 100).round(2)
    return students.reset_index(drop=True).sort_values(["Last Name", "First Name"], na_position="last")


def build_unmatched(snap: Snapshot) -> pd.DataFrame:
    """Submissions not (yet) matched to a student: Gradescope lists them without an email."""
    unmatched = snap.scores[snap.scores["Email"].isna()]
    course_of = snap.versions.set_index("assignment_id")["course"]
    result = pd.DataFrame(
        {
            "part": unmatched["part"],
            "version": unmatched["version"],
            "course": unmatched["assignment_id"].map(course_of),
            "assignment_id": unmatched["assignment_id"],
            "Submission ID": unmatched.get("Submission ID"),
            "Status": unmatched["Status"],
        }
    )
    return result.reset_index(drop=True)


def build_progress(exam: Exam, snap: Snapshot) -> pd.DataFrame:
    """Grading progress per problem, per version and across all versions (version "ALL")."""
    prog = snap.progress.copy()
    prog["problem"] = prog["part"] + "." + prog["number"]
    prog["graded"] = prog["graded"].astype(int)
    prog["total"] = prog["total"].astype(int)
    per_version = prog[["part", "problem", "number", "version", "assignment_id", "title", "points", "graded", "total"]].copy()
    per_version["graders"] = prog["graders"].map(lambda g: "; ".join(sorted(set(g))))

    rows = []
    for (part, problem), g in prog.groupby(["part", "problem"], sort=False):
        if g["points"].nunique() > 1:
            logger.warning(f"{problem} has different point values across versions: {sorted(g['points'].unique())}")
        rows.append(
            {
                "part": part,
                "problem": problem,
                "number": g["number"].iloc[0],
                "version": "ALL",
                "assignment_id": "",
                "title": g["title"].iloc[0],
                "points": g["points"].max(),
                "graded": int(g["graded"].sum()),
                "total": int(g["total"].sum()),
                "graders": "; ".join(sorted({name for names in g["graders"] for name in names})),
            }
        )
    result = pd.concat([pd.DataFrame(rows), per_version], ignore_index=True)
    result["remaining"] = result["total"] - result["graded"]
    result["percent"] = (result["graded"] / result["total"].where(result["total"] > 0) * 100).round(1)
    result["responsible"] = [
        "; ".join(exam.responsible(part, problem)) for part, problem in zip(result["part"], result["problem"])
    ]
    part_order = {p.name: i for i, p in enumerate(exam.parts)}
    result["_order"] = [
        (part_order[p], _number_key(n), v != "ALL", v) for p, n, v in zip(result["part"], result["number"], result["version"])
    ]
    return result.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _describe(values: pd.Series) -> dict:
    values = values.dropna()
    if values.empty:
        return {"n_graded": 0}
    return {
        "n_graded": int(values.size),
        "mean": round(values.mean(), 2),
        "sd": round(values.std(), 2) if values.size > 1 else None,
        "min": round(values.min(), 2),
        "q1": round(values.quantile(0.25), 2),
        "median": round(values.median(), 2),
        "q3": round(values.quantile(0.75), 2),
        "max": round(values.max(), 2),
    }


def build_stats(exam: Exam, snap: Snapshot, students: pd.DataFrame) -> pd.DataFrame:
    """Percent-score statistics over fully graded submissions.

    Only ``Status == "Graded"`` counts: Gradescope's scores CSV fills ungraded questions with
    scores too (not blanks), so a partly graded submission's total is meaningless.

    Each part by version and pooled ("ALL"), and the exam total (students with every part
    graded) pooled and by roster section.
    """
    scores = snap.scores.copy()
    scores["percent"] = pd.to_numeric(scores["Total Score"], errors="coerce") / pd.to_numeric(
        scores["Max Points"], errors="coerce"
    ) * 100
    rows = []
    for part in exam.parts:
        sub = scores[scores["part"] == part.name]
        versions = snap.versions[snap.versions["part"] == part.name]["version"].tolist()
        for version in ["ALL", *versions]:
            g = sub if version == "ALL" else sub[sub["version"] == version]
            if version != "ALL" and g.empty:
                continue
            graded = g[g["Status"] == "Graded"]
            rows.append(
                {"scope": part.name, "group": version, "n_submitted": len(g), **_describe(graded["percent"])}
            )
    on_roster = students[students["On Roster"]]
    complete = on_roster[on_roster["Completeness"] == "complete"]
    rows.append(
        {"scope": "Total", "group": "ALL", "n_submitted": len(complete), **_describe(complete["Total Percent"])}
    )
    for section, g in complete.groupby("Section", dropna=True):
        rows.append(
            {"scope": "Total", "group": f"Section {section}", "n_submitted": len(g), **_describe(g["Total Percent"])}
        )
    return pd.DataFrame(rows)


def update_history(exam: Exam, snap: Snapshot, progress: pd.DataFrame) -> pd.DataFrame:
    """Append this snapshot's per-problem counts to history.csv (once per snapshot)."""
    path = exam.processed_dir / "history.csv"
    fetched_at = snap.fetched_at.isoformat(timespec="seconds")
    current = progress[progress["version"] == "ALL"][["part", "problem", "graded", "total"]].copy()
    current.insert(0, "fetched_at", fetched_at)
    if path.exists():
        history = pd.read_csv(path, dtype={"fetched_at": str})
        history = pd.concat([history[history["fetched_at"] != fetched_at], current], ignore_index=True)
    else:
        history = current
    history = history.sort_values("fetched_at", kind="stable")
    history.to_csv(path, index=False)
    return history


def grading_rates(exam: Exam, progress: pd.DataFrame, history: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """Per problem: recent grading rate (per hour), projected finish, and status.

    Status is "done", "stalled" (nothing graded during the last RATE_WINDOW), "behind" (the
    projected finish is after the exam's deadline), "on track", or "" when there is not yet
    enough history to tell.
    """
    history = history.copy()
    history["at"] = pd.to_datetime(history["fetched_at"], utc=True)
    now_utc = pd.Timestamp(now).tz_convert("UTC")
    rows = []
    for (part, problem), g in progress[progress["version"] == "ALL"].groupby(["part", "problem"], sort=False):
        remaining = int(g["remaining"].iloc[0])
        h = history[history["problem"] == problem].sort_values("at")
        rate = finish = None
        status = ""
        if remaining == 0 and int(g["total"].iloc[0]) > 0:
            status = "done"
        elif len(h) >= 2:
            latest = h.iloc[-1]
            window = h[h["at"] >= latest["at"] - RATE_WINDOW]
            start = window.iloc[0] if len(window) >= 2 else h.iloc[-2]
            span = latest["at"] - start["at"]
            if span >= MIN_RATE_SPAN:
                rate = (latest["graded"] - start["graded"]) / (span.total_seconds() / 3600)
                if rate > 0:
                    finish = now_utc + pd.Timedelta(hours=remaining / rate)
                if rate <= 0 and span >= RATE_WINDOW * 0.9:
                    status = "stalled"
                elif exam.deadline is not None and (finish is None or finish > pd.Timestamp(exam.deadline)):
                    status = "behind"
                elif rate > 0:
                    status = "on track"
        rows.append(
            {
                "problem": problem,
                "rate_per_hour": None if rate is None else round(rate, 1),
                "finish": None if finish is None else finish.tz_convert(now.tzinfo).isoformat(timespec="minutes"),
                "status": status,
            }
        )
    return pd.DataFrame(rows, columns=["problem", "rate_per_hour", "finish", "status"])


def _process_impl(exam_key: Optional[str] = None, snapshot_dir: Optional[Path] = None) -> Path:
    """Process a snapshot (default: the newest) into the exam's processed directory."""
    exam = load_exam(exam_key)
    if snapshot_dir is None:
        found = snapshots(exam)
        if not found:
            raise FileNotFoundError(
                f"No grading snapshots in {exam.raw_dir}; run `dataset get gradescope grading {exam.key}`"
            )
        snapshot_dir = found[-1]
    logger.info(f"Processing grading snapshot {snapshot_dir}")
    snap = load_snapshot(exam, snapshot_dir)
    students = build_students(exam, snap, load_roster(exam))
    unmatched = build_unmatched(snap)
    progress = build_progress(exam, snap)
    stats = build_stats(exam, snap, students)

    out = exam.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    history = update_history(exam, snap, progress)
    rates = grading_rates(exam, progress, history, snap.fetched_at)
    progress = progress.merge(rates, on="problem", how="left")
    progress.loc[progress["version"] != "ALL", ["rate_per_hour", "finish", "status"]] = None

    students.to_csv(out / "students.csv", index=False)
    unmatched.to_csv(out / "unmatched.csv", index=False)
    progress.to_csv(out / "progress.csv", index=False)
    stats.to_csv(out / "stats.csv", index=False)
    (out / "snapshot.json").write_text(
        json.dumps({"snapshot": str(snapshot_dir), "fetched_at": snap.fetched_at.isoformat(timespec="seconds")})
    )
    logger.success(f"Wrote grading data for '{exam.title}' to {out}")
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

_STATUS_BADGES = {
    "done": ("✓", "Done", "good"),
    "on track": ("→", "On track", "good"),
    "behind": ("!", "Behind", "warning"),
    "stalled": ("■", "Stalled", "critical"),
}


def _e(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
        return ""
    return html.escape(str(value))


def _fmt_time(iso: Optional[str], now: datetime) -> str:
    if not iso or (isinstance(iso, float) and pd.isna(iso)):
        return ""
    t = datetime.fromisoformat(iso)
    return t.strftime("%H:%M") if t.date() == now.date() else t.strftime("%a %H:%M")


def _bar(percent: float, label: str) -> str:
    pct = 0.0 if pd.isna(percent) else max(0.0, min(100.0, float(percent)))
    return (
        f'<div class="bar" role="img" aria-label="{_e(label)}" title="{_e(label)}">'
        f'<div class="fill" style="width:{pct:.1f}%"></div></div>'
    )


def _histogram(values: pd.Series, title: str) -> str:
    """Inline SVG histogram of percent scores in 10-point bins."""
    values = values.dropna()
    bins = [0] * 10
    for v in values:
        bins[min(9, max(0, int(v // 10)))] += 1
    w, h, pad_l, pad_r, pad_b, pad_t = 320, 150, 28, 14, 22, 8
    plot_w, plot_h = w - pad_l - pad_r, h - pad_b - pad_t
    top = max(bins) or 1
    bar_w = plot_w / 10
    bars = []
    for i, n in enumerate(bins):
        bh = plot_h * n / top
        x = pad_l + i * bar_w + 1
        y = pad_t + plot_h - bh
        lo, hi = i * 10, 100 if i == 9 else i * 10 + 10
        tip = f"{lo}–{hi}%: {n} student{'s' if n != 1 else ''}"
        # Hit target spans the full column, bigger than the mark itself.
        bars.append(
            f'<g><title>{tip}</title>'
            f'<rect x="{pad_l + i * bar_w:.1f}" y="{pad_t}" width="{bar_w:.1f}" height="{plot_h}" fill="transparent"/>'
            + (f'<rect class="mark" x="{x:.1f}" y="{y:.1f}" width="{bar_w - 2:.1f}" height="{bh:.1f}" rx="2"/>' if n else "")
            + "</g>"
        )
    ticks = "".join(
        f'<text x="{pad_l + i * bar_w:.1f}" y="{h - 6}" class="tick" text-anchor="middle">{i * 10}</text>'
        for i in range(0, 11, 2)
    )
    mean = f" · mean {values.mean():.1f}%" if len(values) else ""
    return (
        f'<figure class="hist"><figcaption>{_e(title)} <span class="muted">n={len(values)}{mean}</span></figcaption>'
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{_e(title)} score distribution">'
        f'<line x1="{pad_l}" x2="{w - pad_r}" y1="{pad_t + plot_h}" y2="{pad_t + plot_h}" class="axis"/>'
        f'<text x="{pad_l - 4}" y="{pad_t + 8}" class="tick" text-anchor="end">{top}</text>'
        f"{''.join(bars)}{ticks}</svg></figure>"
    )


def _table(df: pd.DataFrame, columns: list[str], empty: str) -> str:
    if df.empty:
        return f'<p class="muted">{_e(empty)}</p>'
    head = "".join(f"<th>{_e(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{_e(r.get(c))}</td>" for c in columns) + "</tr>" for _, r in df.iterrows())
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render_dashboard(
    exam: Exam,
    students: pd.DataFrame,
    progress: pd.DataFrame,
    stats: pd.DataFrame,
    unmatched: pd.DataFrame,
    fetched_at: datetime,
    refresh_seconds: int = 60,
    watch_error: Optional[str] = None,
    now: Optional[datetime] = None,
) -> str:
    now = now or datetime.now().astimezone()
    names = [p.name for p in exam.parts]
    on_roster = students[students["On Roster"]]
    age_min = (now - fetched_at).total_seconds() / 60

    # --- tiles
    tiles = []
    for p in names:
        sub = students[students[f"{p} Submissions"] > 0]
        fully = int((sub[f"{p} Status"] == "Graded").sum())
        allrows = progress[(progress["part"] == p) & (progress["version"] == "ALL")]
        g, t = int(allrows["graded"].sum()), int(allrows["total"].sum())
        tiles.append(
            f'<div class="tile"><div class="tile-label">{_e(p)}</div>'
            f'<div class="tile-value">{(g / t * 100 if t else 0):.0f}%</div>'
            f'<div class="tile-sub">questions graded ({g:,}/{t:,})</div>'
            f'<div class="tile-sub">{len(sub)} submitted · {fully} fully graded</div></div>'
        )
    complete = int((on_roster["Completeness"] == "complete").sum())
    tiles.append(
        f'<div class="tile"><div class="tile-label">Complete exams</div>'
        f'<div class="tile-value">{complete}<span class="of">/{len(on_roster)}</span></div>'
        f'<div class="tile-sub">on roster with every part submitted</div>'
        f'<div class="tile-sub">{int(on_roster["Total Percent"].notna().sum())} fully graded</div></div>'
    )
    tiles.append(
        f'<div class="tile"><div class="tile-label">Unmatched submissions</div>'
        f'<div class="tile-value">{len(unmatched)}</div>'
        f'<div class="tile-sub">not yet assigned to a student</div></div>'
    )

    # --- progress table
    part_blocks = []
    for part in names:
        prog_rows = []
        part_all = progress[(progress["part"] == part) & (progress["version"] == "ALL")]
        for _, r in part_all.iterrows():
            versions = progress[(progress["problem"] == r["problem"]) & (progress["version"] != "ALL")]
            chips = " ".join(
                f'<span class="chip{" chip-done" if v["total"] and v["remaining"] == 0 else ""}" '
                f'title="{_e(v["version"])}: {v["graded"]}/{v["total"]} graded'
                f'{" by " + _e(v["graders"]) if v["graders"] else ""}">'
                f'{_e(v["version"])} {v["graded"]}/{v["total"]}</span>'
                for _, v in versions.iterrows()
                if v["total"] > 0
            )
            icon, label, tone = _STATUS_BADGES.get(r.get("status") or "", ("", "", ""))
            badge = f'<span class="badge {tone}">{icon} {label}</span>' if label else '<span class="muted">—</span>'
            pct = r["percent"]
            bar_label = f"{r['graded']} of {r['total']} graded"
            prog_rows.append(
                "<tr>"
                f'<td class="problem"><b>{_e(r["problem"])}</b><div class="muted small">{_e(r["title"])} · {r["points"]:g} pts</div></td>'
                f'<td class="progress-cell">{_bar(pct, bar_label)}'
                f'<div class="small"><b>{"" if pd.isna(pct) else f"{pct:.0f}%"}</b> · {r["graded"]}/{r["total"]} · '
                f'{r["remaining"]} left</div><div class="chips">{chips}</div></td>'
                f'<td class="num">{_e(r.get("rate_per_hour"))}</td>'
                f"<td>{_e(_fmt_time(r.get('finish'), now))}</td>"
                f"<td>{badge}</td>"
                f'<td>{_e(r["responsible"]) or "<span class=muted>unassigned</span>"}</td>'
                f'<td class="small">{_e(r["graders"])}</td>'
                "</tr>"
            )
        graded, total = int(part_all["graded"].sum()), int(part_all["total"].sum())
        done = total > 0 and graded == total
        summary = (
            f"{_e(part)} <span class='count'>{len(part_all)} problems · "
            f"{(graded / total * 100 if total else 0):.0f}% graded{' · all done ✓' if done else ''}</span>"
        )
        part_blocks.append(
            f"<details{'' if done else ' open'}><summary>{summary}</summary>"
            '<div class="scroll"><table class="progress"><thead><tr><th>Problem</th><th>Progress (all versions)</th>'
            "<th>Rate /h</th><th>ETA</th><th>Status</th><th>Responsible</th><th>Graded by</th></tr></thead>"
            f"<tbody>{''.join(prog_rows)}</tbody></table></div></details>"
        )
    progress_table = "".join(part_blocks)

    # --- stats
    stat_cols = ["scope", "group", "n_submitted", "n_graded", "mean", "sd", "min", "q1", "median", "q3", "max"]
    stats_table = _table(stats.reindex(columns=stat_cols), stat_cols, "No graded submissions yet.")
    hists = []
    for p in names:
        sub = students[(students[f"{p} Status"] == "Graded")]
        hists.append(_histogram(sub[f"{p} Score"] / sub[f"{p} Max"] * 100, f"{p} (fully graded)"))
    hists.append(_histogram(on_roster["Total Percent"], "Total (all parts graded)"))

    # --- missing data
    who = ["Last Name", "First Name", "Email", "Section"]
    part_cols = [c for p in names for c in (f"{p} Version", f"{p} Submission ID")]
    missing_sections = []
    for category in sorted(c for c in on_roster["Completeness"].unique() if c != "complete"):
        rows = on_roster[on_roster["Completeness"] == category]
        missing_sections.append(
            f"<h3>{_e(category.capitalize())} <span class='count'>{len(rows)}</span></h3>"
            + _table(rows, who + part_cols, "")
        )
    if unmatched.empty:
        unmatched_rows = '<p class="muted">None.</p>'
    else:
        items = "".join(
            f"<tr><td>{_e(u['part'])}</td><td>{_e(u['version'])}</td><td>{_e(u['Status'])}</td><td>"
            + (
                f'<a href="https://www.gradescope.com/courses/{_e(u["course"])}/assignments/'
                f'{_e(u["assignment_id"])}/submissions/{int(u["Submission ID"])}">{int(u["Submission ID"])}</a>'
                if pd.notna(u["Submission ID"])
                else ""
            )
            + "</td></tr>"
            for _, u in unmatched.iterrows()
        )
        unmatched_rows = (
            '<div class="scroll"><table><thead><tr><th>Part</th><th>Version</th><th>Status</th>'
            f"<th>Submission</th></tr></thead><tbody>{items}</tbody></table></div>"
        )
    missing_sections.insert(
        0,
        f"<h3>Submissions not matched to a student <span class='count'>{len(unmatched)}</span></h3>"
        "<p class='muted small'>Until these are matched, their students show up as missing that part.</p>"
        + unmatched_rows,
    )
    off = students[~students["On Roster"]]
    missing_sections.append(
        f"<h3>Submitted but not on roster <span class='count'>{len(off)}</span></h3>"
        + _table(off, who + part_cols, "None.")
    )
    dup = students[pd.concat([students[f"{p} Submissions"] > 1 for p in names], axis=1).any(axis=1)]
    missing_sections.append(
        f"<h3>More than one submission for a part <span class='count'>{len(dup)}</span></h3>"
        + _table(dup, who + [f"{p} Submissions" for p in names] + part_cols, "None.")
    )

    banners = []
    if watch_error:
        banners.append(f'<div class="banner critical">■ Last refresh failed: {_e(watch_error)}</div>')
    if age_min > 15:
        banners.append(
            f'<div class="banner warning">! Data is {age_min:.0f} minutes old '
            f"(fetched {fetched_at.strftime('%a %H:%M')}).</div>"
        )
    deadline = f" · deadline {exam.deadline.strftime('%a %b %-d %H:%M')}" if exam.deadline else ""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="{int(refresh_seconds)}">
<title>{_e(exam.title)} Grading</title>
<style>
:root {{
  --bg: {LIGHT_GRAY}; --panel: {WHITE}; --ink: #1d1029; --muted: {MEDIUM_GRAY1}; --line: {MEDIUM_GRAY2};
  --accent: {NYU_VIOLET}; --accent-dark: {DEEP_VIOLET}; --track: {LIGHT_VIOLET2}; --soft: {LIGHT_VIOLET1};
  --good: #1a7f37; --warning: #9a6700; --critical: #cf222e;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #17121c; --panel: #221a2a; --ink: #f1ecf6; --muted: #a99fb3; --line: #3d3347;
    --accent: #a77bd0; --accent-dark: #57068c; --track: #33283e; --soft: #6d4f8a;
    --good: #4ac26b; --warning: #d4a72c; --critical: #ff7b72;
  }}
}}
:root[data-theme="dark"] {{
  --bg: #17121c; --panel: #221a2a; --ink: #f1ecf6; --muted: #a99fb3; --line: #3d3347;
  --accent: #a77bd0; --accent-dark: #57068c; --track: #33283e; --soft: #6d4f8a;
  --good: #4ac26b; --warning: #d4a72c; --critical: #ff7b72;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font: 14px/1.45 ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--ink); }}
header {{ padding: 20px 16px; background: linear-gradient(120deg, {NYU_VIOLET}, {DEEP_VIOLET}); color: #fff; }}
header h1 {{ margin: 0 0 4px; font-size: 1.5rem; }}
header p {{ margin: 0; opacity: .85; }}
main {{ padding: 16px; max-width: 1280px; margin: 0 auto; }}
h2 {{ margin: 28px 0 10px; font-size: 1.15rem; }}
h3 {{ margin: 18px 0 6px; font-size: 1rem; }}
.count {{ font-weight: normal; color: var(--muted); }}
.muted {{ color: var(--muted); }} .small {{ font-size: .85em; }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
.tile {{ background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }}
.tile-label {{ color: var(--muted); font-size: .85em; text-transform: uppercase; letter-spacing: .04em; }}
.tile-value {{ font-size: 2rem; font-weight: 650; font-variant-numeric: tabular-nums; }}
.tile-value .of {{ font-size: 1rem; color: var(--muted); font-weight: normal; }}
.tile-sub {{ color: var(--muted); font-size: .85em; }}
.scroll {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); }}
th, td {{ padding: 7px 9px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
th {{ background: var(--track); font-weight: 600; white-space: nowrap; }}
td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
table.progress td.problem {{ white-space: nowrap; }}
.progress-cell {{ min-width: 260px; }}
.bar {{ height: 10px; background: var(--track); border-radius: 4px; overflow: hidden; margin: 3px 0 4px; }}
.fill {{ height: 100%; background: var(--accent); border-radius: 4px; }}
.chips {{ margin-top: 3px; display: flex; flex-wrap: wrap; gap: 4px; }}
.chip {{ font-size: .75em; padding: 1px 6px; border: 1px solid var(--line); border-radius: 9px; color: var(--muted); white-space: nowrap; }}
.chip-done {{ border-color: var(--soft); }}
.badge {{ font-size: .85em; font-weight: 600; white-space: nowrap; }}
.badge.good {{ color: var(--good); }} .badge.warning {{ color: var(--warning); }} .badge.critical {{ color: var(--critical); }}
.banner {{ margin: 0 0 12px; padding: 10px 12px; border-radius: 8px; background: var(--panel); border: 1px solid currentColor; font-weight: 600; }}
.banner.warning {{ color: var(--warning); }} .banner.critical {{ color: var(--critical); }}
details {{ margin-bottom: 10px; }}
summary {{ cursor: pointer; font-weight: 600; padding: 6px 0; }}
.hists {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-bottom: 12px; }}
.hist {{ margin: 0; background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 10px; }}
.hist figcaption {{ font-weight: 600; margin-bottom: 4px; }}
.hist svg {{ width: 100%; height: auto; display: block; }}
.hist .mark {{ fill: var(--accent); }}
.hist g:hover .mark {{ fill: var(--accent-dark); }}
.hist .axis {{ stroke: var(--line); }}
.hist .tick {{ fill: var(--muted); font-size: 10px; }}
</style>
</head>
<body>
<header>
  <h1>{_e(exam.title)} — grading</h1>
  <p>Gradescope data fetched {fetched_at.strftime('%a %b %-d %H:%M')} · page refreshes every {int(refresh_seconds)} s{deadline}</p>
</header>
<main>
{''.join(banners)}
<div class="tiles">{''.join(tiles)}</div>

<h2>Grading progress by problem</h2>
{progress_table}
<p class="muted small">Progress counts graded submissions per question across all versions (chips show each version).
Rate is graded per hour over the last {int(RATE_WINDOW.total_seconds() // 60)} minutes; ETA projects that rate.
“Graded by” comes from Gradescope; “Responsible” from pyproject.toml.</p>

<h2>Scores (percent, fully graded submissions)</h2>
<div class="hists">{''.join(hists)}</div>
{stats_table}

<h2>Missing data</h2>
{''.join(missing_sections)}
</main>
</body>
</html>
"""


def read_students(exam: Exam) -> pd.DataFrame:
    """Read students.csv back with its column types.

    Submission IDs are blank for students missing a part, so without explicit nullable integer
    types pandas would read them as floats ("123.0").
    """
    dtype = {"SID": str, "Section": str}
    for part in exam.parts:
        dtype[f"{part.name} Submission ID"] = "Int64"
        dtype[f"{part.name} Version"] = str
        dtype[f"{part.name} Status"] = str
    return pd.read_csv(exam.processed_dir / "students.csv", dtype=dtype)


def _watch_status_path(exam: Exam) -> Path:
    return exam.processed_dir / "watch.json"


def _report_impl(exam_key: Optional[str] = None, refresh_seconds: int = 60) -> Path:
    """Render dashboard.html from the processed files."""
    exam = load_exam(exam_key)
    out = exam.processed_dir
    if not (out / "snapshot.json").exists():
        raise FileNotFoundError(f"No processed grading data in {out}; run `dataset process gradescope-grading`")
    meta = json.loads((out / "snapshot.json").read_text())
    students = read_students(exam)
    progress = pd.read_csv(out / "progress.csv", dtype={"number": str, "assignment_id": str}, keep_default_na=True)
    progress["graders"] = progress["graders"].fillna("")
    progress["responsible"] = progress["responsible"].fillna("")
    stats = pd.read_csv(out / "stats.csv")
    unmatched = pd.read_csv(out / "unmatched.csv", dtype={"course": str, "assignment_id": str})
    watch_error = None
    status_path = _watch_status_path(exam)
    if status_path.exists():
        watch_error = json.loads(status_path.read_text()).get("last_error")
    page = render_dashboard(
        exam,
        students,
        progress,
        stats,
        unmatched,
        datetime.fromisoformat(meta["fetched_at"]),
        refresh_seconds=refresh_seconds,
        watch_error=watch_error,
    )
    path = out / "dashboard.html"
    path.write_text(page, encoding="utf-8")
    logger.success(f"Wrote grading dashboard {path}")
    return path


# ---------------------------------------------------------------------------
# watch
# ---------------------------------------------------------------------------


def parse_duration(text: str) -> int:
    """Seconds in a duration like "90", "90s", "3m", "1h"."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*", text)
    if not match:
        raise ValueError(f"Not a duration: {text!r} (use e.g. 90s, 3m, 1h)")
    return int(float(match[1]) * {"": 1, "s": 1, "m": 60, "h": 3600}[match[2]])


def _watch_impl(
    exam_key: Optional[str] = None,
    every: int = 180,
    headless: bool = True,
    keep: int = 20,
    open_browser: bool = True,
) -> None:
    """Get, process and report every ``every`` seconds until interrupted."""
    exam = load_exam(exam_key)
    exam.processed_dir.mkdir(parents=True, exist_ok=True)
    status_path = _watch_status_path(exam)
    refresh = max(15, min(60, every))
    opened = False
    logger.info(f"Watching grading of '{exam.title}' every {every} s; Ctrl-C to stop")
    try:
        while True:
            started = time.monotonic()
            status = {"last_attempt": datetime.now().astimezone().isoformat(timespec="seconds"), "last_error": None}
            try:
                _get_impl(exam.key, headless=headless)
            except Exception as exc:  # keep watching; the page shows the last good data
                logger.exception(f"Fetching from Gradescope failed: {exc}")
                status["last_error"] = f"{type(exc).__name__}: {exc}".splitlines()[0][:300]
            status_path.write_text(json.dumps(status))
            try:
                _process_impl(exam.key)
                page = _report_impl(exam.key, refresh_seconds=refresh)
                if open_browser and not opened:
                    webbrowser.open(page.resolve().as_uri())
                    opened = True
            except Exception as exc:
                logger.exception(f"Processing grading data failed: {exc}")
            prune_snapshots(exam, keep)
            elapsed = time.monotonic() - started
            time.sleep(max(0.0, every - elapsed))
    except KeyboardInterrupt:
        logger.info("Stopped watching.")
