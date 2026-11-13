import pytest
import typer

from coursedata import tasks


@pytest.fixture
def posted_courses(monkeypatch):
    course_ids = []

    class FakeGradescopeClient:
        def post_all_grades(self, course_id):
            course_ids.append(course_id)
            return []

    monkeypatch.setattr(tasks, "EDUBAG_AVAILABLE", True)
    monkeypatch.setattr(tasks, "GradescopeClient", FakeGradescopeClient)
    monkeypatch.setattr(tasks.keyring, "get_password", lambda service, username: "password")
    monkeypatch.setenv("GRADESCOPE_USERNAME", "")
    return course_ids


def test_post_grades_uses_task_specific_courses(monkeypatch, posted_courses):
    monkeypatch.setattr(
        tasks,
        "GRADESCOPE_CONFIG",
        {"courses": ["general-1", "general-2"], "post_grades_courses": ["posting-1"]},
    )

    tasks.post_gradescope_grades(username="poster")

    assert posted_courses == ["posting-1"]


def test_post_grades_falls_back_to_general_courses(monkeypatch, posted_courses):
    monkeypatch.setattr(tasks, "GRADESCOPE_CONFIG", {"courses": ["general-1"]})

    tasks.post_gradescope_grades(username="poster")

    assert posted_courses == ["general-1"]


def test_post_grades_cli_courses_override_config(monkeypatch, posted_courses):
    monkeypatch.setattr(
        tasks,
        "GRADESCOPE_CONFIG",
        {"courses": ["general-1"], "post_grades_courses": ["posting-1"]},
    )

    tasks.post_gradescope_grades(courses=["cli-1", "cli-2"], username="poster")

    assert posted_courses == ["cli-1", "cli-2"]


def test_post_grades_empty_task_specific_courses_do_not_fall_back(
    monkeypatch, posted_courses
):
    monkeypatch.setattr(
        tasks,
        "GRADESCOPE_CONFIG",
        {"courses": ["general-1"], "post_grades_courses": []},
    )

    with pytest.raises(typer.Exit):
        tasks.post_gradescope_grades(username="poster")

    assert posted_courses == []

def _capture_fetch_details(monkeypatch, tmp_path, gradescope_config):
    calls = []

    class FakeGradescopeClient:
        def fetch_class_details(self, *args, output=None, **kwargs):
            calls.append((args, kwargs))
            output.write_text("[]")
            return []

    monkeypatch.setattr(tasks, "EDUBAG_AVAILABLE", True)
    monkeypatch.setattr(tasks, "GRADESCOPE_SECTIONS_AVAILABLE", True)
    monkeypatch.setattr(tasks, "BRIGHTSPACE_AVAILABLE", True)
    monkeypatch.setattr(tasks, "GradescopeClient", FakeGradescopeClient)
    monkeypatch.setattr(tasks, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(tasks, "GRADESCOPE_CONFIG", gradescope_config)
    monkeypatch.delenv("GRADESCOPE_USERNAME", raising=False)

    # No LTI-linked courses in the (empty) details, so pairing must fail loudly.
    with pytest.raises(typer.Exit):
        tasks.sync_gradescope_sections()
    return calls


def test_sync_sections_fetches_configured_courses_by_id(monkeypatch, tmp_path):
    calls = _capture_fetch_details(monkeypatch, tmp_path, {"courses": ["1352536"]})

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ()
    assert kwargs["course_ids"] == ["1352536"]


def test_sync_sections_falls_back_to_course_name(monkeypatch, tmp_path):
    calls = _capture_fetch_details(monkeypatch, tmp_path, {})

    assert calls[0][0] == (tasks.COURSE_NAME, tasks.TERM_NAME)


# ----- upload-engagement ---------------------------------------------------------

import json as _json
from datetime import datetime as _datetime, timedelta as _timedelta


def _write_engagement(dir_path, computed_at):
    csv = dir_path / "engagement.csv"
    csv.write_text(
        "Username,Poll Participation Rate Points Grade,Office Hour Visits Points Grade,End-of-Line Indicator\n"
        "#ab123,50.0,0.0,#\n"
    )
    stamp = computed_at.isoformat(timespec="seconds")
    csv.with_suffix(".json").write_text(_json.dumps({"computed_at": stamp, "as_of": stamp}))
    return csv


@pytest.fixture
def fake_brightspace(monkeypatch):
    calls = []

    class FakeBrightspaceClient:
        def import_grades(self, course, csv_path, headless=True, dry_run=False):
            calls.append(("import", course, csv_path.name, dry_run))
            return {"status": "dry run" if dry_run else "imported", "students": 1, "changed": {}}

        def stamp_grade_item_descriptions(self, course, names, stamp, marker, headless=True, dry_run=False):
            calls.append(("stamp", course, list(names), stamp, marker))
            return {n: {"status": "updated"} for n in names}

    monkeypatch.setattr(tasks, "BRIGHTSPACE_AVAILABLE", True)
    monkeypatch.setattr(tasks, "BrightspaceClient", FakeBrightspaceClient)
    return calls


def test_upload_engagement_imports_then_stamps(tmp_path, monkeypatch, fake_brightspace):
    for day in ("2026-11-06", "2026-11-13"):
        (tmp_path / "grades" / day).mkdir(parents=True)
    now = _datetime(2026, 11, 13, 6, 5).astimezone()
    _write_engagement(tmp_path / "grades" / "2026-11-06", now - _timedelta(days=7))
    _write_engagement(tmp_path / "grades" / "2026-11-13", now)
    monkeypatch.setattr(tasks, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(tasks, "ENGAGEMENT_CONFIG", {
        "output_path": "grades/{date}/engagement.csv",
        "upload": {"course": "611388", "max_age_hours": 24 * 365 * 10},
    })

    tasks.upload_engagement()

    (_, course, name, dry_run), (_, _, names, stamp, marker) = fake_brightspace
    assert (course, name, dry_run) == ("611388", "engagement.csv", False)
    assert names == ["Poll Participation Rate", "Office Hour Visits"]
    assert marker == "Last updated"
    assert stamp.startswith("<p><em>Last updated Fri Nov 13, 2026, 6:05 AM.")


def test_upload_engagement_refuses_stale_scores(tmp_path, monkeypatch, fake_brightspace):
    csv = _write_engagement(tmp_path, _datetime.now().astimezone() - _timedelta(days=2))
    monkeypatch.setattr(tasks, "ENGAGEMENT_CONFIG", {"upload": {"course": "611388"}})

    with pytest.raises(typer.Exit):
        tasks.upload_engagement(csv_path=csv)
    assert fake_brightspace == []
