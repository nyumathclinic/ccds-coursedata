from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from coursedata.dataset.gradescope import grading
from coursedata.dataset.gradescope.grading import (
    Exam,
    Part,
    Snapshot,
    _short_labels,
    build_progress,
    build_stats,
    build_students,
    build_unmatched,
    grading_rates,
    parse_duration,
    render_dashboard,
)

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)


def _exam(**kwargs) -> Exam:
    return Exam(
        key="t1",
        title="Test 1",
        roster="*.csv",
        parts=[Part("MC", "1", ["10"]), Part("FR", "1", ["20"])],
        graders={"FR.1": ["Ann"], "FR": ["Bo"]},
        **kwargs,
    )


def _snapshot() -> Snapshot:
    versions = pd.DataFrame(
        {
            "part": ["MC", "FR", "FR"],
            "course": ["1", "1", "1"],
            "container": ["10", "20", "20"],
            "assignment_id": ["101", "201", "202"],
            "label": ["T MC A", "T FR A", "T FR B"],
            "version": ["A", "A", "B"],
        }
    )
    progress = pd.DataFrame(
        {
            "part": ["MC", "FR", "FR", "FR", "FR"],
            "assignment_id": ["101", "201", "201", "202", "202"],
            "version": ["A", "A", "A", "B", "B"],
            "number": ["1", "1", "2", "1", "2"],
            "title": ["Q1", "Lim", "Ser", "Lim", "Ser"],
            "points": [2.0, 9.0, 8.0, 9.0, 8.0],
            "graded": [3, 2, 0, 1, 1],
            "total": [3, 2, 2, 1, 1],
            "graders": [["AI"], ["Ann"], [], ["Cy"], ["Ann"]],
        }
    )

    def sub(email, part, aid, version, status, score, max_points, sid):
        return {
            "First Name": "F",
            "Last Name": email or "Student",
            "SID": None,
            "Email": email,
            "Sections": "016",
            "part": part,
            "assignment_id": aid,
            "version": version,
            "Status": status,
            "Total Score": score,
            "Max Points": max_points,
            "Submission ID": sid,
            "Submission Time": "2026-10-10",
        }

    scores = pd.DataFrame(
        [
            sub("a@x.edu", "MC", "101", "A", "Graded", 2.0, 2.0, 1),
            sub("b@x.edu", "MC", "101", "A", "Graded", 1.0, 2.0, 2),
            sub("C@x.edu", "MC", "101", "A", "Graded", 0.0, 2.0, 3),
            sub("a@x.edu", "FR", "201", "A", "Graded", 17.0, 17.0, 4),
            sub("b@x.edu", "FR", "201", "A", "Ungraded", 9.0, 17.0, 5),
            sub("z@x.edu", "FR", "202", "B", "Ungraded", 0.0, 17.0, 6),
            sub(None, "FR", "202", "B", "Ungraded", 0.0, 17.0, 7),
        ]
    )
    return Snapshot(fetched_at=NOW, versions=versions, progress=progress, scores=scores)


ROSTER = pd.DataFrame(
    {
        "First Name": ["A", "B", "C", "D"],
        "Last Name": ["Aa", "Bb", "Cc", "Dd"],
        "SID": ["N10000001", "N10000002", "N10000003", "N10000004"],
        "Email": ["a@x.edu", "b@x.edu", "c@x.edu", "d@x.edu"],
        "Role": ["Student"] * 4,
        "Section": ["16", "16", "21", "21"],
    }
)


def test_short_labels_drop_shared_words():
    labels = [
        "Ch 5 Test Section 16 Free Response Version A",
        "Ch 5 Test Section 21 Free Response Version A",
        "Ch 5 Test Makeup Free Response Version B",
    ]
    assert _short_labels(labels) == ["Section 16 A", "Section 21 A", "Makeup B"]
    assert _short_labels(["Only one"]) == ["Only one"]


@pytest.mark.parametrize("text, seconds", [("90", 90), ("90s", 90), ("3m", 180), ("1h", 3600), ("1.5m", 90)])
def test_parse_duration(text, seconds):
    assert parse_duration(text) == seconds


def test_parse_duration_rejects_garbage():
    with pytest.raises(ValueError):
        parse_duration("soon")


def test_build_students_completeness_and_totals():
    exam, snap = _exam(), _snapshot()
    students = build_students(exam, snap, ROSTER).set_index("Email")
    assert students.loc["a@x.edu", "Completeness"] == "complete"
    assert students.loc["a@x.edu", "Total Percent"] == pytest.approx(100.0)
    # FR not fully graded yet: no total.
    assert students.loc["b@x.edu", "Completeness"] == "complete"
    assert pd.isna(students.loc["b@x.edu", "Total Percent"])
    # Email matching ignores case.
    assert students.loc["c@x.edu", "Completeness"] == "missing FR"
    assert students.loc["d@x.edu", "Completeness"] == "none"
    # Submitted but not on the roster; the unmatched (email-less) submission is not a student.
    assert not students.loc["z@x.edu", "On Roster"]
    assert students.loc["z@x.edu", "Completeness"] == "missing MC"
    assert len(students) == 5


def test_build_unmatched():
    unmatched = build_unmatched(_snapshot())
    assert unmatched["Submission ID"].tolist() == [7]
    assert unmatched["course"].tolist() == ["1"]


def test_build_progress_aggregates_versions_and_responsibility():
    progress = build_progress(_exam(), _snapshot())
    all_rows = progress[progress["version"] == "ALL"].set_index("problem")
    assert list(all_rows.index) == ["MC.1", "FR.1", "FR.2"]
    assert all_rows.loc["FR.1", ["graded", "total", "remaining"]].tolist() == [3, 3, 0]
    assert all_rows.loc["FR.2", ["graded", "total"]].tolist() == [1, 3]
    assert all_rows.loc["FR.1", "graders"] == "Ann; Cy"
    assert all_rows.loc["FR.1", "responsible"] == "Ann"  # problem entry
    assert all_rows.loc["FR.2", "responsible"] == "Bo"  # part fallback
    assert all_rows.loc["MC.1", "responsible"] == ""
    assert len(progress[progress["problem"] == "FR.2"]) == 3  # ALL + two versions


def test_build_stats_uses_fully_graded_only():
    exam, snap = _exam(), _snapshot()
    stats = build_stats(exam, snap, build_students(exam, snap, ROSTER)).set_index(["scope", "group"])
    assert stats.loc[("MC", "ALL"), "n_graded"] == 3
    assert stats.loc[("MC", "ALL"), "mean"] == pytest.approx(50.0)
    assert stats.loc[("FR", "ALL"), "n_submitted"] == 4  # includes the unmatched submission
    assert stats.loc[("FR", "ALL"), "n_graded"] == 1
    assert stats.loc[("Total", "ALL"), "n_graded"] == 1


def _history(points: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fetched_at": [(NOW - timedelta(minutes=m)).isoformat() for m, _ in points],
            "part": "FR",
            "problem": "FR.2",
            "graded": [g for _, g in points],
            "total": 100,
        }
    )


def _rates(exam, history, graded=40):
    progress = pd.DataFrame(
        {"part": ["FR"], "problem": ["FR.2"], "version": ["ALL"], "graded": [graded], "total": [100], "remaining": [100 - graded]}
    )
    return grading_rates(exam, progress, history, NOW).iloc[0]


def test_grading_rates_eta_and_status():
    r = _rates(_exam(), _history([(30, 10), (15, 25), (0, 40)]))
    assert r["rate_per_hour"] == pytest.approx(60.0)
    assert r["finish"] == (NOW + timedelta(hours=1)).isoformat(timespec="minutes")
    assert r["status"] == "on track"

    behind = _rates(_exam(deadline=NOW + timedelta(minutes=30)), _history([(30, 10), (0, 40)]))
    assert behind["status"] == "behind"

    stalled = _rates(_exam(), _history([(30, 40), (0, 40)]))
    assert stalled["status"] == "stalled"

    assert _rates(_exam(), _history([(0, 40)]))["status"] == ""


def test_render_dashboard_smoke():
    exam, snap = _exam(), _snapshot()
    students = build_students(exam, snap, ROSTER)
    progress = build_progress(exam, snap)
    progress = progress.merge(grading_rates(exam, progress, _history([(0, 1)]), NOW), on="problem", how="left")
    page = render_dashboard(
        exam,
        students,
        progress,
        build_stats(exam, snap, students),
        build_unmatched(snap),
        NOW,
        watch_error="Timeout <x>",
        now=NOW + timedelta(minutes=20),
    )
    assert "Test 1" in page
    assert "Last refresh failed: Timeout &lt;x&gt;" in page
    # Staleness is shown by the reader's browser, from the embedded fetch time.
    assert f'data-fetched="{NOW.isoformat()}"' in page
    assert 'id="stale" hidden' in page
    assert 'id="pause"' in page and "http-equiv" not in page
    assert "/assignments/202/submissions/7" in page


def _public_page(snap=None, exam=None):
    exam, snap = exam or _exam(), snap or _snapshot()
    students = build_students(exam, snap, ROSTER)
    progress = build_progress(exam, snap)
    progress = progress.merge(grading_rates(exam, progress, _history([(0, 1)]), NOW), on="problem", how="left")
    args = (exam, students, progress, build_stats(exam, snap, students), build_unmatched(snap), NOW)
    return render_dashboard(*args, watch_error="Timeout at /Users/x", public=True), students


def test_public_dashboard_identifies_no_student():
    page, students = _public_page()
    for column in ("Email", "Last Name", "SID"):
        for value in students[column].dropna().astype(str):
            if value not in ("Student",):  # the unmatched placeholder name
                assert value not in page, (column, value)
    assert "submissions/7" not in page  # no links to unmatched submissions
    assert "/Users/x" not in page and "Last refresh failed" in page  # error details stay local
    # ...but the counts are there.
    assert "Missing FR" in page and "Not matched to a student: FR B" in page
    assert "Submitted but not on roster" in page
    assert "Lists of students are only on the instructor" in page
    assert "Ann; Cy" in page  # graders are staff, so they are shown
    assert "No part submitted" in page
    # Statistics of fewer than PUBLIC_MIN_GROUP graded exams would give away scores.
    assert "fewer than 5" in page and "Hidden until at least 5 are graded" in page
    assert ">100.0<" not in page  # the lone fully graded FR exam's score


def test_exam_keys_skip_publish(monkeypatch):
    monkeypatch.setattr(grading, "GRADING_CONFIG", {"t1": {}, "publish": {"host": "h"}})
    assert grading.exam_keys() == ["t1"]


def test_load_publish(monkeypatch):
    monkeypatch.setattr(grading, "GRADING_CONFIG", {})
    assert grading.load_publish() is None
    cfg = {"host": "access.example.edu", "user": "mpl5", "remote_dir": "/web/mpl5/grading/", "viewers": ["mpl5"]}
    monkeypatch.setattr(grading, "GRADING_CONFIG", {"publish": cfg})
    target = grading.load_publish()
    assert target.destination == "mpl5@access.example.edu"
    assert target.remote_dir == "/web/mpl5/grading"
    for bad in ({**cfg, "viewers": []}, {**cfg, "viewers": ["a b"]}, {k: v for k, v in cfg.items() if k != "host"}):
        monkeypatch.setattr(grading, "GRADING_CONFIG", {"publish": bad})
        with pytest.raises(ValueError):
            grading.load_publish()


def test_htaccess():
    text = grading.htaccess(["mpl5", "abc123"], 'Test "1" grading')
    assert "AuthType openid-connect" in text
    assert "Require claim sub:mpl5\nRequire claim sub:abc123" in text
    assert 'AuthName "Test 1 grading"' in text
    with pytest.raises(ValueError):
        grading.htaccess([], "x")


class FakeRun:
    def __init__(self, returncodes=()):
        self.calls, self.returncodes = [], list(returncodes)

    def __call__(self, args, **kwargs):
        self.calls.append(args)
        code = self.returncodes.pop(0) if self.returncodes else 0
        if kwargs.get("check") and code:
            raise RuntimeError(code)
        return type("Done", (), {"returncode": code})()


def test_publish_dashboard_commands(monkeypatch, tmp_path):
    exam = _exam()
    monkeypatch.setattr(grading, "PROCESSED_GRADING_DIR", tmp_path)
    monkeypatch.setattr(grading, "load_exam", lambda key=None: exam)
    monkeypatch.setattr(
        grading,
        "GRADING_CONFIG",
        {"publish": {"host": "h.edu", "user": "u", "remote_dir": "/web/u/g", "url": "https://h.edu/~u/g",
                     "viewers": ["u", "v1"]}},
    )  # fmt: skip
    with pytest.raises(FileNotFoundError):
        grading.publish_dashboard(run=FakeRun())
    public = tmp_path / "t1" / "public"
    public.mkdir(parents=True)
    (public / "index.html").write_text("<html></html>")
    run = FakeRun()
    assert grading.publish_dashboard(run=run) == "https://h.edu/~u/g/t1/"
    assert "Require claim sub:v1" in (public / ".htaccess").read_text()
    ssh, rsync = run.calls
    assert ssh[0] == "ssh" and "BatchMode=yes" in ssh and "u@h.edu" in ssh
    assert "mkdir -p /web/u/g/t1" in ssh[-1]
    assert rsync[0] == "rsync" and "-a" in rsync
    assert oct(public.stat().st_mode & 0o777) == "0o711"
    assert oct((public / ".htaccess").stat().st_mode & 0o777) == "0o644"
    assert "ControlPath=" in rsync[rsync.index("-e") + 1] and "BatchMode=yes" in rsync[rsync.index("-e") + 1]
    assert rsync[-2:] == [f"{public}/", "u@h.edu:/web/u/g/t1/"]


def test_ensure_connection(monkeypatch, tmp_path):
    monkeypatch.setattr(grading, "CONTROL_DIR", tmp_path / "cm")
    target = grading.PublishTarget(host="h.edu", remote_dir="/r", viewers=["u"], user="u")
    run = FakeRun([0])
    grading.ensure_connection(target, run=run)
    assert len(run.calls) == 1 and "-O" in run.calls[0]  # already open: only checked
    run = FakeRun([255, 0])
    grading.ensure_connection(target, run=run)
    assert run.calls[1][-4:] == ["-M", "-N", "-f", "u@h.edu"]
    assert (tmp_path / "cm").is_dir()


def test_load_exam(monkeypatch):
    monkeypatch.setattr(
        grading,
        "GRADING_CONFIG",
        {
            "t1": {
                "title": "Test 1",
                "course": "1",
                "roster": "r*.csv",
                "deadline": "2026-10-20T17:00",
                "parts": [
                    {"name": "MC", "containers": [10]},
                    {"name": "FR", "containers": ["20"], "course": "2", "question_map": {"201": {"3": "4"}}},
                ],
                "graders": {"FR.1": "Ann", "FR": ["Bo", "Cy"]},
            }
        },
    )
    exam = grading.load_exam()
    assert [p.course for p in exam.parts] == ["1", "2"]
    assert exam.parts[0].containers == ["10"]
    assert exam.parts[1].question_map == {"201": {"3": "4"}}
    assert exam.responsible("FR", "FR.1") == ["Ann"]
    assert exam.responsible("FR", "FR.9") == ["Bo", "Cy"]
    assert exam.deadline.tzinfo is not None
    with pytest.raises(ValueError):
        grading.load_exam("nope")


def test_load_exam_requires_parts(monkeypatch):
    monkeypatch.setattr(grading, "GRADING_CONFIG", {"t1": {"course": "1", "roster": "r.csv"}})
    with pytest.raises(ValueError, match="parts"):
        grading.load_exam("t1")
