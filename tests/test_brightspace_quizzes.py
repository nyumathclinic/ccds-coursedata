from datetime import datetime, timedelta, timezone
import json

import pytest
import typer

from coursedata.dataset import brightspace

NOW = datetime(2026, 10, 29, 14, 0, tzinfo=timezone.utc)
PAST = "2026-10-27T12:00:00.000Z"
FUTURE = "2026-11-03T13:00:00.000Z"


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def test_quiz_courses_defaults_to_all_and_rejects_unknown(monkeypatch):
    monkeypatch.setattr(brightspace, "BRIGHTSPACE_CONFIG", {"courses": ["1", "2"]})
    assert brightspace._quiz_courses() == ["1", "2"]

    monkeypatch.setattr(brightspace, "BRIGHTSPACE_CONFIG", {"courses": ["1", "2"], "quizzes": {"courses": ["2"]}})
    assert brightspace._quiz_courses() == ["2"]

    monkeypatch.setattr(brightspace, "BRIGHTSPACE_CONFIG", {"courses": ["1"], "quizzes": {"courses": ["3"]}})
    with pytest.raises(typer.Exit):
        brightspace._quiz_courses()


def test_fetch_reason_fetches_once_then_once_more_after_due():
    quiz = {"id": "1", "due": PAST, "end": None}
    reason = brightspace._quiz_fetch_reason
    assert reason(quiz, None, [], NOW, False) == "new"

    before_due = {"fetched_at": "2026-10-26T12:00:00+00:00", "reports": {"10": {"name": "User Attempts"}}}
    assert reason(quiz, before_due, [], NOW, False) == "past due"

    after_due = {**before_due, "fetched_at": _iso(NOW - timedelta(hours=1))}
    assert reason(quiz, after_due, [], NOW, False) is None
    assert reason(quiz, after_due, [], NOW, True) == "forced"
    assert reason(quiz, {**after_due, "failed": True}, [], NOW, False) == "retry"
    # A newly configured report name that was never fetched.
    assert reason(quiz, after_due, ["Question Details"], NOW, False) == "new"


def test_fetch_reason_waits_for_future_due_date():
    quiz = {"id": "1", "due": FUTURE, "end": None}
    entry = {"fetched_at": _iso(NOW - timedelta(days=1)), "reports": {}}
    assert brightspace._quiz_fetch_reason(quiz, entry, [], NOW, False) is None


class FakeClient:
    def __init__(self, quiz_list):
        self.quiz_list = quiz_list
        self.saved = []

    def authenticate(self, **kwargs):
        pass

    def list_quizzes(self, course, headless=True):
        return self.quiz_list

    def save_quiz_reports(self, course, quiz_ids, reports, save_dir, headless=True):
        self.saved.append(list(quiz_ids))
        results = []
        for quiz_id in quiz_ids:
            path = save_dir / f"quiz {quiz_id} - User Attempts.xlsx"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"xlsx")
            results.append({"quiz_id": quiz_id, "quiz_name": f"quiz {quiz_id}", "report_id": "10",
                            "report": "User Attempts", "report_type": "User Attempts",
                            "path": path, "status": "saved"})
        return results


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeClient([
        {"id": "1", "name": "§8.2 Pre-Quiz", "due": PAST, "end": None},
        {"id": "2", "name": "§8.4a Pre-Quiz", "due": "2099-01-01T00:00:00.000Z", "end": None},
        {"id": "3", "name": "§9.4 Pre-Quiz", "start": "2099-01-01T00:00:00.000Z", "due": None, "end": None},
    ])
    monkeypatch.setattr(brightspace, "BRIGHTSPACE_AVAILABLE", True)
    monkeypatch.setattr(brightspace, "BrightspaceClient", lambda: client, raising=False)
    monkeypatch.setattr(brightspace, "BRIGHTSPACE_CONFIG", {"courses": ["611388"]})
    monkeypatch.setattr(brightspace, "get_sso_credentials", lambda: (None, None))
    return client


def test_quizzes_impl_skips_reports_already_fetched_after_due(tmp_path, fake_client):
    # Quiz 3 hasn't opened, so it is never fetched.
    brightspace._quizzes_impl(output_dir=tmp_path)
    assert fake_client.saved == [["1", "2"]]
    manifest = json.loads((tmp_path / "611388" / "manifest.json").read_text())
    assert manifest["1"]["reports"]["10"]["file"] == "quiz 1 - User Attempts.xlsx"
    assert (tmp_path / "611388" / "quizzes.json").exists()

    brightspace._quizzes_impl(output_dir=tmp_path)
    assert fake_client.saved == [["1", "2"]]

    brightspace._quizzes_impl(output_dir=tmp_path, quiz=["8.2"], force=True)
    assert fake_client.saved == [["1", "2"], ["1"]]


def test_quizzes_impl_due_only_fetches_past_due(tmp_path, fake_client):
    brightspace._quizzes_impl(output_dir=tmp_path, due=True)
    assert fake_client.saved == [["1"]]
