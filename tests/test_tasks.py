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
