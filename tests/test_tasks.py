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