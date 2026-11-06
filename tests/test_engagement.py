from datetime import datetime, timezone
import json

import pandas as pd

from coursedata.dataset import engagement

NOW = datetime(2026, 11, 6, 15, 0, tzinfo=timezone.utc)  # Fri 10:00 New York


def _col(name: str, category: str) -> str:
    return (
        f"{name} Points Grade <Numeric MaxPoints:4 Weight:0 "
        f"Category:Participation Data - {category} CategoryWeight:0>"
    )


PAST_QUIZ = _col("§8.4a Pre-Quiz", "Pre-Quizzes")
FUTURE_QUIZ = _col("§8.4b Pre-Quiz", "Pre-Quizzes")
FUTURE_SURVEY = _col("§8.4b Pre-Survey", "Pre-Surveys")
PAST_POLL = _col("2026-11-05 Polls - Participation", "PollEverywhere")
TODAY_POLL = _col("2026-11-06 Polls - Participation", "PollEverywhere")
FUTURE_POLL = _col("2026-11-10 Polls - Participation", "PollEverywhere")
UNDATED = _col("Bonus Poll", "PollEverywhere")


def _grades() -> pd.DataFrame:
    cols = ["Username", PAST_QUIZ, FUTURE_QUIZ, FUTURE_SURVEY, PAST_POLL, TODAY_POLL, FUTURE_POLL, UNDATED]
    return pd.DataFrame([["abc123"] + [1] * (len(cols) - 1)], columns=cols)


def _quizzes(tmp_path):
    path = tmp_path / "quizzes.json"
    path.write_text(json.dumps([
        {"id": "1", "name": "§8.4a Pre-Quiz", "due": "2026-11-05T13:00:00.000Z", "end": "2026-12-15T04:59:59.000Z"},
        {"id": "2", "name": "§8.4b Pre-Quiz", "due": "2026-11-10T13:00:00.000Z", "end": "2026-12-15T04:59:59.000Z"},
        {"id": "3", "name": "§8.4b Pre-Survey", "due": None, "end": "2026-12-15T04:59:59.000Z"},
    ]))
    return str(path)


def test_item_name_strips_header_suffix():
    assert engagement._item_name(PAST_QUIZ) == "§8.4a Pre-Quiz"
    assert engagement._item_name(FUTURE_POLL) == "2026-11-10 Polls - Participation"


def test_future_items_from_quizzes_json_and_title_dates(tmp_path):
    cfg = {"quizzes_json": _quizzes(tmp_path), "title_dates": True}
    future = engagement._future_item_columns(_grades(), cfg, NOW)
    # Due-in-future quiz, and a survey with no due date falls back to its end date.
    assert FUTURE_QUIZ in future and FUTURE_SURVEY in future
    assert FUTURE_POLL in future
    for kept in (PAST_QUIZ, PAST_POLL, TODAY_POLL, UNDATED, "Username"):
        assert kept not in future


def test_no_date_settings_keeps_everything(tmp_path):
    assert engagement._future_item_columns(_grades(), {}, NOW) == []
    # title_dates off: only quiz deadlines apply.
    future = engagement._future_item_columns(_grades(), {"quizzes_json": _quizzes(tmp_path)}, NOW)
    assert set(future) == {FUTURE_QUIZ, FUTURE_SURVEY}


def test_denominators_subtract_unprefixed_base_gradebook_exemptions():
    # The base gradebook's columns are merged without a "gradebook_" prefix.
    cat = "Participation Data - PollEverywhere"
    merged = pd.DataFrame({"Username": ["a", "b"], f"{cat}_exemptions": [0, 2]})
    engagement._add_denominator_columns(
        merged,
        {"gradebook": {"type": "brightspace_gradebook"}},
        {"gradebook": {cat: {"active_items": 18}}},
    )
    assert merged["gradebook_Participation_Data___PollEverywhere_denominator"].tolist() == [18, 16]
