import datetime as dt
import sys

import pytest

from coursedata import schedule
from coursedata.schedule import ScheduleConfigError, parse_entries

# 2026-11-03 is a Tuesday.
TUE = dt.date(2026, 11, 3)


def _entry(**overrides):
    item = {"name": "quiz", "days": ["Tue", "Thu"], "time": "08:05", "run": ["true"]}
    item.update(overrides)
    return parse_entries([item])[0]


def _at(date, hhmm):
    hour, minute = map(int, hhmm.split(":"))
    return dt.datetime.combine(date, dt.time(hour, minute))


def test_due_after_time_on_listed_day():
    entry = _entry()
    assert not entry.is_due(_at(TUE, "08:04"), None)
    assert entry.is_due(_at(TUE, "08:05"), None)
    assert entry.is_due(_at(TUE, "13:30"), None)  # catch-up later the same day


def test_not_due_on_other_days():
    entry = _entry()
    assert not entry.is_due(_at(TUE + dt.timedelta(days=1), "09:00"), None)  # Wednesday


def test_not_due_twice_same_day():
    entry = _entry()
    assert not entry.is_due(_at(TUE, "09:00"), "2026-11-03")
    assert entry.is_due(_at(TUE + dt.timedelta(days=2), "09:00"), "2026-11-03")  # Thursday


def test_date_window_inclusive():
    entry = _entry(days="daily", **{"from": dt.date(2026, 11, 3), "until": "2026-11-05"})
    assert not entry.is_due(_at(dt.date(2026, 11, 2), "09:00"), None)
    assert entry.is_due(_at(dt.date(2026, 11, 3), "09:00"), None)
    assert entry.is_due(_at(dt.date(2026, 11, 5), "09:00"), None)
    assert not entry.is_due(_at(dt.date(2026, 11, 6), "09:00"), None)


def test_full_day_names_accepted():
    assert _entry(days=["Tuesday", "thursday"]).days == frozenset({1, 3})


@pytest.mark.parametrize(
    "overrides",
    [
        {"days": ["Tue", "Blursday"]},
        {"days": []},
        {"time": "8am"},
        {"run": []},
        {"until": "next week"},
        {"extra": 1},
    ],
)
def test_invalid_entries(overrides):
    with pytest.raises(ScheduleConfigError):
        _entry(**overrides)


def test_duplicate_names_rejected():
    item = {"name": "a", "days": "daily", "time": "05:00", "run": ["true"]}
    with pytest.raises(ScheduleConfigError):
        parse_entries([item, dict(item)])


def test_python_replaced_by_current_interpreter():
    assert schedule._argv("python -m coursedata.tasks daily")[0] == sys.executable
    assert schedule._argv("make daily") == ["make", "daily"]


def test_run_records_failure_and_continues(tmp_path, monkeypatch):
    marker = tmp_path / "ran"
    monkeypatch.setattr(
        schedule,
        "_config",
        {
            "schedule": [
                {
                    "name": "job",
                    "days": "daily",
                    "time": "00:00",
                    "run": ["false", f"touch {marker}"],
                }
            ]
        },
    )
    state_path = tmp_path / "state.json"
    monkeypatch.setattr(schedule, "STATE_PATH", state_path)
    notes = []
    monkeypatch.setattr(schedule, "_notify", notes.append)
    monkeypatch.setattr(schedule, "PROJ_ROOT", tmp_path)

    schedule.run(dry_run=False)

    assert marker.exists()  # second command still ran after the first failed
    assert schedule.load_state()["job"] == dt.date.today().isoformat()
    assert len(notes) == 1

    marker.unlink()
    schedule.run(dry_run=False)  # already ran today
    assert not marker.exists()
