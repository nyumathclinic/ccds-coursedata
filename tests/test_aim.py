from datetime import date
from pathlib import Path

import pytest

from coursedata.dataset import aim
from edubag.aim.client import DownloadBatch
from edubag.aim.portal import ExamRow

CH7 = {"crn": ["10488", "10493"], "type": ["Exam"], "date": date(2026, 11, 5), "until": date(2026, 11, 13)}


def _row(email="pp4321@nyu.edu", uploaded=date(2026, 11, 5), crn="10488", exam_date=date(2026, 11, 5)):
    return ExamRow(
        crn=crn,
        subject="MATH-UA",
        course="122",
        section="016",
        name="Pat Person",
        email=email,
        type="Exam",
        exam_date=exam_date,
        time_range="02:55 PM - 04:40 PM (105)",
        late="",
        proctor="Not Assigned",
        uploaded_on=uploaded,
        checkbox_id="cb",
    )


@pytest.fixture
def config(monkeypatch, tmp_path):
    monkeypatch.setattr(aim, "AIM_CONFIG", {"exams": {"ch7": dict(CH7)}})
    monkeypatch.setattr(aim, "ALBERT_CONFIG", {})
    monkeypatch.setattr(aim, "RAW_DATA_DIR", tmp_path)
    return monkeypatch


def test_spec_from_config(config):
    spec = aim.exam_spec("ch7", CH7)
    assert spec.artifacts == "ch7"
    assert spec.filter.crn == {"10488", "10493"}
    assert spec.filter.type == {"exam"}
    assert (spec.filter.date, spec.filter.until) == (date(2026, 11, 5), date(2026, 11, 13))
    assert spec.save_dir.name == "ch7" and spec.save_dir.parent.name == "artifacts"


def test_crn_defaults_to_albert_courses(config):
    cfg = {"date": date(2026, 11, 5)}
    with pytest.raises(aim.ConfigError, match="albert"):
        aim.exam_spec("ch7", cfg)
    config.setattr(aim, "ALBERT_CONFIG", {"courses": ["10488", 10493]})
    assert aim.exam_spec("ch7", cfg).filter.crn == {"10488", "10493"}


def test_dates(config):
    one_day = aim.exam_spec("q", {"crn": ["1"], "date": date(2026, 11, 5)}).filter
    assert one_day.end == date(2026, 11, 5)
    grace = aim.exam_spec("q", {"crn": ["1"], "date": date(2026, 11, 5), "grace_days": 8}).filter
    assert grace.until == date(2026, 11, 13)
    with pytest.raises(aim.ConfigError, match="not both"):
        aim.exam_spec("q", {"crn": ["1"], "date": date(2026, 11, 5), "until": date(2026, 11, 6), "grace_days": 1})
    with pytest.raises(aim.ConfigError, match="date is required"):
        aim.exam_spec("q", {"crn": ["1"], "until": date(2026, 11, 6)})
    with pytest.raises(aim.ConfigError, match="unknown keys"):
        aim.exam_spec("q", {"crn": ["1"], "date": date(2026, 11, 5), "from": date(2026, 11, 5)})


def test_active_window(config):
    spec = aim.exam_spec("ch7", CH7)
    assert not spec.active(date(2026, 11, 4))
    assert spec.active(date(2026, 11, 5))
    assert spec.active(date(2026, 11, 20))  # until + 7
    assert not spec.active(date(2026, 11, 21))
    assert aim.exam_spec("ch7", {**CH7, "fetch_days": 0}).active(date(2026, 11, 13))
    assert not aim.exam_spec("ch7", {**CH7, "fetch_days": 0}).active(date(2026, 11, 14))


def test_select_specs(config):
    assert [s.name for s in aim.select_specs([], {}, date(2026, 11, 7))] == ["ch7"]
    assert aim.select_specs([], {}, date(2026, 12, 1)) == []
    assert [s.name for s in aim.select_specs(["ch7"], {}, date(2026, 12, 1))] == ["ch7"]  # named: always
    with pytest.raises(aim.ConfigError, match="configured: \\['ch7'\\]"):
        aim.select_specs(["ch8"], {}, date(2026, 11, 7))
    # Ad hoc: overrides for an unconfigured folder, or over a configured exam.
    [adhoc] = aim.select_specs(["ch8"], {"crn": ["10488"], "date": "2026-12-01"}, date(2026, 11, 7))
    assert adhoc.artifacts == "ch8" and adhoc.filter.end == date(2026, 12, 1)
    [over] = aim.select_specs(["ch7"], {"date": "2026-11-06", "until": None, "grace_days": None}, date(2026, 11, 7))
    assert (over.filter.date, over.filter.until) == (date(2026, 11, 6), None)
    with pytest.raises(aim.ConfigError, match="exactly one"):
        aim.select_specs([], {"crn": ["1"]}, date(2026, 11, 7))


def test_fetch_reason():
    row = _row()
    assert aim.fetch_reason(row, None, False) == "new"
    assert aim.fetch_reason(row, {"uploaded_on": "2026-11-05"}, False) is None
    assert aim.fetch_reason(row, {"uploaded_on": "2026-11-05"}, True) == "forced"
    assert aim.fetch_reason(_row(uploaded=date(2026, 11, 6)), {"uploaded_on": "2026-11-05"}, False) == "re-uploaded"


class FakeClient:
    """Stands in for AimClient: 'downloads' each wanted row as one PDF."""

    def __init__(self, rows):
        self.rows = rows
        self.calls = 0

    def list_exams(self, flt, **kw):
        return [r for r in self.rows if flt.matches(r)]

    def save_exams(self, save_dir, flt, want=None, **kw):
        self.calls += 1
        picked = [r for r in self.rows if flt.matches(r) and (want is None or want(r))]
        if not picked:
            return []
        save_dir.mkdir(parents=True, exist_ok=True)
        zips = save_dir / "_zips"
        zips.mkdir(exist_ok=True)
        z = zips / f"batch{self.calls}.zip"
        z.write_bytes(b"zip")
        files = []
        for r in picked:
            f = save_dir / f"({r.email[:6]}) E{self.calls}.pdf"
            f.write_bytes(b"%PDF")
            files.append(f)
        return [DownloadBatch(rows=picked, zip_path=z, files=files)]


def test_fetch_exam_downloads_only_new_and_reuploaded(config, tmp_path):
    spec = aim.exam_spec("ch7", CH7)
    a, b = _row("aa1111@nyu.edu"), _row("bb2222@nyu.edu")
    other_course = _row("cc3333@nyu.edu", crn="10433")
    client = FakeClient([a, b, other_course])
    run = dict(list_only=False, force=False, headless=True, credentials=(None, None))

    assert aim.fetch_exam(client, spec, **run) == 2
    manifest = aim.load_manifest(spec.save_dir)
    assert set(manifest["exams"]) == {a.key, b.key}
    entry = manifest["exams"][a.key]
    assert entry["uploaded_on"] == "2026-11-05"
    assert entry["zip"] == str(Path("_zips") / "batch1.zip")
    assert len(entry["batch_files"]) == 2

    assert aim.fetch_exam(client, spec, **run) == 0  # nothing new

    client.rows[1] = _row("bb2222@nyu.edu", uploaded=date(2026, 11, 7))  # rescanned
    assert aim.fetch_exam(client, spec, **run) == 1
    assert aim.load_manifest(spec.save_dir)["exams"][b.key]["uploaded_on"] == "2026-11-07"

    assert aim.fetch_exam(client, spec, **{**run, "force": True}) == 2


def test_list_only_downloads_nothing(config, capsys):
    spec = aim.exam_spec("ch7", CH7)
    client = FakeClient([_row()])
    assert aim.fetch_exam(client, spec, list_only=True, force=False, headless=True, credentials=(None, None)) == 1
    assert client.calls == 0
    assert not (spec.save_dir / aim.MANIFEST).exists()
    assert "new" in capsys.readouterr().out
