from pathlib import Path

import numpy as np
import pytest

from coursedata.dataset.artifacts import (
    Exam,
    ScannedPage,
    assign_pages,
    base_version,
    code39_check_char,
    forget_batch,
    group_exams,
    hold_reason,
    infer_missing_serials,
    is_exam_serial,
    personalize,
    read_ledger,
    serial_sort_key,
    split_code39,
    version_label,
    write_ledger,
)


@pytest.mark.parametrize(
    "text, serial",
    [("12681049310132187", "1268104931013218"), ("1268104931013102%", "1268104931013102")],
)
def test_split_code39_valid(text, serial):
    assert split_code39(text) == (serial, True)


def test_split_code39_bad_check_char():
    assert split_code39("12681049310132188") == ("1268104931013218", False)


def test_code39_check_char():
    assert code39_check_char("1268104931013101") == "+"


def test_version_label():
    assert version_label("ch6-16A.qns") == "ch6-16A"


def test_serial_sort_key_numeric():
    assert sorted(["100", "99", "1000"], key=serial_sort_key) == ["99", "100", "1000"]


def test_infer_missing_serials_between_same_exam():
    serials = ["a", None, "a", "b"]
    covers = [True, False, False, True]
    assert infer_missing_serials(serials, covers) == ["a", "a", "a", "b"]


def test_infer_missing_serials_unread_cover_starts_next_exam():
    serials = ["a", "a", None, "b"]
    covers = [True, False, True, False]
    assert infer_missing_serials(serials, covers) == ["a", "a", "b", "b"]


def test_infer_missing_serials_last_page_before_next_cover():
    serials = ["a", "a", None, "b"]
    covers = [True, False, False, True]
    assert infer_missing_serials(serials, covers) == ["a", "a", "a", "b"]


def test_infer_missing_serials_ambiguous_stays_none():
    serials = ["a", None, "b"]
    covers = [False, False, False]
    assert infer_missing_serials(serials, covers) == ["a", None, "b"]


def _cost(true_slots: list[int | None], n_tpl: int) -> np.ndarray:
    """Cost matrix where scanned page i matches template page true_slots[i] (None: nothing)."""
    cost = np.full((len(true_slots), n_tpl), 1.0)
    for i, j in enumerate(true_slots):
        if j is not None:
            cost[i, j] = 0.1
    return cost


def test_assign_pages_restores_order():
    slots, extras = assign_pages(_cost([0, 2, 1, 3], 4), threshold=3.0)
    assert slots == [0, 2, 1, 3]
    assert extras == []


def test_assign_pages_missing_page():
    slots, extras = assign_pages(_cost([0, 1, 3], 4), threshold=3.0)
    assert slots == [0, 1, None, 2]
    assert extras == []


def test_assign_pages_extra_page():
    slots, extras = assign_pages(_cost([0, None, 1, 2], 3), threshold=3.0)
    assert slots == [0, 2, 3]
    assert extras == [1]


def test_assign_pages_rejects_over_threshold():
    cost = _cost([0, 1], 2)
    cost[1, 1] = 5.0  # e.g. a feature penalty: not actually that page
    cost[1, 0] = 5.0
    slots, extras = assign_pages(cost, threshold=3.0)
    assert slots == [0, None]
    assert extras == [1]


def test_assign_pages_identical_pages_keep_scan_order():
    cost = np.full((3, 3), 0.5)
    slots, _ = assign_pages(cost, threshold=3.0)
    assert slots == [0, 1, 2]


def test_is_exam_serial():
    assert is_exam_serial("12681049310132187") == "1268104931013218"
    # A testing-center form's Code39: no valid check character.
    assert is_exam_serial("E82941240129493") is None


def _page(source: str, index: int, serial=None, aztec=None) -> ScannedPage:
    return ScannedPage(
        source=Path(source),
        index=index,
        serial=serial,
        aztec=aztec,
        bubblesheet=False,
        image=b"",
        blank=False,
    )


def test_group_exams_by_serial_across_files_and_unbarcoded_file():
    stack1 = [_page("s1.pdf", 0, "1", "v.qns"), _page("s1.pdf", 1, "1"), _page("s1.pdf", 2)]
    stack2 = [_page("s2.pdf", 0, "1")]
    moses = [_page("m.pdf", 0), _page("m.pdf", 1, aztec="w.qns")]
    exams, unassigned = group_exams(
        {Path("s1.pdf"): stack1, Path("s2.pdf"): stack2, Path("m.pdf"): moses},
        {Path("s1.pdf"): "a" * 64, Path("s2.pdf"): "b" * 64, Path("m.pdf"): "c" * 64},
    )
    by_key = {e.key: e for e in exams}
    assert set(by_key) == {"serial:1", "file:" + "c" * 16}
    assert len(by_key["serial:1"].pages) == 3
    assert by_key["file:" + "c" * 16].label == "m.pdf"
    assert unassigned == [stack1[2]]


def test_hold_reason():
    barcoded = Exam(key="serial:1", label="1", pages=[_page("s.pdf", 0, "1", "v.qns")])
    assert hold_reason(barcoded, {}, {}) is None
    assert "batch B1" in hold_reason(barcoded, {"serial:1": "B1"}, {})

    moses = Exam(key="file:abc", label="m.pdf", pages=[_page("m.pdf", 0, aztec="w.qns")])
    assert hold_reason(moses, {}, {}) is None
    assert "rescan" in hold_reason(moses, {}, {"w": "B1"})

    two_covers = Exam(
        key="file:def",
        label="x.pdf",
        pages=[_page("x.pdf", 0, aztec="w.qns"), _page("x.pdf", 1, aztec="w.qns")],
    )
    assert "split it by hand" in hold_reason(two_covers, {}, {})


def test_forget_batch(tmp_path):
    write_ledger(
        tmp_path / "processed_files.csv",
        ["batch", "file", "sha256", "pages"],
        [{"batch": "B1", "file": "a", "sha256": "x", "pages": 1},
         {"batch": "B2", "file": "b", "sha256": "y", "pages": 1}],
    )  # fmt: skip
    (tmp_path / "B2").mkdir()
    forget_batch(tmp_path, "B2")
    assert [r["batch"] for r in read_ledger(tmp_path / "processed_files.csv")] == ["B1"]
    assert not (tmp_path / "B2").exists()


ROWS = [
    {"exam": "*", "netid": "mab10389", "name": "M B", "version": "21A 1100"},
    {"exam": "ch* final", "netid": "alt9145", "name": "A T R", "version": "21A 1100"},
    {"exam": "final", "netid": "zz1", "name": "Z", "version": "1100"},
]


def test_base_version_first_existing_candidate():
    assert base_version("ch6-alt9145.qns", ROWS, {"ch6-21A", "ch6-1100"}) == (
        "alt9145",
        "ch6-21A",
    )


def test_base_version_falls_through_to_next_candidate():
    assert base_version("ch6-mab10389.qns", ROWS, {"ch6-1100"}) == ("mab10389", "ch6-1100")


def test_base_version_no_existing_candidate():
    assert base_version("ch6-mab10389.qns", ROWS, set()) == ("mab10389", None)


def test_base_version_exam_glob_must_match():
    assert base_version("ch6-zz1.qns", ROWS, {"ch6-1100"}) is None


def test_base_version_unknown_netid_or_regular_version():
    assert base_version("ch6-xy123.qns", ROWS, {"ch6-21A"}) is None
    assert base_version("ch6-21A.qns", ROWS, {"ch6-21A"}) is None


def test_personalize_sets_key_and_version():
    exam = Exam(key="file:abc", label="m.pdf", pages=[_page("m.pdf", 0, aztec="ch6-alt9145.qns")])
    personalize(exam, ROWS, {"ch6-21A"})
    assert exam.key == "netid:ch6:alt9145"
    assert (exam.netid, exam.form, exam.version) == ("alt9145", "ch6-alt9145", "ch6-21A")
    # A rescan of the same student's exam is held by its key.
    assert "already emitted" in hold_reason(exam, {"netid:ch6:alt9145": "B1"}, {})
