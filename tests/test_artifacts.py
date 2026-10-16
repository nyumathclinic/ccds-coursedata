import numpy as np
import pytest

from coursedata.dataset.artifacts import (
    assign_pages,
    code39_check_char,
    infer_missing_serials,
    serial_sort_key,
    split_code39,
    version_label,
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
