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
    rescan_warning,
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
    assert hold_reason(barcoded, {}) is None
    assert "batch B1" in hold_reason(barcoded, {"serial:1": "B1"})

    moses = Exam(key="file:abc", label="m.pdf", pages=[_page("m.pdf", 0, aztec="w.qns")])
    # Another unbarcoded exam of the same version is emitted, with a warning.
    assert hold_reason(moses, {}) is None
    assert rescan_warning(moses, {}) is None
    assert "rescan" in rescan_warning(moses, {"w": "B1"})
    assert rescan_warning(barcoded, {"v": "B1"}) is None

    two_covers = Exam(
        key="file:def",
        label="x.pdf",
        pages=[_page("x.pdf", 0, aztec="w.qns"), _page("x.pdf", 1, aztec="w.qns")],
    )
    assert "split it by hand" in hold_reason(two_covers, {})


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
    assert "already emitted" in hold_reason(exam, {"netid:ch6:alt9145": "B1"})


# ---------------------------------------------------------------------------
# Gradescope submission exports
# ---------------------------------------------------------------------------

import itertools  # noqa: E402
import zipfile  # noqa: E402

import cv2  # noqa: E402
import pymupdf  # noqa: E402
import zxingcpp  # noqa: E402

from coursedata.dataset import artifacts  # noqa: E402
from coursedata.dataset.artifacts import (  # noqa: E402
    ExportSubmission,
    Template,
    duplicate_reason,
    read_export,
    register_bubblesheet,
    resolve_export_versions,
    stamp_bubblesheet,
)

METADATA = """---
101.pdf:
  :submitters:
  - :name: Ada Lovelace
    :sid: N12345678
    :email: al123@nyu.edu
  :created_at: 2026-10-15 17:00:00.000000000 -04:00
  :score: 0.0
  :original_filename: 2026101517.pdf
22.pdf:
  :submitters:
  - :name: Alan Turing
    :sid: N87654321
    :email: amt4@nyu.edu
  :original_filename: t1-A.pdf
"""


def _png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def _aztec(text: str) -> np.ndarray:
    try:
        return np.array(
            zxingcpp.write_barcode_to_image(
                zxingcpp.create_barcode(text, zxingcpp.BarcodeFormat.Aztec), scale=8
            )
        )
    except AttributeError:  # older zxing-cpp
        return np.array(zxingcpp.write_barcode(zxingcpp.BarcodeFormat.Aztec, text, 200, 200))


@pytest.fixture(scope="module")
def template(tmp_path_factory) -> Template:
    """A two-page template: a cover (Aztec + Name/NetID boxes) and a bubblesheet (ArUco + Name/ID)."""
    path = tmp_path_factory.mktemp("templates") / "t1-A.qns.pdf"
    doc = pymupdf.open()
    cover = doc.new_page(width=612, height=792)
    cover.insert_image(pymupdf.Rect(450, 40, 560, 150), stream=_png(_aztec("t1-A.qns")))
    for y, label in ((200, "Name:"), (260, "NetID:")):
        cover.draw_rect(pymupdf.Rect(72, y, 540, y + 40), color=(0, 0, 0), width=1)
        cover.insert_text((78, y + 24), label, fontsize=12)
    bubble = doc.new_page(width=612, height=792)
    dictionary = cv2.aruco.getPredefinedDictionary(artifacts.ARUCO_DICT)
    for i, (x, y) in enumerate(((36, 36), (516, 36), (516, 696), (36, 696))):
        marker = cv2.copyMakeBorder(
            cv2.aruco.generateImageMarker(dictionary, i, 120), 12, 12, 12, 12,
            cv2.BORDER_CONSTANT, value=255,
        )  # fmt: skip
        bubble.insert_image(pymupdf.Rect(x, y, x + 60, y + 60), stream=_png(marker))
    for y, label in ((150, "Name"), (210, "ID")):
        bubble.draw_rect(pymupdf.Rect(72, y, 540, y + 40), color=(0, 0, 0), width=1)
        bubble.insert_text((78, y + 24), label, fontsize=12)
    doc.save(path)
    return Template.load(path, "t1-A.qns")


def _bubblesheet_page(template: Template, quarter_turns: int = 0, rotation: int = 0):
    """A 'scan' of the blank bubblesheet: its image turned ``quarter_turns`` times, shown with
    /Rotate ``rotation``."""
    img = np.ascontiguousarray(np.rot90(template.bubblesheet_img, quarter_turns))
    doc = pymupdf.open()
    page = doc.new_page(width=img.shape[1] * 72 / artifacts.DPI, height=img.shape[0] * 72 / artifacts.DPI)
    page.insert_image(page.rect, stream=_png(img))
    page.set_rotation(rotation)
    return doc, page


def _box_ink(page, template: Template, key: str) -> np.ndarray:
    registered, _ = register_bubblesheet(artifacts.to_gray(page), template)
    return artifacts.new_ink(registered, template.bubblesheet_img, template.bubblesheet_boxes[key])


def test_stamp_bubblesheet_types_only_in_blank_boxes(template):
    doc, page = _bubblesheet_page(template)
    id_area = template.bubblesheet_boxes["id"].writing_area
    page.insert_text((id_area.x0 + 10, id_area.y0 + 24), "scribble", fontsize=14)  # student's ink
    filled, warnings = stamp_bubblesheet(page, template, {"name": "Ada Lovelace", "id": "al123"})
    assert filled == {"name": True, "id": False}
    assert warnings == []
    name_area = template.bubblesheet_boxes["name"].writing_area
    words = [w[4] for w in page.get_text("words") if pymupdf.Rect(w[:4]).intersects(name_area)]
    assert words == ["Ada", "Lovelace"]


def test_stamp_bubblesheet_without_markers(template):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    filled, warnings = stamp_bubblesheet(page, template, {"name": "Ada"})
    assert filled == {}
    assert "markers not found" in warnings[0]


def test_stamp_bubblesheet_upright_in_any_orientation(template):
    _, ref_page = _bubblesheet_page(template)
    stamp_bubblesheet(ref_page, template, {"name": "Ada Lovelace"})
    ref = _box_ink(ref_page, template, "name")
    ry, rx = np.nonzero(ref)
    ref_crop = ref[ry.min() : ry.max() + 1, rx.min() : rx.max() + 1].astype(np.float32)
    for turns, rotation in itertools.product(range(4), (0, 90, 180, 270)):
        _, page = _bubblesheet_page(template, turns, rotation)
        filled, _ = stamp_bubblesheet(page, template, {"name": "Ada Lovelace"})
        assert filled["name"], (turns, rotation)
        mask = _box_ink(page, template, "name")
        ys, xs = np.nonzero(mask)
        crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1].astype(np.float32)
        # Same extent as upright text (not wrapped or sideways)...
        assert abs(crop.shape[1] - ref_crop.shape[1]) < 30 and abs(crop.shape[0] - ref_crop.shape[0]) < 15
        # ...and matches it better than its upside-down image does.
        h, w = min(crop.shape[0], ref_crop.shape[0]), min(crop.shape[1], ref_crop.shape[1])
        same = (crop[:h, :w] * ref_crop[:h, :w]).sum()
        flipped = (crop[::-1, ::-1][:h, :w] * ref_crop[:h, :w]).sum()
        assert same > flipped, (turns, rotation)


def _write_export(folder: Path) -> None:
    folder.mkdir()
    (folder / "submission_metadata.yml").write_text(METADATA)
    for name in ("101.pdf", "22.pdf"):
        doc = pymupdf.open()
        doc.new_page()
        doc.save(folder / name)


def test_read_export_folder_and_zip(tmp_path):
    folder = tmp_path / "assignment_9_export"
    _write_export(folder)
    subs = read_export(folder)
    assert [s.submission_id for s in subs] == ["22", "101"]  # numeric order
    ada = subs[1]
    assert (ada.name, ada.email, ada.sid, ada.netid) == ("Ada Lovelace", "al123@nyu.edu", "N12345678", "al123")
    assert ada.key == "submission:101"
    assert ada.file == "assignment_9_export/101.pdf"
    assert ada.pdf.startswith(b"%PDF")
    assert ada.stamp_values() == {"name": "Ada Lovelace", "id": "al123"}

    zipped = tmp_path / "zips" / "assignment_9_export.zip"
    zipped.parent.mkdir()
    with zipfile.ZipFile(zipped, "w") as zf:
        for f in folder.iterdir():
            zf.write(f, f"assignment_9_export/{f.name}")
    assert [s.submission_id for s in read_export(zipped)] == ["22", "101"]
    assert artifacts.export_sources(zipped.parent) == [zipped]
    assert artifacts.export_sources(tmp_path) == [folder]


def test_read_export_rejects_group_submissions(tmp_path):
    folder = tmp_path / "assignment_9_export"
    _write_export(folder)
    text = METADATA.replace(
        "    :email: al123@nyu.edu\n",
        "    :email: al123@nyu.edu\n  - :name: Bo\n    :sid: N1\n    :email: bo1@nyu.edu\n",
    )
    (folder / "submission_metadata.yml").write_text(text)
    with pytest.raises(ValueError, match="2 submitters"):
        read_export(folder)


def _sub(export: str, sid: str, aztecs: list) -> ExportSubmission:
    return ExportSubmission(
        export=export, submission_id=sid, pdf=b"", name="", email="x@nyu.edu", sid="",
        original_filename="", aztecs=aztecs,
    )  # fmt: skip


def test_resolve_export_versions():
    rows = [{"exam": "ch6", "netid": "ak1", "name": "", "version": "16B 21A"}]
    existing = {"ch6-21A", "ch6-21B", "ch6-ak1"}
    subs = [
        _sub("e1", "1", [None, "ch6-21A.qns"]),
        _sub("e1", "2", [None, None]),  # no cover: takes the export's version
        _sub("e1", "3", ["ch6-ak1.qns"]),  # personalized: its base version
        _sub("e2", "4", [None]),  # no cover anywhere in its export
    ]
    resolve_export_versions(subs, rows, existing)
    assert [s.version for s in subs] == ["ch6-21A", "ch6-21A", "ch6-21A", None]
    assert "no readable cover" in subs[1].warnings[0]
    assert subs[3].skipped and not subs[0].skipped


def test_duplicate_reason():
    sub = _sub("e1", "1", [])
    sub.original_filename = "ch6-21A.pdf"
    assert "batch b1" in duplicate_reason(sub, {"ch6-21A": "b1"})
    sub.original_filename = "20261015171803575.pdf"
    assert duplicate_reason(sub, {"ch6-21A": "b1"}) is None


def test_output_stem():
    assert artifacts.output_stem("ch6-16A", "2026-10-17T1513") == "ch6-16A-2026-10-17T1513"
    assert (
        artifacts.output_stem("ch6-16A", "2026-10-17T1513-2", "bubblesheets")
        == "ch6-16A-bubblesheets-2026-10-17T1513-2"
    )


def test_duplicate_reason_stamped():
    sub = _sub("e1", "1", [])
    stems = {"ch6-21A": "b1", artifacts.output_stem("ch6-21A", "b1"): "b1"}
    sub.original_filename = "ch6-21A-b1.pdf"
    assert "batch b1" in duplicate_reason(sub, stems)
    sub.original_filename = "ch6-21A-bubblesheets-b1.pdf"
    assert duplicate_reason(sub, stems) is None
