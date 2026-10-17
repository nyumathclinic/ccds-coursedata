"""Scanned exam stacks -> per-version PDFs for uploading to Gradescope.

Commands are wired into:
    python -m coursedata.dataset process artifacts [EXAM_DIR ...] [--dry-run] [--forget BATCH]

Each exam directory (e.g. ``RAW_DATA_DIR/artifacts/ch6``) holds scanner PDFs plus
``templates/<version code>.pdf`` -- one blank exam per version. The cover of each exam carries an
Aztec code whose text is the version code (e.g. ``ch6-16A.qns``), and the Gradescope bubblesheet
carries ArUco markers 0-3 in its corners. Scanner PDFs come in two kinds:

* barcoded stacks of concatenated exams, where every page carries a Code39 serial number (the
  last character is a mod-43 check character) identifying its exam;
* unbarcoded PDFs (e.g. from the testing center), each holding exactly one exam.

Runs are incremental. Each run processes only raw PDFs not seen before (by content hash) and
writes the exams in them to a new batch folder, ``PROCESSED_DATA_DIR/artifacts/<exam>/<timestamp>/``:

* ``<version>.pdf``: the batch's exams of that version, sorted by serial number, with each exam's
  pages put in template order. A missing page is replaced by the blank template page; a scanned
  page that matches no template page is appended after the exam's template pages, unless it is
  totally blank, in which case it is dropped.
* ``<version>-bubblesheets.pdf``: only the bubblesheet pages, in the same order. A blank Name or
  ID box is filled with an image of what the student wrote in the matching box on the cover.
* ``held/<exam>.pdf``: exams that look like ones emitted in an earlier batch (same serial, or an
  unbarcoded exam of an already-emitted unbarcoded version). Not for upload; resolve by hand.
* ``report.csv``: one row per exam, recording the page mapping, fills, and any warnings.

Gradescope submission exports (``assignment_<id>_export/`` folders or ``.zip`` files in the exam
directory, from "Export Submissions" or ``dataset get gradescope submissions``) are exams already
uploaded and matched to students, e.g. to the free-response part. Only their bubblesheets are
emitted, into the same ``<version>-bubblesheets.pdf`` files, for the multiple-choice part: a blank
Name or ID box gets the student's name and NetID (or ``stamp_id = "sid"``) typed in, and
``bubblesheets.csv`` lists whose bubblesheet is on each page. A submission uploaded from an earlier
batch's ``<version>.pdf`` is skipped, since that batch already has its bubblesheet.

Here ``<version>`` is the Aztec text without its extension (``ch6-16A``). The ledgers
``processed_files.csv`` and ``processed_exams.csv`` next to the batch folders record what has been
processed; ``--forget BATCH`` deletes a batch and its ledger rows so it can be redone.
"""

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, field
from datetime import datetime
from fnmatch import fnmatch
import hashlib
from pathlib import Path
import re
import shutil
import zipfile

import cv2
from loguru import logger
import numpy as np
import pymupdf
from scipy.optimize import linear_sum_assignment
import yaml
import zxingcpp

from coursedata.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, _config

ARTIFACTS_CONFIG = _config.get("artifacts", {})
ARTIFACTS_RAW_DIR = RAW_DATA_DIR / "artifacts"
ARTIFACTS_PROCESSED_DIR = PROCESSED_DATA_DIR / "artifacts"

# Labels of the boxes students write in, found by exact word match on the template pages.
COVER_LABELS = ARTIFACTS_CONFIG.get("cover_labels", {"name": "Name:", "id": "NetID:"})
BUBBLESHEET_LABELS = ARTIFACTS_CONFIG.get("bubblesheet_labels", {"name": "Name", "id": "ID"})
# Optional regex an exam serial (Code39 without its check character) must fully match, to keep
# other Code39 barcodes (e.g. on testing-center forms) from being taken for exam serials.
SERIAL_PATTERN = ARTIFACTS_CONFIG.get("serial_pattern")

# Personalized exams (Aztec "<exam>-<netid>.qns") are grouped under the student's base version
# from this CSV (columns exam, netid, name, version; exam and version are space-separated
# lists: exam globs, and candidate version labels of which the first existing one is used).
PERSONALIZED_CSV = Path(
    ARTIFACTS_CONFIG.get("personalized_csv", RAW_DATA_DIR / "artifacts" / "personalized-exams.csv")
)

FILES_LEDGER = "processed_files.csv"
EXAMS_LEDGER = "processed_exams.csv"

DPI = 200  # render resolution for barcode reading and registration
RETRY_DPI = 300  # second try at the Code39 strip when the first render doesn't decode
LETTER = (8.5, 11.0)  # inches; pages are cropped/padded to this before matching
ALIGN_DPI = 100  # resolution at which scanned pages are aligned to templates
THUMB_DPI = 25  # resolution of matching thumbnails
INK_DPI = 50  # resolution of ink masks for matching
EDGE_MARGIN = 0.02  # fraction of each page edge ignored in matching (scanner border lines)
# A scanned page is aligned to its template by ORB features matched against the header/footer
# that most template pages share. Some scanners shrink the page (~4% seen); pages without the
# header (cover, bubblesheet, reference sheets, extra paper) fall back to the median alignment of
# the other pages from the same file.
MAX_ALIGN_SCALE_ERROR = 0.1
MAX_ALIGN_ROTATION = 3.0  # degrees
MIN_ALIGN_INLIERS = 8
BARCODE_STRIP = 0.08  # top fraction of the page masked out of matching (serial differs)
# Template pages whose thumbnails correlate above this (blank scratch pages, mostly-empty
# free-response pages) are told apart only by the regions where they differ.
SIMILAR_PAGE_CORR = 0.9
# How far (in DPI pixels) a scan may be shifted relative to the template.
MAX_SHIFT = 80
# Matching cost of a scanned page against a template page is the sum of
#   * 1 - the fraction of the template page's distinctive printed ink (ink not shared by most
#     pages, e.g. question text) present in the scan. Handwriting can't remove print, but can
#     cover the sparse ink of a nearly blank page; and
#   * 1 - correlation of blurred thumbnails, which handwriting degrades but which tells a
#     written-on page from a blank one.
# Each alone mismatches some real pages; the sum separated true from false matches by >= 0.2
# on the ch6 scans. FEATURE_PENALTY is added when a page's cover/bubblesheet codes disagree with
# the template page's; an assignment costing more than the threshold is rejected, so the codes
# always win.
FEATURE_PENALTY = 5.0
MATCH_COST_THRESHOLD = float(ARTIFACTS_CONFIG.get("match_cost_threshold", 3.0))
# Fraction of a box's pixels that must be new ink for the box to count as filled in.
INK_THRESHOLD = float(ARTIFACTS_CONFIG.get("ink_threshold", 0.002))
# An unmatched page with less than this fraction of marked pixels is totally blank and dropped.
BLANK_PAGE_THRESHOLD = float(ARTIFACTS_CONFIG.get("blank_page_threshold", 0.0005))

ARUCO_DICT = cv2.aruco.DICT_5X5_50
BUBBLESHEET_MARKER_IDS = {0, 1, 2, 3}

CODE39_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%"


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------


def code39_check_char(data: str) -> str:
    """Return the mod-43 check character for a Code39 payload."""
    return CODE39_ALPHABET[sum(CODE39_ALPHABET.index(c) for c in data) % 43]


def split_code39(text: str) -> tuple[str, bool]:
    """Split a decoded Code39 string into (serial, check character is valid)."""
    serial, check = text[:-1], text[-1:]
    try:
        return serial, code39_check_char(serial) == check
    except ValueError:  # character outside the Code39 alphabet
        return serial, False


def is_exam_serial(text: str) -> str | None:
    """The exam serial encoded by a decoded Code39 string, or None if it isn't one.

    An exam serial's check character must validate, and the serial must match SERIAL_PATTERN
    if one is configured.
    """
    serial, ok = split_code39(text)
    if not ok or (SERIAL_PATTERN and not re.fullmatch(SERIAL_PATTERN, serial)):
        return None
    return serial


def version_label(aztec_text: str) -> str:
    """Version label for output filenames: the Aztec text without its extension."""
    return Path(aztec_text).stem


def serial_sort_key(serial: str) -> tuple:
    """Sort numeric serials numerically, anything else after them as strings."""
    return (0, int(serial), "") if serial.isdigit() else (1, 0, serial)


def to_gray(page: pymupdf.Page, dpi: int = DPI) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def fit_letter(img: np.ndarray, dpi: int = DPI) -> np.ndarray:
    """Crop or pad (with white) a page image at ``dpi`` to letter size, anchored top left.

    Scanners differ in the page size they report, but not in physical scale, so cropping keeps
    scans and templates at the same scale for matching.
    """
    w, h = round(LETTER[0] * dpi), round(LETTER[1] * dpi)
    out = np.full((h, w), 255, np.uint8)
    crop = img[:h, :w]
    out[: crop.shape[0], : crop.shape[1]] = crop
    return out


def letter_size(dpi: int) -> tuple[int, int]:
    """(width, height) in pixels of a letter page at ``dpi``."""
    return round(LETTER[0] * dpi), round(LETTER[1] * dpi)


def align_image(img: np.ndarray) -> np.ndarray:
    """A DPI page image cropped/padded to letter size and reduced to ALIGN_DPI."""
    return cv2.resize(fit_letter(img), letter_size(ALIGN_DPI), interpolation=cv2.INTER_AREA)


def features(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(thumbnail, ink mask) for matching, from an ALIGN_DPI letter-size image.

    The thumbnail is blurred and standardized with the barcode strip masked out; the ink mask
    is the dark pixels at INK_DPI. Page edges are whitened in both.
    """
    small = cv2.resize(img, letter_size(INK_DPI), interpolation=cv2.INTER_AREA)
    h, w = small.shape
    m = int(EDGE_MARGIN * min(h, w))
    small[:m], small[h - m :], small[:, :m], small[:, w - m :] = 255, 255, 255, 255
    thumb = cv2.resize(small, letter_size(THUMB_DPI), interpolation=cv2.INTER_AREA)
    thumb = cv2.GaussianBlur(thumb.astype(np.float32), (5, 5), 0)
    thumb[: int(thumb.shape[0] * BARCODE_STRIP)] = 255
    thumb -= thumb.mean()
    norm = np.linalg.norm(thumb)
    return (thumb / norm if norm else thumb), small < 160


_ORB = cv2.ORB_create(nfeatures=3000)
_MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING)


def estimate_alignment(img: np.ndarray, ref_kp, ref_des) -> np.ndarray | None:
    """Similarity transform (2x3) taking ``img`` onto the reference features, or None."""
    kp, des = _ORB.detectAndCompute(img, None)
    if des is None or ref_des is None or len(kp) < 2:
        return None
    good = [
        pair[0]
        for pair in _MATCHER.knnMatch(des, ref_des, k=2)
        if len(pair) == 2 and pair[0].distance < 0.8 * pair[1].distance
    ]
    if len(good) < MIN_ALIGN_INLIERS:
        return None
    src = np.float32([kp[m.queryIdx].pt for m in good])
    dst = np.float32([ref_kp[m.trainIdx].pt for m in good])
    transform, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3
    )
    if transform is None or inliers.sum() < MIN_ALIGN_INLIERS:
        return None
    scale = np.hypot(transform[0, 0], transform[1, 0])
    angle = np.degrees(np.arctan2(transform[1, 0], transform[0, 0]))
    if abs(scale - 1) > MAX_ALIGN_SCALE_ERROR or abs(angle) > MAX_ALIGN_ROTATION:
        return None
    return transform


def warp_to(img: np.ndarray, transform: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Warp ``img`` by a 2x3 transform into an image of ``shape``, padding with white."""
    return cv2.warpAffine(
        img, transform, (shape[1], shape[0]), flags=cv2.INTER_LINEAR, borderValue=255
    )


def marked_fraction(img: np.ndarray) -> float:
    """Fraction of a page marked relative to its local background (works on colored paper)."""
    h, w = img.shape
    m = int(0.05 * min(h, w))  # ignore scanner edges
    inner = img[m : h - m, m : w - m]
    marks = cv2.adaptiveThreshold(
        inner, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 25
    )
    marks = cv2.morphologyEx(marks, cv2.MORPH_OPEN, np.ones((3, 3)))
    return float(np.count_nonzero(marks)) / marks.size


def ink_recall_cost(scan_ink: np.ndarray, tpl_ink: np.ndarray, distinct: np.ndarray) -> float:
    """1 - fraction of ``distinct`` template ink found in the scan, after aligning the two."""
    (dx, dy), _ = cv2.phaseCorrelate(tpl_ink.astype(np.float32), scan_ink.astype(np.float32))
    shift = np.float32([[1, 0, -dx], [0, 1, -dy]])
    tolerant = cv2.dilate(scan_ink.astype(np.uint8), np.ones((3, 3)))
    aligned = cv2.warpAffine(tolerant, shift, tpl_ink.shape[::-1])
    total = distinct.sum()
    return 1.0 - float((aligned[distinct] > 0).sum()) / total if total else 0.0


def detect_bubblesheet_markers(img: np.ndarray) -> dict[int, np.ndarray]:
    """ArUco marker id -> 4x2 corner array, for the bubblesheet's corner markers."""
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(ARUCO_DICT))
    corners, ids, _ = detector.detectMarkers(img)
    if ids is None:
        return {}
    found = {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners)}
    return {i: c for i, c in found.items() if i in BUBBLESHEET_MARKER_IDS}


def barcode_corners(result) -> np.ndarray:
    p = result.position
    return np.array(
        [[pt.x, pt.y] for pt in (p.top_left, p.top_right, p.bottom_right, p.bottom_left)],
        dtype=np.float32,
    )


def assign_pages(cost: np.ndarray, threshold: float) -> tuple[list[int | None], list[int]]:
    """Assign scanned pages (rows) to template pages (columns).

    Returns ``(slots, extras)``: ``slots[j]`` is the scanned page index for template page ``j``
    (``None`` if missing), and ``extras`` are scanned page indices matching no template page,
    in scan order.
    """
    n_scan, n_tpl = cost.shape
    # A tiny penalty on displacement makes interchangeable pages (blank scratch pages) keep
    # their scan order.
    rows = np.arange(n_scan)[:, None] / max(n_scan, 1)
    cols = np.arange(n_tpl)[None, :] / max(n_tpl, 1)
    tie_break = 1e-3 * np.abs(rows - cols)
    row_ind, col_ind = linear_sum_assignment(cost + tie_break)
    slots: list[int | None] = [None] * n_tpl
    assigned = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] <= threshold:
            slots[c] = int(r)
            assigned.add(int(r))
    extras = [r for r in range(n_scan) if r not in assigned]
    return slots, extras


def infer_missing_serials(serials: list[str | None], is_cover: list[bool]) -> list[str | None]:
    """Fill in unread serials from neighbors within one scanned stack.

    A page takes the serial of the nearest read pages before and after it when they agree. When
    they differ, the page belongs to the earlier exam unless a cover lies between it and the later
    read page (inclusive of the page itself), in which case it belongs to the later exam. Pages
    are never assigned across a cover boundary in the wrong direction; unresolvable pages stay
    ``None``.
    """
    out = list(serials)
    n = len(serials)
    for i in range(n):
        if serials[i] is not None:
            continue
        prev = next((j for j in range(i - 1, -1, -1) if serials[j] is not None), None)
        nxt = next((j for j in range(i + 1, n) if serials[j] is not None), None)
        prev_s = serials[prev] if prev is not None else None
        next_s = serials[nxt] if nxt is not None else None
        # Does a cover start a new exam between prev (exclusive) and this page (inclusive)?
        lo = prev + 1 if prev is not None else 0
        cover_before = any(is_cover[lo : i + 1])
        # ...or between this page (exclusive) and next (inclusive)?
        hi = nxt if nxt is not None else n - 1
        cover_after = any(is_cover[i + 1 : hi + 1])
        if cover_before and not cover_after:
            out[i] = next_s
        elif not cover_before and (cover_after or prev_s == next_s):
            out[i] = prev_s
    return out


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


@dataclass
class ScannedPage:
    source: Path
    index: int
    serial: str | None
    aztec: str | None
    bubblesheet: bool
    image: bytes  # PNG of align_image()
    blank: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def ref(self) -> str:
        return f"{self.source.name}:{self.index + 1}"

    def align_image(self) -> np.ndarray:
        return cv2.imdecode(np.frombuffer(self.image, np.uint8), cv2.IMREAD_GRAYSCALE)


def read_code39(page: pymupdf.Page, results) -> tuple[str | None, list[str]]:
    texts = [r.text for r in results if r.format == zxingcpp.BarcodeFormat.Code39]
    if not texts:
        strip = to_gray(page, RETRY_DPI)
        strip = strip[: int(strip.shape[0] * 0.15)]
        texts = [
            r.text for r in zxingcpp.read_barcodes(strip, formats=zxingcpp.BarcodeFormat.Code39)
        ]
    warnings = []
    serials = set()
    for text in texts:
        serial = is_exam_serial(text)
        if serial is None:
            logger.debug(f"page {page.number + 1}: ignoring Code39 {text!r} (not an exam serial)")
        else:
            serials.add(serial)
    if len(serials) > 1:
        warnings.append(f"several Code39 serials on one page: {sorted(serials)}")
        return None, warnings
    return (serials.pop() if serials else None), warnings


def scan_stack(pdf_path: Path) -> list[ScannedPage]:
    """Read the codes on every page of one scanned stack."""
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for index, page in enumerate(doc):
            img = to_gray(page)
            results = zxingcpp.read_barcodes(img)
            serial, warnings = read_code39(page, results)
            aztecs = [r.text for r in results if r.format == zxingcpp.BarcodeFormat.Aztec]
            pages.append(
                ScannedPage(
                    source=pdf_path,
                    index=index,
                    serial=serial,
                    aztec=aztecs[0] if aztecs else None,
                    bubblesheet=len(detect_bubblesheet_markers(img)) >= 3,
                    image=cv2.imencode(".png", align_image(img))[1].tobytes(),
                    blank=marked_fraction(img) < BLANK_PAGE_THRESHOLD,
                    warnings=warnings,
                )
            )
    inferred = infer_missing_serials(
        [p.serial for p in pages], [p.aztec is not None for p in pages]
    )
    for page, serial in zip(pages, inferred):
        if page.serial is None and serial is not None:
            page.serial = serial
            page.warnings.append("serial unreadable; inferred from neighboring pages")
    logger.info(f"{pdf_path.name}: {len(pages)} pages scanned")
    return pages


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def _segments(page: pymupdf.Page) -> tuple[list, list]:
    """Horizontal (y, x0, x1) and vertical (x, y0, y1) line segments drawn on a page."""
    horiz, vert = [], []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l":
                lines = [(item[1], item[2])]
            elif item[0] == "re":
                r = item[1]
                lines = [
                    (r.top_left, r.top_right),
                    (r.bottom_left, r.bottom_right),
                    (r.top_left, r.bottom_left),
                    (r.top_right, r.bottom_right),
                ]
            else:
                continue
            for a, b in lines:
                if abs(a.y - b.y) < 0.5 and abs(a.x - b.x) > 0.5:
                    horiz.append((a.y, min(a.x, b.x), max(a.x, b.x)))
                elif abs(a.x - b.x) < 0.5 and abs(a.y - b.y) > 0.5:
                    vert.append((a.x, min(a.y, b.y), max(a.y, b.y)))
    return horiz, vert


@dataclass
class Box:
    """A box students write in: its outline, its printed label, and the free writing area."""

    rect: pymupdf.Rect
    label: pymupdf.Rect

    @property
    def writing_area(self) -> pymupdf.Rect:
        """The larger of the parts of the box right of and below the label."""
        r, lab = self.rect, self.label
        right = pymupdf.Rect(lab.x1, r.y0, r.x1, r.y1)
        below = pymupdf.Rect(r.x0, lab.y1, r.x1, r.y1)
        return right if right.get_area() >= below.get_area() else below


def find_box(page: pymupdf.Page, label: str) -> Box:
    """Locate the drawn box enclosing the topmost word exactly equal to ``label``."""
    words = sorted(
        (w for w in page.get_text("words") if w[4] == label), key=lambda w: (w[1], w[0])
    )
    if not words:
        raise ValueError(f"label {label!r} not found on template page {page.number + 1}")
    lab = pymupdf.Rect(words[0][:4])
    cx, cy = (lab.x0 + lab.x1) / 2, (lab.y0 + lab.y1) / 2
    horiz, vert = _segments(page)
    tol = 1.0
    tops = [y for y, x0, x1 in horiz if y <= lab.y0 + tol and x0 - tol <= cx <= x1 + tol]
    bottoms = [y for y, x0, x1 in horiz if y >= lab.y1 - tol and x0 - tol <= cx <= x1 + tol]
    lefts = [x for x, y0, y1 in vert if x <= lab.x0 + tol and y0 - tol <= cy <= y1 + tol]
    rights = [x for x, y0, y1 in vert if x >= lab.x1 - tol and y0 - tol <= cy <= y1 + tol]
    if not (tops and bottoms and lefts and rights):
        raise ValueError(f"no box drawn around {label!r} on template page {page.number + 1}")
    return Box(pymupdf.Rect(max(lefts), max(tops), min(rights), min(bottoms)), lab)


@dataclass
class SimilarPages:
    """A class of near-identical template pages and the patches that tell them apart."""

    members: list[int]
    regions: list[tuple[int, int, int, int]]  # x0, y0, x1, y1 in DPI pixels
    patches: list[list[np.ndarray]]  # patches[k][r]: member k's pixels in region r

    @classmethod
    def build(cls, members: list[int], imgs: list[np.ndarray]) -> "SimilarPages":
        stack = np.stack([imgs[j].astype(np.int16) for j in members])
        differ = (stack.max(axis=0) - stack.min(axis=0)) > 80
        differ = cv2.dilate(differ.astype(np.uint8), np.ones((15, 15)))
        n, _, stats, _ = cv2.connectedComponentsWithStats(differ)
        regions = [(x, y, x + w, y + h) for x, y, w, h, area in stats[1:n] if area >= 100]
        patches = [[imgs[j][y0:y1, x0:x1] for x0, y0, x1, y1 in regions] for j in members]
        return cls(members, regions, patches)

    def costs(self, img: np.ndarray) -> list[float]:
        """Per member: mean squared difference of its patches at their best-fitting shift."""
        out = []
        for member_patches in self.patches:
            total = 0.0
            for (x0, y0, x1, y1), patch in zip(self.regions, member_patches):
                window = img[
                    max(y0 - MAX_SHIFT, 0) : y1 + MAX_SHIFT,
                    max(x0 - MAX_SHIFT, 0) : x1 + MAX_SHIFT,
                ]
                if window.shape[0] < patch.shape[0] or window.shape[1] < patch.shape[1]:
                    continue
                sq = cv2.matchTemplate(
                    window.astype(np.float32), patch.astype(np.float32), cv2.TM_SQDIFF
                )
                total += float(sq.min()) / patch.size
            out.append(total)
        return out


def _similar_classes(thumbs: list[np.ndarray]) -> list[list[int]]:
    """Connected components of template pages whose thumbnails correlate highly."""
    flat = np.stack([t.ravel() for t in thumbs])
    corr = flat @ flat.T
    parent = list(range(len(thumbs)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(corr > SIMILAR_PAGE_CORR, k=1))):
        parent[find(int(i))] = find(int(j))
    classes = defaultdict(list)
    for i in range(len(thumbs)):
        classes[find(i)].append(i)
    return [c for c in classes.values() if len(c) > 1]


@dataclass
class Template:
    path: Path
    aztec: str
    thumbs: list[np.ndarray]
    inks: list[np.ndarray]
    distinct: list[np.ndarray]  # ink not shared by most template pages
    cover_index: int
    bubblesheet_index: int
    cover_img: np.ndarray
    bubblesheet_img: np.ndarray
    cover_anchor: np.ndarray  # Aztec corners on the cover, in DPI pixels
    bubblesheet_anchor: dict[int, np.ndarray]  # ArUco corners on the bubblesheet
    cover_boxes: dict[str, Box]
    bubblesheet_boxes: dict[str, Box]
    prefilled: set[str]  # bubblesheet boxes printed on the template (e.g. per-student versions)
    is_cover: list[bool]
    is_bubblesheet: list[bool]
    similar: list[SimilarPages]
    header_kp: tuple  # ORB keypoints/descriptors of the ink most pages share, at ALIGN_DPI
    header_des: np.ndarray | None

    @property
    def version(self) -> str:
        return version_label(self.aztec)

    @classmethod
    def load(cls, path: Path, aztec: str) -> "Template":
        imgs, covers, bubbles, aztec_corners, markers = [], [], [], {}, {}
        with pymupdf.open(path) as doc:
            for index, page in enumerate(doc):
                img = to_gray(page)
                imgs.append(img)
                found = [
                    r
                    for r in zxingcpp.read_barcodes(img, formats=zxingcpp.BarcodeFormat.Aztec)
                    if r.text == aztec
                ]
                covers.append(bool(found))
                if found:
                    aztec_corners[index] = barcode_corners(found[0])
                page_markers = detect_bubblesheet_markers(img)
                bubbles.append(len(page_markers) == len(BUBBLESHEET_MARKER_IDS))
                if bubbles[-1]:
                    markers[index] = page_markers
            if sum(covers) != 1 or sum(bubbles) != 1:
                raise ValueError(
                    f"{path}: expected one cover (Aztec {aztec!r}) and one bubblesheet "
                    f"(ArUco markers), found {sum(covers)} and {sum(bubbles)}"
                )
            cover_index, bubblesheet_index = covers.index(True), bubbles.index(True)
            cover_boxes = {k: find_box(doc[cover_index], v) for k, v in COVER_LABELS.items()}
            bubblesheet_boxes = {
                k: find_box(doc[bubblesheet_index], v) for k, v in BUBBLESHEET_LABELS.items()
            }
        aligns = [align_image(img) for img in imgs]
        thumbs, inks = zip(*(features(a) for a in aligns))
        common = np.mean(inks, axis=0) >= 0.5
        common = cv2.dilate(common.astype(np.uint8), np.ones((3, 3))).astype(bool)
        header = np.where(np.mean([a < 160 for a in aligns], axis=0) >= 0.5, 0, 255)
        header_kp, header_des = _ORB.detectAndCompute(header.astype(np.uint8), None)
        prefilled = {
            k
            for k, box in bubblesheet_boxes.items()
            if ink_fraction(printed_ink(imgs[bubblesheet_index], box), box) >= INK_THRESHOLD
        }
        return cls(
            path=path,
            aztec=aztec,
            thumbs=list(thumbs),
            inks=list(inks),
            distinct=[ink & ~common for ink in inks],
            cover_index=cover_index,
            bubblesheet_index=bubblesheet_index,
            cover_img=imgs[cover_index],
            bubblesheet_img=imgs[bubblesheet_index],
            cover_anchor=aztec_corners[cover_index],
            bubblesheet_anchor=markers[bubblesheet_index],
            cover_boxes=cover_boxes,
            bubblesheet_boxes=bubblesheet_boxes,
            prefilled=prefilled,
            is_cover=covers,
            is_bubblesheet=bubbles,
            similar=[SimilarPages.build(c, imgs) for c in _similar_classes(list(thumbs))],
            header_kp=header_kp,
            header_des=header_des,
        )

    def align_pages(self, pages: list[ScannedPage]) -> list[np.ndarray]:
        """Per page, a 2x3 transform from its ALIGN_DPI image into this template's frame.

        Pages whose header can't be matched get the median transform of the other pages from
        the same file (or the identity if there are none).
        """
        found = [
            estimate_alignment(p.align_image(), self.header_kp, self.header_des) for p in pages
        ]
        by_file = defaultdict(list)
        for p, t in zip(pages, found):
            if t is not None:
                by_file[p.source].append(t)
        fallback = {f: np.median(ts, axis=0) for f, ts in by_file.items()}
        identity = np.eye(2, 3, dtype=np.float64)
        return [
            t if t is not None else fallback.get(p.source, identity) for p, t in zip(pages, found)
        ]

    def cost_matrix(
        self, pages: list[ScannedPage], transforms: list[np.ndarray] | None = None
    ) -> np.ndarray:
        """Matching cost of each scanned page (rows) against each template page (columns)."""
        if transforms is None:
            transforms = self.align_pages(pages)
        shape = letter_size(ALIGN_DPI)[::-1]
        feats = [features(warp_to(p.align_image(), t, shape)) for p, t in zip(pages, transforms)]
        scan = np.stack([thumb.ravel() for thumb, _ in feats])
        tpl = np.stack([t.ravel() for t in self.thumbs])
        cost = 1.0 - scan @ tpl.T
        for i, (_, scan_ink) in enumerate(feats):
            cost[i] += [
                ink_recall_cost(scan_ink, ink, distinct)
                for ink, distinct in zip(self.inks, self.distinct)
            ]
        scan_cover = np.array([p.aztec is not None for p in pages])[:, None]
        scan_bubble = np.array([p.bubblesheet for p in pages])[:, None]
        cost += FEATURE_PENALTY * (scan_cover != np.array(self.is_cover)[None, :])
        cost += FEATURE_PENALTY * (scan_bubble != np.array(self.is_bubblesheet)[None, :])
        return cost

    def refine(self, slots: list[int | None], render) -> list[int | None]:
        """Re-sort scanned pages among each class of near-identical template pages.

        ``render(i)`` returns the DPI image of scanned page ``i``. Within a class, each scanned
        page is compared with each member's distinguishing patches, searched over a window to
        allow for scanner shift, and the class is reassigned by least squared difference.
        """
        slots = list(slots)
        for similar in self.similar:
            scanned = [slots[j] for j in similar.members if slots[j] is not None]
            if not scanned:
                continue
            cost = np.array([similar.costs(render(i)) for i in scanned])
            rows, cols = linear_sum_assignment(cost)
            for j in similar.members:
                slots[j] = None
            for r, c in zip(rows, cols):
                slots[similar.members[c]] = scanned[r]
        return slots


def find_template(templates_dir: Path, aztec: str) -> Path | None:
    """``templates/<aztec>.pdf`` (or ``<version>.pdf``), or None if neither exists."""
    for candidate in (
        templates_dir / f"{aztec}.pdf",
        templates_dir / f"{version_label(aztec)}.pdf",
    ):
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Name/ID fill
# ---------------------------------------------------------------------------


def _px(rect: pymupdf.Rect) -> tuple[int, int, int, int]:
    s = DPI / 72
    return int(rect.x0 * s), int(rect.y0 * s), int(rect.x1 * s), int(rect.y1 * s)


def new_ink(registered: np.ndarray, template_img: np.ndarray, box: Box) -> np.ndarray:
    """Mask of ink in ``box``'s writing area that isn't printed on the template.

    ``registered`` is the scan warped into the template's pixel frame. The returned mask covers
    the box's outline rect (in template DPI pixels) with the label and border excluded.
    """
    x0, y0, x1, y1 = _px(box.rect)
    scan = registered[y0:y1, x0:x1] < 160
    printed = cv2.dilate((template_img[y0:y1, x0:x1] < 200).astype(np.uint8), np.ones((9, 9)))
    mask = scan & ~printed.astype(bool)
    keep = np.zeros_like(mask)
    ax0, ay0, ax1, ay1 = _px(box.writing_area)
    inset = int(4 * DPI / 72)
    keep[
        max(ay0 - y0, 0) + inset : ay1 - y0 - inset, max(ax0 - x0, 0) + inset : ax1 - x0 - inset
    ] = True
    mask &= keep
    # Drop isolated specks (scanner noise).
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2))).astype(bool)
    return mask


def printed_ink(template_img: np.ndarray, box: Box) -> np.ndarray:
    """Mask of ink printed in ``box``'s writing area on the template itself."""
    x0, y0, x1, y1 = _px(box.rect)
    mask = np.zeros((y1 - y0, x1 - x0), bool)
    ax0, ay0, ax1, ay1 = _px(box.writing_area)
    inset = int(4 * DPI / 72)
    ys = slice(max(ay0 - y0, 0) + inset, ay1 - y0 - inset)
    xs = slice(max(ax0 - x0, 0) + inset, ax1 - x0 - inset)
    mask[ys, xs] = template_img[y0:y1, x0:x1][ys, xs] < 160
    return mask


def ink_fraction(mask: np.ndarray, box: Box) -> float:
    x0, y0, x1, y1 = _px(box.writing_area)
    return float(mask.sum()) / max((x1 - x0) * (y1 - y0), 1)


def handwriting_png(registered: np.ndarray, box: Box, mask: np.ndarray) -> bytes:
    """PNG of the handwriting in ``box``, cropped to the ink, with printed matter removed."""
    x0, y0, x1, y1 = _px(box.rect)
    crop = registered[y0:y1, x0:x1]
    keep = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5))).astype(bool)
    clean = np.where(keep, crop, 255).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    pad = 6
    clean = clean[
        max(ys.min() - pad, 0) : ys.max() + pad + 1, max(xs.min() - pad, 0) : xs.max() + pad + 1
    ]
    _, buf = cv2.imencode(".png", clean)
    return buf.tobytes()


def register(
    img: np.ndarray,
    scan_pts: np.ndarray,
    tpl_pts: np.ndarray,
    template_img: np.ndarray,
    homography: bool,
):
    """Return (scan warped into the template frame, 3x3 template->scan pixel transform)."""
    if homography:
        h, _ = cv2.findHomography(scan_pts, tpl_pts)
    else:
        a, _ = cv2.estimateAffinePartial2D(scan_pts, tpl_pts)
        h = np.vstack([a, [0, 0, 1]])
    warped = cv2.warpPerspective(
        img,
        h,
        (template_img.shape[1], template_img.shape[0]),
        borderValue=255,
        flags=cv2.INTER_LINEAR,
    )
    return warped, np.linalg.inv(h)


def map_rect(rect: pymupdf.Rect, h: np.ndarray) -> pymupdf.Rect:
    """Map a rect in PDF points through a pixel-space transform."""
    s = DPI / 72
    corners = np.array(
        [[[rect.x0 * s, rect.y0 * s]], [[rect.x1 * s, rect.y0 * s]],
         [[rect.x1 * s, rect.y1 * s]], [[rect.x0 * s, rect.y1 * s]]],
        dtype=np.float32,
    )  # fmt: skip
    out = cv2.perspectiveTransform(corners, h).reshape(4, 2) / s
    return pymupdf.Rect(out[:, 0].min(), out[:, 1].min(), out[:, 0].max(), out[:, 1].max())


def registered_cover(page: pymupdf.Page | None, template: Template):
    """The scanned cover warped into the template frame (the template itself if missing)."""
    if page is None:
        return template.cover_img
    img = to_gray(page)
    found = [
        r
        for r in zxingcpp.read_barcodes(img, formats=zxingcpp.BarcodeFormat.Aztec)
        if r.text == template.aztec
    ]
    if not found:
        raise ValueError("cover Aztec code not found on re-read")
    # The Aztec code only pins down one corner of the page; ECC aligns the rest.
    warped, _ = register(
        img, barcode_corners(found[0]), template.cover_anchor, template.cover_img, homography=False
    )
    return refine_alignment(warped, template.cover_img)


def refine_alignment(img: np.ndarray, template_img: np.ndarray) -> np.ndarray:
    """Affine ECC alignment of a roughly registered scan onto the template image."""
    scale = 0.25

    def prep(a: np.ndarray) -> np.ndarray:
        small = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(small.astype(np.float32), (5, 5), 0)

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        _, warp = cv2.findTransformECC(
            prep(template_img), prep(img), warp, cv2.MOTION_AFFINE, criteria, None, 5
        )
    except cv2.error:
        logger.warning("ECC alignment did not converge; using the code-based registration")
        return img
    warp[:, 2] /= scale
    return cv2.warpAffine(
        img,
        warp,
        (template_img.shape[1], template_img.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderValue=255,
    )


def register_bubblesheet(img: np.ndarray, template: Template):
    """(scan warped into the template frame, template->scan transform) by the ArUco markers.

    None if fewer than two of the template's markers are found on the scan.
    """
    markers = detect_bubblesheet_markers(img)
    common = sorted(set(markers) & set(template.bubblesheet_anchor))
    if len(common) < 2:
        return None
    scan_pts = np.concatenate([markers[i] for i in common])
    tpl_pts = np.concatenate([template.bubblesheet_anchor[i] for i in common])
    return register(img, scan_pts, tpl_pts, template.bubblesheet_img, homography=True)


def fill_bubblesheet(
    bubble_page: pymupdf.Page, cover_page: pymupdf.Page | None, template: Template, inserted: bool
) -> tuple[dict[str, bool], list[str]]:
    """Stamp cover handwriting into blank bubblesheet boxes. Returns (filled flags, warnings)."""
    warnings: list[str] = []
    img = to_gray(bubble_page)
    if inserted:
        registered, tpl_to_scan = template.bubblesheet_img, np.eye(3)
    else:
        found = register_bubblesheet(img, template)
        if found is None:
            return {}, ["bubblesheet markers not found; Name/ID not checked"]
        registered, tpl_to_scan = found
    try:
        cover = registered_cover(cover_page, template)
    except ValueError as exc:
        cover = None
        warnings.append(f"cover not registered ({exc}); Name/ID not filled")

    filled = {}
    for key, box in template.bubblesheet_boxes.items():
        filled[key] = False
        if key in template.prefilled:
            continue
        mask = new_ink(registered, template.bubblesheet_img, box)
        if ink_fraction(mask, box) >= INK_THRESHOLD:
            continue
        if cover is None:
            warnings.append(f"bubblesheet {key} blank and cover unusable")
            continue
        cover_box = template.cover_boxes[key]
        cover_mask = new_ink(cover, template.cover_img, cover_box)
        if ink_fraction(cover_mask, cover_box) < INK_THRESHOLD:
            warnings.append(f"bubblesheet {key} blank, and cover {key} blank too")
            continue
        dest = map_rect(box.writing_area, tpl_to_scan)
        inset = 2
        dest = pymupdf.Rect(dest.x0 + inset, dest.y0 + inset, dest.x1 - inset, dest.y1 - inset)
        bubble_page.insert_image(
            dest, stream=handwriting_png(cover, cover_box, cover_mask), keep_proportion=True
        )
        filled[key] = True
        logger.info(f"bubblesheet {key} was blank; filled from the cover")
    return filled, warnings


def text_rotation(page: pymupdf.Page, tpl_to_scan: np.ndarray) -> int:
    """``insert_textbox`` rotation that writes text upright in a registered template's frame.

    The scan may be stored sideways (and shown upright by the page's /Rotate) or shown sideways;
    the template's x axis, mapped onto the displayed page, says which way is "along the line".
    """
    pts = cv2.perspectiveTransform(np.float32([[[0, 0]], [[100, 0]]]), tpl_to_scan).reshape(2, 2)
    dx, dy = pts[1] - pts[0]
    angle = int(round(np.degrees(np.arctan2(-dy, dx)) / 90)) % 4 * 90
    return (page.rotation + angle) % 360


def insert_fitting_text(
    page: pymupdf.Page, area: pymupdf.Rect, tpl_to_scan: np.ndarray, text: str
) -> bool:
    """Write ``text`` in ``area`` (template points) of a registered page, shrinking it to fit.

    The text is centered vertically in the template's frame. Returns False if it doesn't fit even
    at the smallest size.
    """
    rotate = text_rotation(page, tpl_to_scan)
    for size in (16, 14, 12, 11, 10, 9, 8, 7, 6):
        # insert_textbox starts at the top of its rect.
        pad = max((area.height - size * 1.3) / 2, 0)
        dest = map_rect(pymupdf.Rect(area.x0, area.y0 + pad, area.x1, area.y1), tpl_to_scan)
        if page.rotation:
            dest = dest * page.derotation_matrix
        if page.insert_textbox(dest, text, fontsize=size, fontname="helv", rotate=rotate) >= 0:
            return True
    return False


def stamp_bubblesheet(
    page: pymupdf.Page, template: Template, values: dict[str, str]
) -> tuple[dict[str, bool], list[str]]:
    """Type ``values`` (box key -> text) into the bubblesheet's blank Name/ID boxes.

    A box the student wrote in, or one printed on the template, is left alone. Returns (filled
    flags, warnings).
    """
    found = register_bubblesheet(to_gray(page), template)
    if found is None:
        return {}, ["bubblesheet markers not found; Name/ID not typed in"]
    registered, tpl_to_scan = found
    filled, warnings = {}, []
    for key, box in template.bubblesheet_boxes.items():
        filled[key] = False
        if key in template.prefilled or not values.get(key):
            continue
        if ink_fraction(new_ink(registered, template.bubblesheet_img, box), box) >= INK_THRESHOLD:
            continue
        inset = 3
        area = box.writing_area
        area = pymupdf.Rect(area.x0 + inset, area.y0 + inset, area.x1 - inset, area.y1 - inset)
        if insert_fitting_text(page, area, tpl_to_scan, values[key]):
            filled[key] = True
        else:
            warnings.append(f"bubblesheet {key} blank, but {values[key]!r} does not fit in it")
    return filled, warnings


# ---------------------------------------------------------------------------
# Exams
# ---------------------------------------------------------------------------


@dataclass
class Exam:
    # "serial:<serial>" for barcoded exams; for unbarcoded ones "netid:<exam>:<netid>" when
    # personalized, else "file:<sha256 prefix>"
    key: str
    label: str  # the serial, or the source file name
    pages: list[ScannedPage]
    netid: str | None = None  # personalized exams: the student
    base: str | None = None  # personalized exams: the base version they are grouped under
    template: Template | None = None
    slots: list[int | None] = field(default_factory=list)
    extras: list[int] = field(default_factory=list)
    dropped: list[int] = field(default_factory=list)  # totally blank unmatched pages
    filled: dict[str, bool] = field(default_factory=dict)
    held: str | None = None  # why the exam is held back from upload
    warnings: list[str] = field(default_factory=list)

    @property
    def barcoded(self) -> bool:
        return self.key.startswith("serial:")

    @property
    def sort_key(self) -> tuple:
        """Barcoded exams by serial, then personalized ones by netid, then other files."""
        if self.barcoded:
            return (0, serial_sort_key(self.label))
        if self.netid:
            return (1, (1, 0, self.netid))
        return (2, (1, 0, self.label))

    @property
    def cover_versions(self) -> list[str]:
        return sorted({p.aztec for p in self.pages if p.aztec})

    @property
    def form(self) -> str:
        """The version printed on the cover (per student, for personalized exams)."""
        if self.template is not None:
            return self.template.version
        return ",".join(version_label(a) for a in self.cover_versions)

    @property
    def version(self) -> str:
        """The version the exam is uploaded under."""
        return self.base or self.form

    def match(self, templates: dict[str, Template], sources: dict[Path, pymupdf.Document]) -> None:
        covers = set(self.cover_versions)
        if len(covers) > 1:
            self.warnings.append(f"several cover versions: {sorted(covers)}")
        if covers & templates.keys():
            self.template = templates[min(covers & templates.keys())]
        else:
            # No readable cover: pick the version whose template fits best.
            self.template = min(
                templates.values(),
                key=lambda t: t.cost_matrix(self.pages).min(axis=1).sum(),
            )
            self.warnings.append(f"no cover; version {self.template.version} inferred")
        transforms = self.template.align_pages(self.pages)
        cost = self.template.cost_matrix(self.pages, transforms)
        slots, extras = assign_pages(cost, MATCH_COST_THRESHOLD)

        def render(i: int) -> np.ndarray:
            """Page ``i`` at DPI, aligned to the template frame."""
            p = self.pages[i]
            t = transforms[i].copy()
            t[:, 2] *= DPI / ALIGN_DPI  # same similarity, in DPI pixels
            img = fit_letter(to_gray(sources[p.source][p.index]))
            return warp_to(img, t, letter_size(DPI)[::-1])

        self.slots = self.template.refine(slots, render)
        self.extras = [i for i in extras if not self.pages[i].blank]
        self.dropped = [i for i in extras if self.pages[i].blank]
        for j, slot in enumerate(self.slots):
            if slot is None:
                self.warnings.append(f"template page {j + 1} missing; blank template inserted")
        for i in self.extras:
            self.warnings.append(f"{self.pages[i].ref} matches no template page; appended")
        for p in self.pages:
            self.warnings += [f"{p.ref}: {w}" for w in p.warnings]

    def page_refs(self) -> list[str]:
        refs = [self.pages[s].ref if s is not None else "TEMPLATE" for s in self.slots]
        return refs + [self.pages[i].ref for i in self.extras]

    def is_identity(self) -> bool:
        return self.slots == list(range(len(self.slots))) and not self.extras and not self.dropped


def group_exams(
    pages_by_file: dict[Path, list[ScannedPage]], hashes: dict[Path, str]
) -> tuple[list[Exam], list[ScannedPage]]:
    """Group scanned pages into exams; return (exams, pages that belong to no exam).

    In a file where some page carries an exam serial, pages are grouped by serial (across all
    files, so an exam split between files is reassembled); pages without one are unassigned. A
    file where no page carries a serial is a single unbarcoded exam.
    """
    by_serial: dict[str, list[ScannedPage]] = defaultdict(list)
    exams, unassigned = [], []
    for path, pages in pages_by_file.items():
        if any(p.serial for p in pages):
            for p in pages:
                (by_serial[p.serial] if p.serial else unassigned).append(p)
        else:
            exams.append(Exam(key=f"file:{hashes[path][:16]}", label=path.name, pages=pages))
    exams += [Exam(key=f"serial:{s}", label=s, pages=ps) for s, ps in by_serial.items()]
    return exams, unassigned


def read_personalized(path: Path) -> list[dict[str, str]]:
    """Rows of the personalized-exams CSV, or [] if there is none."""
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def base_version(
    aztec: str, rows: list[dict[str, str]], existing: set[str]
) -> tuple[str, str | None] | None:
    """For a personalized exam's Aztec text, (netid, base version or None); None if it isn't one.

    The Aztec stem is ``<exam>-<netid>``. The student's CSV row is the one with that netid
    whose exam globs match ``<exam>``; the base version is ``<exam>-<label>`` for the first of
    its version labels in ``existing`` (version stems with a template), as in the
    personalization script.
    """
    exam, _, netid = version_label(aztec).partition("-")
    for row in rows:
        if row["netid"].strip() != netid:
            continue
        if not any(fnmatch(exam, pattern) for pattern in row["exam"].split()):
            continue
        for label in row["version"].split():
            if f"{exam}-{label}" in existing:
                return netid, f"{exam}-{label}"
        return netid, None
    return None


def personalize(exam: Exam, rows: list[dict[str, str]], existing: set[str]) -> None:
    """Give an unbarcoded personalized exam its netid key and base version."""
    if exam.barcoded or len(exam.cover_versions) != 1:
        return
    found = base_version(exam.cover_versions[0], rows, existing)
    if found is None:
        return
    exam.netid, exam.base = found
    exam.key = f"netid:{exam.form.partition('-')[0]}:{exam.netid}"
    if exam.base is None:
        exam.warnings.append(
            f"personalized for {exam.netid}, but no template for any of their versions; "
            f"kept as {exam.form}"
        )


def hold_reason(
    exam: Exam, emitted: dict[str, str], unbarcoded_forms: dict[str, str]
) -> str | None:
    """Why ``exam`` should be held back from upload, or None.

    ``emitted`` maps exam keys already emitted to their batch; ``unbarcoded_forms`` maps cover
    versions (forms) already emitted for unbarcoded exams to their batch.
    """
    if exam.key in emitted:
        return f"already emitted in batch {emitted[exam.key]}"
    if not exam.barcoded:
        covers = sum(p.aztec is not None for p in exam.pages)
        if covers > 1:
            return f"{covers} covers in one unbarcoded file; split it by hand"
        if exam.form in unbarcoded_forms:
            return (
                f"version {exam.form} already emitted in batch "
                f"{unbarcoded_forms[exam.form]} (rescan?)"
            )
    return None


# ---------------------------------------------------------------------------
# Gradescope submission exports
# ---------------------------------------------------------------------------

EXPORT_GLOB = "assignment_*_export"
# What to type in a blank bubblesheet ID box: "netid" (the email's local part, what the cover
# asks for) or "sid" (Gradescope's student ID).
STAMP_ID = ARTIFACTS_CONFIG.get("stamp_id", "netid")


@dataclass
class ExportSubmission:
    """One submission of a Gradescope "Export Submissions" download, already matched to a student."""

    export: str  # the export folder name, assignment_<id>_export
    submission_id: str
    pdf: bytes
    name: str
    email: str
    sid: str
    original_filename: str  # the file the submission was uploaded from
    aztecs: list[str | None] = field(default_factory=list)  # per page
    marker_pages: list[int] = field(default_factory=list)  # pages with bubblesheet markers
    version: str | None = None
    page_index: int | None = None  # the bubblesheet page
    filled: dict[str, bool] = field(default_factory=dict)
    skipped: str | None = None  # why it is left out of the output
    warnings: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"submission:{self.submission_id}"

    @property
    def file(self) -> str:
        return f"{self.export}/{self.submission_id}.pdf"

    @property
    def netid(self) -> str:
        return self.email.partition("@")[0]

    @property
    def sort_key(self) -> tuple:
        return serial_sort_key(self.submission_id)

    def stamp_values(self) -> dict[str, str]:
        return {"name": self.name, "id": self.sid if STAMP_ID == "sid" else self.netid}


def export_sources(raw_dir: Path) -> list[Path]:
    """Export folders in ``raw_dir``, plus export zips that haven't been extracted there."""
    folders = sorted(d for d in raw_dir.glob(EXPORT_GLOB) if d.is_dir())
    zips = sorted(z for z in raw_dir.glob(f"{EXPORT_GLOB}.zip") if not z.with_suffix("").is_dir())
    return folders + zips


def _meta(entry: dict, key: str):
    """A submission_metadata.yml value; its keys are Ruby symbols (":name")."""
    return entry.get(f":{key}", entry.get(key))


def read_export(path: Path) -> list[ExportSubmission]:
    """The submissions in a Gradescope submissions export, extracted (a folder) or zipped.

    Raises:
        ValueError: If the metadata is missing, or a submission doesn't have exactly one submitter.
    """
    if path.is_dir():
        name = path.name
        files = {p.name: p.read_bytes for p in path.iterdir() if p.is_file()}
    else:
        name = path.name.removesuffix(".zip")
        with zipfile.ZipFile(path) as zf:
            contents = {Path(m).name: zf.read(m) for m in zf.namelist() if not m.endswith("/")}
        files = {k: (lambda v=v: v) for k, v in contents.items()}
    if "submission_metadata.yml" not in files:
        raise ValueError(f"{path}: no submission_metadata.yml")
    metadata = yaml.safe_load(files["submission_metadata.yml"]()) or {}
    submissions = []
    for filename, entry in metadata.items():
        submitters = _meta(entry, "submitters") or []
        if len(submitters) != 1:
            raise ValueError(f"{name}/{filename}: {len(submitters)} submitters; expected one")
        if filename not in files:
            raise ValueError(f"{name}: {filename} is in the metadata but not the export")
        who = submitters[0]
        submissions.append(
            ExportSubmission(
                export=name,
                submission_id=Path(filename).stem,
                pdf=files[filename](),
                name=str(_meta(who, "name") or ""),
                email=str(_meta(who, "email") or ""),
                sid=str(_meta(who, "sid") or ""),
                original_filename=str(_meta(entry, "original_filename") or ""),
            )
        )
    return sorted(submissions, key=lambda sub: sub.sort_key)


def scan_submission(sub: ExportSubmission) -> None:
    """Read each page's cover Aztec code and bubblesheet markers."""
    with pymupdf.open(stream=sub.pdf, filetype="pdf") as doc:
        for index, page in enumerate(doc):
            img = to_gray(page)
            aztecs = [r.text for r in zxingcpp.read_barcodes(img, formats=zxingcpp.BarcodeFormat.Aztec)]
            sub.aztecs.append(aztecs[0] if aztecs else None)
            if len(detect_bubblesheet_markers(img)) >= 3:
                sub.marker_pages.append(index)


def submission_version(aztecs: list[str | None], rows: list[dict[str, str]], existing: set[str]) -> str | None:
    """The version (template stem) a submission's cover names: its base version if personalized."""
    for aztec in filter(None, aztecs):
        found = base_version(aztec, rows, existing)
        if found is not None and found[1] is not None:
            return found[1]
        if version_label(aztec) in existing:
            return version_label(aztec)
    return None


def resolve_export_versions(
    subs: list[ExportSubmission], rows: list[dict[str, str]], existing: set[str]
) -> None:
    """Set each submission's version from its cover, else that of most of its export.

    An export is one Gradescope assignment, i.e. one version.
    """
    by_export: dict[str, list[ExportSubmission]] = defaultdict(list)
    for sub in subs:
        sub.version = submission_version(sub.aztecs, rows, existing)
        by_export[sub.export].append(sub)
    for export, group in by_export.items():
        counts = Counter(sub.version for sub in group if sub.version)
        if len(counts) > 1:
            logger.warning(f"{export}: covers of several versions: {dict(counts)}")
        majority = counts.most_common(1)[0][0] if counts else None
        for sub in group:
            if sub.version is not None:
                continue
            if majority is None:
                sub.skipped = "no readable cover, and none in its export either; version unknown"
            else:
                sub.version = majority
                sub.warnings.append(f"no readable cover; version {majority}, as for the rest of {export}")


def duplicate_reason(sub: ExportSubmission, emitted_versions: dict[str, str]) -> str | None:
    """Why a submission's bubblesheet was already produced: it was uploaded from a batch's output.

    ``emitted_versions`` maps versions emitted from scanned PDFs to their (latest) batch.
    """
    stem = Path(sub.original_filename).stem
    if stem in emitted_versions:
        return f"uploaded from {sub.original_filename} of batch {emitted_versions[stem]}, which has its bubblesheet"
    return None


def find_bubblesheet_page(sub: ExportSubmission, doc: pymupdf.Document, template: Template) -> int | None:
    """Index of the submission's bubblesheet page, or None (with a warning) if there is none.

    The page with the bubblesheet markers; failing that (none or several), the page that best
    matches the template's bubblesheet.
    """
    if len(sub.marker_pages) == 1:
        return sub.marker_pages[0]
    pages = [
        ScannedPage(
            source=Path(sub.file),
            index=i,
            serial=None,
            aztec=sub.aztecs[i],
            bubblesheet=i in sub.marker_pages,
            image=cv2.imencode(".png", align_image(to_gray(page)))[1].tobytes(),
            blank=False,
        )
        for i, page in enumerate(doc)
    ]
    cost = template.cost_matrix(pages)[:, template.bubblesheet_index]
    best = int(np.argmin(cost))
    which = "no page has" if not sub.marker_pages else f"{len(sub.marker_pages)} pages have"
    if cost[best] > MATCH_COST_THRESHOLD:
        sub.warnings.append(f"{which} bubblesheet markers, and no page looks like the bubblesheet")
        return None
    sub.warnings.append(f"{which} bubblesheet markers; page {best + 1} chosen by layout")
    return best


# ---------------------------------------------------------------------------
# Ledgers
# ---------------------------------------------------------------------------

FILES_FIELDS = ["batch", "file", "sha256", "pages"]
EXAMS_FIELDS = ["batch", "key", "label", "version", "form", "status", "sources"]


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_ledger(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_ledger(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def new_batch_name(out_root: Path) -> str:
    base = datetime.now().astimezone().strftime("%Y-%m-%dT%H%M")
    name, n = base, 1
    while (out_root / name).exists():
        n += 1
        name = f"{base}-{n}"
    return name


def forget_batch(out_root: Path, batch: str) -> None:
    """Delete a batch folder and its ledger rows, so its raw files are processed again."""
    found = False
    for name, fields in ((FILES_LEDGER, FILES_FIELDS), (EXAMS_LEDGER, EXAMS_FIELDS)):
        rows = read_ledger(out_root / name)
        keep = [r for r in rows if r["batch"] != batch]
        if len(keep) != len(rows):
            found = True
            write_ledger(out_root / name, fields, keep)
            logger.info(f"{name}: removed {len(rows) - len(keep)} rows of batch {batch}")
    if (out_root / batch).is_dir():
        found = True
        shutil.rmtree(out_root / batch)
        logger.info(f"Deleted {out_root / batch}")
    if not found:
        logger.warning(f"No batch {batch!r} in {out_root}")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _append_exam(
    out: pymupdf.Document, exam: Exam, sources: dict, template_doc: pymupdf.Document
) -> int:
    """Append the exam's pages to ``out``, filling its bubblesheet; return the bubblesheet index."""
    template = exam.template
    start = out.page_count
    order = [(j, s) for j, s in enumerate(exam.slots)] + [(None, i) for i in exam.extras]
    for j, s in order:
        if s is None:
            out.insert_pdf(template_doc, from_page=j, to_page=j)
        else:
            p = exam.pages[s]
            out.insert_pdf(sources[p.source], from_page=p.index, to_page=p.index)
    bubble_out = start + template.bubblesheet_index
    cover_page = (
        out[start + template.cover_index] if exam.slots[template.cover_index] is not None else None
    )
    inserted = exam.slots[template.bubblesheet_index] is None
    exam.filled, warnings = fill_bubblesheet(out[bubble_out], cover_page, template, inserted)
    exam.warnings += warnings
    return bubble_out


def _save(doc: pymupdf.Document, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path, garbage=3, deflate=True)
    doc.close()


def _bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def process_exam_dir(raw_dir: Path, out_root: Path, dry_run: bool = False) -> Path | None:
    """Process the raw PDFs and submission exports in ``raw_dir`` not processed before.

    Writes a new batch in ``out_root``; returns the batch folder, or None if nothing was written.
    """
    files_ledger = read_ledger(out_root / FILES_LEDGER)
    exams_ledger = read_ledger(out_root / EXAMS_LEDGER)
    seen = {r["sha256"]: r["batch"] for r in files_ledger}

    hashes: dict[Path, str] = {}
    for path in sorted(raw_dir.glob("*.pdf")):
        sha = file_sha256(path)
        if sha in seen:
            logger.debug(f"{path.name}: already processed in batch {seen[sha]}")
        elif sha in hashes.values():
            logger.warning(f"{path.name}: identical to another new file; skipped")
        else:
            hashes[path] = sha

    done = {r["key"] for r in exams_ledger}
    submissions = [
        sub for source in export_sources(raw_dir) for sub in read_export(source) if sub.key not in done
    ]
    if not hashes and not submissions:
        logger.info(f"{raw_dir.name}: no new scanned PDFs or exported submissions")
        return None

    # Versions emitted from scanned PDFs (not exports), for spotting re-exported submissions.
    emitted_versions = {
        r["version"]: r["batch"]
        for r in exams_ledger
        if r["status"] == "emitted" and not r["key"].startswith("submission:")
    }
    for sub in submissions:
        sub.skipped = duplicate_reason(sub, emitted_versions)
    active = [sub for sub in submissions if not sub.skipped]
    for sub in active:
        scan_submission(sub)

    pages_by_file = {path: scan_stack(path) for path in hashes}
    exams, unassigned = group_exams(pages_by_file, hashes)
    for p in unassigned:
        logger.warning(f"{p.ref}: no serial number; left out of the output")
    rows = read_personalized(PERSONALIZED_CSV)
    template_stems = {version_label(p.stem): p.stem for p in (raw_dir / "templates").glob("*.pdf")}
    existing = set(template_stems)
    for exam in exams:
        personalize(exam, rows, existing)
    resolve_export_versions(active, rows, existing)
    active = [sub for sub in active if not sub.skipped]
    aztecs = sorted(
        {a for e in exams for a in e.cover_versions} | {template_stems[sub.version] for sub in active}
    )
    template_paths = {a: find_template(raw_dir / "templates", a) for a in aztecs}
    missing = [a for a, p in template_paths.items() if p is None]

    if dry_run:
        logger.info(
            f"{raw_dir.name}: {len(hashes)} new files: {', '.join(p.name for p in hashes)}"
        )
        for exam in sorted(exams, key=lambda e: e.sort_key):
            logger.info(f"  {exam.label}: {len(exam.pages)} pages, version {exam.version or '?'}")
        for export, group in sorted(_group_by(submissions, lambda sub: sub.export).items()):
            versions = Counter(sub.version for sub in group if not sub.skipped)
            skipped = [sub for sub in group if sub.skipped]
            logger.info(
                f"  {export}: {len(group)} new submissions; bubblesheets by version {dict(versions)}; "
                f"{len(skipped)} skipped"
            )
            for sub in skipped:
                logger.info(f"    skip {sub.submission_id}: {sub.skipped}")
        for a in missing:
            logger.warning(f"missing template: {raw_dir / 'templates' / f'{a}.pdf'}")
        return None
    if missing:
        raise FileNotFoundError(
            "Missing templates (nothing written): "
            + ", ".join(str(raw_dir / "templates" / f"{a}.pdf") for a in missing)
        )

    templates = {a: Template.load(p, a) for a, p in template_paths.items()}
    batch = new_batch_name(out_root)
    batch_dir = out_root / batch
    emitted = {r["key"]: r["batch"] for r in exams_ledger if r["status"] == "emitted"}
    unbarcoded_forms = {
        r.get("form") or r["version"]: r["batch"]
        for r in exams_ledger
        if r["status"] == "emitted" and r["key"].split(":")[0] not in ("serial", "submission")
    }

    sources = {path: pymupdf.open(path) for path in hashes}
    template_docs = {t.aztec: pymupdf.open(t.path) for t in templates.values()}
    manifest: list[dict] = []
    try:
        for exam in sorted(exams, key=lambda e: e.sort_key):
            exam.match(templates, sources)
            exam.held = hold_reason(exam, emitted, unbarcoded_forms)
            if exam.held:
                exam.warnings.append(f"HELD: {exam.held}")
            elif not exam.barcoded:
                unbarcoded_forms[exam.form] = batch

        by_version: dict[str, list[Exam]] = defaultdict(list)
        for exam in exams:
            if not exam.held:
                by_version[exam.version].append(exam)
        subs_by_version = _group_by(active, lambda sub: sub.version)
        for version in sorted(by_version.keys() | subs_by_version.keys()):
            exam_doc, bubble_doc = pymupdf.open(), pymupdf.open()
            version_exams = sorted(by_version.get(version, []), key=lambda e: e.sort_key)
            for exam in version_exams:
                template_doc = template_docs[exam.template.aztec]
                bubble_out = _append_exam(exam_doc, exam, sources, template_doc)
                bubble_doc.insert_pdf(exam_doc, from_page=bubble_out, to_page=bubble_out)
            if version_exams:
                _save(exam_doc, batch_dir / f"{version}.pdf")
                logger.info(f"{version}: {len(version_exams)} exams")
            else:
                exam_doc.close()
            version_subs = subs_by_version.get(version, [])
            template = templates[template_stems[version]]
            for sub in version_subs:
                with pymupdf.open(stream=sub.pdf, filetype="pdf") as doc:
                    sub.page_index = find_bubblesheet_page(sub, doc, template)
                    if sub.page_index is None:
                        sub.skipped = "no bubblesheet page found"
                        continue
                    bubble_doc.insert_pdf(doc, from_page=sub.page_index, to_page=sub.page_index)
                sub.filled, warnings = stamp_bubblesheet(bubble_doc[-1], template, sub.stamp_values())
                sub.warnings += warnings
                manifest.append(
                    {
                        "version": version,
                        "page": bubble_doc.page_count,
                        "submission_id": sub.submission_id,
                        "name": sub.name,
                        "email": sub.email,
                        "export": sub.export,
                    }
                )
            if version_subs:
                logger.info(f"{version}: {sum(not s.skipped for s in version_subs)} exported submissions")
            _save(bubble_doc, batch_dir / f"{version}-bubblesheets.pdf")

        for exam in (e for e in exams if e.held):
            doc = pymupdf.open()
            _append_exam(doc, exam, sources, template_docs[exam.template.aztec])
            _save(doc, batch_dir / "held" / f"{Path(exam.label).stem}.pdf")
    finally:
        for doc in [*sources.values(), *template_docs.values()]:
            doc.close()

    batch_dir.mkdir(parents=True, exist_ok=True)
    _write_report(batch_dir / "report.csv", exams, unassigned, submissions)
    if manifest:
        write_ledger(
            batch_dir / "bubblesheets.csv",
            ["version", "page", "submission_id", "name", "email", "export"],
            manifest,
        )
    write_ledger(
        out_root / FILES_LEDGER,
        FILES_FIELDS,
        files_ledger
        + [
            {"batch": batch, "file": p.name, "sha256": sha, "pages": len(pages_by_file[p])}
            for p, sha in hashes.items()
        ]
        + [
            {
                "batch": batch,
                "file": sub.file,
                "sha256": _bytes_sha256(sub.pdf),
                "pages": len(sub.aztecs) or "",
            }
            for sub in submissions
        ],
    )
    write_ledger(
        out_root / EXAMS_LEDGER,
        EXAMS_FIELDS,
        exams_ledger
        + [
            {
                "batch": batch,
                "key": e.key,
                "label": e.label,
                "version": e.version,
                "form": e.form,
                "status": "held" if e.held else "emitted",
                "sources": " ".join(sorted({p.source.name for p in e.pages})),
            }
            for e in sorted(exams, key=lambda e: e.sort_key)
        ]
        + [
            {
                "batch": batch,
                "key": sub.key,
                "label": sub.file,
                "version": sub.version or "",
                "form": sub.version or "",
                "status": "skipped" if sub.skipped else "emitted",
                "sources": sub.export,
            }
            for sub in submissions
        ],
    )

    for exam in exams:
        for w in exam.warnings:
            logger.warning(f"{exam.label}: {w}")
    for sub in submissions:
        for w in sub.warnings:
            logger.warning(f"{sub.file}: {w}")
        if sub.skipped:
            logger.info(f"{sub.file}: skipped: {sub.skipped}")
    held = [e for e in exams if e.held]
    emitted_subs = [sub for sub in submissions if not sub.skipped]
    logger.success(
        f"{raw_dir.name}: batch {batch}: {len(exams) - len(held)} exams for upload, "
        f"{len(held)} held, {len(unassigned)} unassigned pages; "
        f"{len(emitted_subs)} exported submissions' bubblesheets, "
        f"{len(submissions) - len(emitted_subs)} skipped -> {batch_dir}"
    )
    return batch_dir


def _group_by(items, key) -> dict:
    groups = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    return dict(groups)


def _write_report(
    path: Path,
    exams: list[Exam],
    unassigned: list[ScannedPage],
    submissions: list[ExportSubmission] = (),
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "exam",
                "version",
                "form",
                "netid",
                "status",
                "pages",
                "reordered",
                "dropped_blank",
                "name_filled",
                "id_filled",
                "warnings",
            ]
        )
        for exam in sorted(exams, key=lambda e: e.sort_key):
            writer.writerow(
                [
                    exam.label,
                    exam.version,
                    exam.form,
                    exam.netid or "",
                    "held" if exam.held else "emitted",
                    " ".join(exam.page_refs()),
                    not exam.is_identity(),
                    " ".join(exam.pages[i].ref for i in exam.dropped),
                    exam.filled.get("name", False),
                    exam.filled.get("id", False),
                    "; ".join(exam.warnings),
                ]
            )
        for p in unassigned:
            writer.writerow(
                ["", "", "", "", "unassigned", p.ref, "", "", "", "", "no serial number"]
            )
        for sub in submissions:
            writer.writerow(
                [
                    sub.key,
                    sub.version or "",
                    "",
                    sub.netid,
                    "skipped" if sub.skipped else "emitted",
                    "" if sub.page_index is None else f"{sub.file}:{sub.page_index + 1}",
                    "",
                    "",
                    sub.filled.get("name", False),
                    sub.filled.get("id", False),
                    "; ".join(([f"SKIPPED: {sub.skipped}"] if sub.skipped else []) + sub.warnings),
                ]
            )


def process_all(
    exam_dirs: list[Path] | None = None, dry_run: bool = False, forget: str | None = None
) -> None:
    """Process the given exam directories, or every subdirectory of ``RAW_DATA_DIR/artifacts``."""
    if not exam_dirs:
        exam_dirs = sorted(d for d in ARTIFACTS_RAW_DIR.glob("*") if d.is_dir())
    if not exam_dirs:
        logger.info(f"No exam directories in {ARTIFACTS_RAW_DIR}; skipping.")
        return
    for raw_dir in exam_dirs:
        # A bare exam name ("mt1") means RAW_DATA_DIR/artifacts/mt1.
        if not raw_dir.exists() and (ARTIFACTS_RAW_DIR / raw_dir).is_dir():
            raw_dir = ARTIFACTS_RAW_DIR / raw_dir
        if not raw_dir.is_dir():
            raise FileNotFoundError(f"No exam directory {raw_dir} (or {ARTIFACTS_RAW_DIR / raw_dir})")
        raw_dir = raw_dir.resolve()
        try:
            rel = raw_dir.relative_to(ARTIFACTS_RAW_DIR.resolve())
        except ValueError:
            rel = Path(raw_dir.name)
        out_root = ARTIFACTS_PROCESSED_DIR / rel
        if forget:
            forget_batch(out_root, forget)
        else:
            process_exam_dir(raw_dir, out_root, dry_run=dry_run)
