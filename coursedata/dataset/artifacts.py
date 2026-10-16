"""Scanned exam stacks -> per-version PDFs for uploading to Gradescope.

Commands are wired into:
    python -m coursedata.dataset process artifacts [EXAM_DIR ...]

Each exam directory (e.g. ``RAW_DATA_DIR/artifacts/ch6``) holds scanner PDFs, each a stack of
concatenated exams, plus ``templates/<version code>.pdf`` -- one blank exam per version. Every
scanned page carries a Code39 serial number (the last character is a mod-43 check character);
the cover additionally carries an Aztec code whose text is the version code (e.g.
``ch6-16A.qns``), and the Gradescope bubblesheet carries ArUco markers 0-3 in its corners.

Output goes to the same relative path under ``PROCESSED_DATA_DIR``:

* ``<version>.pdf``: every exam of that version, sorted by serial number, with each exam's pages
  put in template order. A missing page is replaced by the blank template page; a scanned page
  that matches no template page is appended after the exam's template pages.
* ``<version>-bubblesheets.pdf``: only the bubblesheet pages, in the same order. A blank Name or
  ID box is filled with an image of what the student wrote in the matching box on the cover.
* ``report.csv``: one row per exam, recording the page mapping, fills, and any warnings.

Here ``<version>`` is the Aztec text without its extension (``ch6-16A``).
"""

from collections import defaultdict
import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
from loguru import logger
import numpy as np
import pymupdf
from scipy.optimize import linear_sum_assignment
import zxingcpp

from coursedata.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, _config

ARTIFACTS_CONFIG = _config.get("artifacts", {})
ARTIFACTS_RAW_DIR = RAW_DATA_DIR / "artifacts"
ARTIFACTS_PROCESSED_DIR = PROCESSED_DATA_DIR / "artifacts"

# Labels of the boxes students write in, found by exact word match on the template pages.
COVER_LABELS = ARTIFACTS_CONFIG.get("cover_labels", {"name": "Name:", "id": "NetID:"})
BUBBLESHEET_LABELS = ARTIFACTS_CONFIG.get("bubblesheet_labels", {"name": "Name", "id": "ID"})

DPI = 200  # render resolution for barcode reading and registration
RETRY_DPI = 300  # second try at the Code39 strip when the first render doesn't decode
THUMB_SCALE = 1 / 8  # matching thumbnails are DPI * THUMB_SCALE = 25 dpi
INK_SCALE = 1 / 4  # ink masks for matching are DPI * INK_SCALE = 50 dpi
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


def version_label(aztec_text: str) -> str:
    """Version label for output filenames: the Aztec text without its extension."""
    return Path(aztec_text).stem


def serial_sort_key(serial: str) -> tuple:
    """Sort numeric serials numerically, anything else after them as strings."""
    return (0, int(serial), "") if serial.isdigit() else (1, 0, serial)


def to_gray(page: pymupdf.Page, dpi: int = DPI) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()


def thumbnail(img: np.ndarray) -> np.ndarray:
    """Blurred, standardized low-res thumbnail with the barcode strip masked out."""
    small = cv2.resize(img, None, fx=THUMB_SCALE, fy=THUMB_SCALE, interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small.astype(np.float32), (5, 5), 0)
    small[: int(small.shape[0] * BARCODE_STRIP)] = 255
    small -= small.mean()
    norm = np.linalg.norm(small)
    return small / norm if norm else small


def ink_mask(img: np.ndarray) -> np.ndarray:
    """Low-res mask of dark pixels, for matching."""
    small = cv2.resize(img, None, fx=INK_SCALE, fy=INK_SCALE, interpolation=cv2.INTER_AREA)
    return small < 160


def ink_recall_cost(scan_ink: np.ndarray, tpl_ink: np.ndarray, distinct: np.ndarray) -> float:
    """1 - fraction of ``distinct`` template ink found in the scan, after aligning the two."""
    if scan_ink.shape != tpl_ink.shape:
        scan_ink = cv2.resize(
            scan_ink.astype(np.uint8), tpl_ink.shape[::-1], interpolation=cv2.INTER_NEAREST
        ).astype(bool)
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
    thumb: np.ndarray
    ink: np.ndarray  # packed bits of ink_mask()
    ink_shape: tuple[int, int]
    warnings: list[str] = field(default_factory=list)

    @property
    def ref(self) -> str:
        return f"{self.source.name}:{self.index + 1}"

    def ink_mask(self) -> np.ndarray:
        n = self.ink_shape[0] * self.ink_shape[1]
        return np.unpackbits(self.ink)[:n].reshape(self.ink_shape).astype(bool)


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
        serial, ok = split_code39(text)
        if not ok:
            warnings.append(f"Code39 {text!r} fails its check character")
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
            mask = ink_mask(img)
            pages.append(
                ScannedPage(
                    source=pdf_path,
                    index=index,
                    serial=serial,
                    aztec=aztecs[0] if aztecs else None,
                    bubblesheet=len(detect_bubblesheet_markers(img)) >= 3,
                    thumb=thumbnail(img).astype(np.float16),
                    ink=np.packbits(mask),
                    ink_shape=mask.shape,
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
    is_cover: list[bool]
    is_bubblesheet: list[bool]
    similar: list[SimilarPages]

    @property
    def version(self) -> str:
        return version_label(self.aztec)

    @classmethod
    def load(cls, path: Path, aztec: str) -> "Template":
        thumbs, imgs, covers, bubbles, aztec_corners, markers = [], [], [], [], {}, {}
        with pymupdf.open(path) as doc:
            for index, page in enumerate(doc):
                img = to_gray(page)
                imgs.append(img)
                thumbs.append(thumbnail(img))
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
            inks = [ink_mask(img) for img in imgs]
            common = np.mean(inks, axis=0) >= 0.5
            common = cv2.dilate(common.astype(np.uint8), np.ones((3, 3))).astype(bool)
            bubblesheet_boxes = {
                k: find_box(doc[bubblesheet_index], v) for k, v in BUBBLESHEET_LABELS.items()
            }
        return cls(
            path=path,
            aztec=aztec,
            thumbs=thumbs,
            inks=inks,
            distinct=[ink & ~common for ink in inks],
            cover_index=cover_index,
            bubblesheet_index=bubblesheet_index,
            cover_img=imgs[cover_index],
            bubblesheet_img=imgs[bubblesheet_index],
            cover_anchor=aztec_corners[cover_index],
            bubblesheet_anchor=markers[bubblesheet_index],
            cover_boxes=cover_boxes,
            bubblesheet_boxes=bubblesheet_boxes,
            is_cover=covers,
            is_bubblesheet=bubbles,
            similar=[SimilarPages.build(c, imgs) for c in _similar_classes(thumbs)],
        )

    def cost_matrix(self, pages: list[ScannedPage]) -> np.ndarray:
        """Matching cost of each scanned page (rows) against each template page (columns)."""
        scan = np.stack([p.thumb.ravel().astype(np.float32) for p in pages])
        tpl = np.stack([t.ravel() for t in self.thumbs])
        cost = 1.0 - scan @ tpl.T
        for i, page in enumerate(pages):
            scan_ink = page.ink_mask()
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


def find_template(templates_dir: Path, aztec: str) -> Path:
    for candidate in (
        templates_dir / f"{aztec}.pdf",
        templates_dir / f"{version_label(aztec)}.pdf",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No template for version {aztec!r}: expected {templates_dir / f'{aztec}.pdf'}"
    )


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


def register(img: np.ndarray, scan_pts: np.ndarray, tpl_pts: np.ndarray, homography: bool):
    """Return (scan warped into the template frame, 3x3 template->scan pixel transform)."""
    if homography:
        h, _ = cv2.findHomography(scan_pts, tpl_pts)
    else:
        a, _ = cv2.estimateAffinePartial2D(scan_pts, tpl_pts)
        h = np.vstack([a, [0, 0, 1]])
    warped = cv2.warpPerspective(
        img, h, (img.shape[1], img.shape[0]), borderValue=255, flags=cv2.INTER_LINEAR
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
    warped, _ = register(img, barcode_corners(found[0]), template.cover_anchor, homography=False)
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


def fill_bubblesheet(
    bubble_page: pymupdf.Page, cover_page: pymupdf.Page | None, template: Template, inserted: bool
) -> tuple[dict[str, bool], list[str]]:
    """Stamp cover handwriting into blank bubblesheet boxes. Returns (filled flags, warnings)."""
    warnings: list[str] = []
    img = to_gray(bubble_page)
    if inserted:
        registered, tpl_to_scan = img, np.eye(3)
    else:
        markers = detect_bubblesheet_markers(img)
        common = sorted(set(markers) & set(template.bubblesheet_anchor))
        if len(common) < 2:
            return {}, ["bubblesheet markers not found; Name/ID not checked"]
        scan_pts = np.concatenate([markers[i] for i in common])
        tpl_pts = np.concatenate([template.bubblesheet_anchor[i] for i in common])
        registered, tpl_to_scan = register(img, scan_pts, tpl_pts, homography=len(common) >= 2)
    try:
        cover = registered_cover(cover_page, template)
    except ValueError as exc:
        cover = None
        warnings.append(f"cover not registered ({exc}); Name/ID not filled")

    filled = {}
    for key, box in template.bubblesheet_boxes.items():
        filled[key] = False
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


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


@dataclass
class Exam:
    serial: str
    pages: list[ScannedPage]
    template: Template | None = None
    slots: list[int | None] = field(default_factory=list)
    extras: list[int] = field(default_factory=list)
    filled: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def match(self, templates: dict[str, Template], sources: dict[Path, pymupdf.Document]) -> None:
        covers = {p.aztec for p in self.pages if p.aztec}
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
        cost = self.template.cost_matrix(self.pages)
        slots, self.extras = assign_pages(cost, MATCH_COST_THRESHOLD)

        def render(i: int) -> np.ndarray:
            p = self.pages[i]
            return to_gray(sources[p.source][p.index])

        self.slots = self.template.refine(slots, render)
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
        return self.slots == list(range(len(self.slots))) and not self.extras


def _append_exam(out: pymupdf.Document, exam: Exam, sources: dict, template_doc) -> int:
    """Append the exam's pages to ``out``; return the output index of its bubblesheet."""
    bubble_out = None
    order = [(j, s) for j, s in enumerate(exam.slots)] + [(None, i) for i in exam.extras]
    for j, s in order:
        if j == exam.template.bubblesheet_index:
            bubble_out = out.page_count
        if s is None:
            out.insert_pdf(template_doc, from_page=j, to_page=j)
        else:
            p = exam.pages[s]
            out.insert_pdf(sources[p.source], from_page=p.index, to_page=p.index)
    return bubble_out


def process_exam_dir(raw_dir: Path, output_dir: Path) -> None:
    """Reorganize every scanned stack in ``raw_dir`` into per-version PDFs in ``output_dir``."""
    stacks = sorted(raw_dir.glob("*.pdf"))
    if not stacks:
        logger.warning(f"No scanned PDFs in {raw_dir}")
        return
    pages = [p for stack in stacks for p in scan_stack(stack)]

    by_serial: dict[str, list[ScannedPage]] = defaultdict(list)
    unassigned = []
    for p in pages:
        (by_serial[p.serial] if p.serial else unassigned).append(p)
    for p in unassigned:
        logger.warning(f"{p.ref}: no serial number; left out of the output")

    aztecs = sorted({p.aztec for p in pages if p.aztec})
    templates = {a: Template.load(find_template(raw_dir / "templates", a), a) for a in aztecs}
    exams = [Exam(serial, ps) for serial, ps in by_serial.items()]
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = {s: pymupdf.open(s) for s in stacks}
    try:
        for exam in exams:
            exam.match(templates, sources)
        for template in templates.values():
            version_exams = sorted(
                (e for e in exams if e.template is template),
                key=lambda e: serial_sort_key(e.serial),
            )
            exam_doc, bubble_doc = pymupdf.open(), pymupdf.open()
            with pymupdf.open(template.path) as template_doc:
                for exam in version_exams:
                    bubble_out = _append_exam(exam_doc, exam, sources, template_doc)
                    cover_slot = exam.slots[template.cover_index]
                    cover_page = (
                        exam_doc[bubble_out - template.bubblesheet_index + template.cover_index]
                        if cover_slot is not None
                        else None
                    )
                    inserted = exam.slots[template.bubblesheet_index] is None
                    exam.filled, warnings = fill_bubblesheet(
                        exam_doc[bubble_out], cover_page, template, inserted
                    )
                    exam.warnings += warnings
                    bubble_doc.insert_pdf(exam_doc, from_page=bubble_out, to_page=bubble_out)
            exam_path = output_dir / f"{template.version}.pdf"
            bubble_path = output_dir / f"{template.version}-bubblesheets.pdf"
            exam_doc.save(exam_path, garbage=3, deflate=True)
            bubble_doc.save(bubble_path, garbage=3, deflate=True)
            logger.info(
                f"{template.version}: {len(version_exams)} exams -> {exam_path.name}, "
                f"{bubble_path.name}"
            )
    finally:
        for doc in sources.values():
            doc.close()

    _write_report(output_dir / "report.csv", exams, unassigned)
    flagged = [e for e in exams if e.warnings]
    for exam in flagged:
        for w in exam.warnings:
            logger.warning(f"serial {exam.serial}: {w}")
    reordered = [e.serial for e in exams if not e.is_identity()]
    if reordered:
        logger.info(f"Pages reordered/patched for serials: {', '.join(sorted(reordered))}")
    logger.success(
        f"{raw_dir.name}: {len(exams)} exams, {len(flagged)} with warnings, "
        f"{len(unassigned)} unassigned pages -> {output_dir}"
    )


def _write_report(path: Path, exams: list[Exam], unassigned: list[ScannedPage]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["serial", "version", "pages", "reordered", "name_filled", "id_filled", "warnings"]
        )
        for exam in sorted(exams, key=lambda e: serial_sort_key(e.serial)):
            writer.writerow(
                [
                    exam.serial,
                    exam.template.version,
                    " ".join(exam.page_refs()),
                    not exam.is_identity(),
                    exam.filled.get("name", False),
                    exam.filled.get("id", False),
                    "; ".join(exam.warnings),
                ]
            )
        for p in unassigned:
            writer.writerow(["", "", p.ref, "", "", "", "no serial number; not in output"])


def process_all(exam_dirs: list[Path] | None = None) -> None:
    """Process the given exam directories, or every subdirectory of ``RAW_DATA_DIR/artifacts``."""
    if not exam_dirs:
        exam_dirs = sorted(d for d in ARTIFACTS_RAW_DIR.glob("*") if d.is_dir())
    if not exam_dirs:
        logger.info(f"No exam directories in {ARTIFACTS_RAW_DIR}; skipping.")
        return
    for raw_dir in exam_dirs:
        raw_dir = raw_dir.resolve()
        try:
            rel = raw_dir.relative_to(ARTIFACTS_RAW_DIR.resolve())
        except ValueError:
            rel = Path(raw_dir.name)
        process_exam_dir(raw_dir, ARTIFACTS_PROCESSED_DIR / rel)
