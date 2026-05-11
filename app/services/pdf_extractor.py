import fitz
import re

SCAN_MARKERS_RE = re.compile(
    r"scanned\s+(by|with|using)|"
    r"camscanner|tap[\s\-]?scanner|adobe\s+scan|"
    r"сканировано|отсканировано",
    re.IGNORECASE,
)
LARGE_IMG_RATIO = 0.50
SCAN_MARKER_MAX_LEN = 200
MIN_CHARS = 30

def extract_pages_pymupdf(path: str) -> list[dict]:
    out = []
    try:
        doc = fitz.open(path)
    except Exception:
        return out

    for page in doc:
        try:
            text = page.get_text("text") or ""
        except Exception:
            text = ""

        page_area = max(page.rect.width * page.rect.height, 1.0)
        max_img_ratio = 0.0
        try:
            for img in page.get_images(full=True):
                xref = img[0]
                rects = page.get_image_rects(xref)
                for r in rects:
                    ratio = (r.width * r.height) / page_area
                    if ratio > max_img_ratio:
                        max_img_ratio = ratio
        except Exception:
            pass

        out.append({"text": text, "max_img_ratio": float(max_img_ratio)})

    doc.close()
    return out

def page_is_gap(page_info: dict) -> tuple[bool, str]:
    text = (page_info.get("text") or "").strip()
    img_ratio = page_info.get("max_img_ratio", 0.0)

    if len(text) < MIN_CHARS:
        return True, "empty_text"

    if len(text) <= SCAN_MARKER_MAX_LEN and SCAN_MARKERS_RE.search(text):
        return True, "scan_marker"

    if img_ratio >= LARGE_IMG_RATIO and len(text) < 400:
        return True, "image_heavy"

    return False, ""
