import pytesseract
from PIL import Image
import fitz
from app.core.config import settings

if settings.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD

OCR_LANG = "kaz+rus+eng"
OCR_DPI = 300
TESS_CONFIG = "--oem 1 --psm 3"

def page_to_image(path: str, page_num: int, dpi: int = OCR_DPI) -> Image.Image:
    doc = fitz.open(path)
    pix = doc[page_num].get_pixmap(dpi=dpi, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def ocr_page(path: str, page_num: int) -> str:
    img = page_to_image(path, page_num)
    try:
        return pytesseract.image_to_string(img, lang=OCR_LANG, config=TESS_CONFIG)
    except pytesseract.TesseractError as e:
        print(f"[OCR ERR] {path} p.{page_num}: {e}")
        return ""
