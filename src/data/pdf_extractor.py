"""PDF text extraction with multi-strategy fallback.

Strategy order:
1. PyMuPDF (fitz) - fast, handles most modern PDFs
2. pdfminer.six - fallback for complex layouts
3. Tesseract OCR - last resort for scanned images (requires pytesseract)

Usage:
    python -m src.data.pdf_extractor
"""

import logging
from pathlib import Path

import fitz  # PyMuPDF

from configs.settings import settings

logger = logging.getLogger(__name__)


def extract_with_pymupdf(pdf_path: Path) -> str | None:
    """Extract text using PyMuPDF (fitz). Fast and reliable for most PDFs."""
    try:
        doc = fitz.open(str(pdf_path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        text = "\n".join(text_parts).strip()
        return text if len(text) > 100 else None  # Too short = likely failed
    except Exception as e:
        logger.debug("PyMuPDF failed for %s: %s", pdf_path.name, e)
        return None


def extract_with_pdfminer(pdf_path: Path) -> str | None:
    """Extract text using pdfminer.six. Better for complex layouts."""
    try:
        from pdfminer.high_level import extract_text

        text = extract_text(str(pdf_path)).strip()
        return text if len(text) > 100 else None
    except Exception as e:
        logger.debug("pdfminer failed for %s: %s", pdf_path.name, e)
        return None


def extract_with_ocr(pdf_path: Path) -> str | None:
    """Extract text using Tesseract OCR for scanned PDFs.

    Requires pytesseract and Tesseract to be installed.
    """
    try:
        import pytesseract

        doc = fitz.open(str(pdf_path))
        text_parts = []
        for page_num, page in enumerate(doc):
            # Render page as image at 300 DPI
            pix = page.get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")

            # OCR the image
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(img_bytes))
            page_text = pytesseract.image_to_string(img)
            text_parts.append(page_text)
        doc.close()

        text = "\n".join(text_parts).strip()
        return text if len(text) > 100 else None
    except ImportError:
        logger.warning("pytesseract not installed. Skipping OCR for %s", pdf_path.name)
        return None
    except Exception as e:
        logger.debug("OCR failed for %s: %s", pdf_path.name, e)
        return None


def extract_text(pdf_path: Path) -> tuple[str | None, str]:
    """Extract text from a PDF using multi-strategy fallback.

    Returns:
        Tuple of (extracted_text, method_used).
        text is None if all methods fail.
    """
    # Strategy 1: PyMuPDF
    text = extract_with_pymupdf(pdf_path)
    if text:
        return text, "pymupdf"

    # Strategy 2: pdfminer.six
    text = extract_with_pdfminer(pdf_path)
    if text:
        return text, "pdfminer"

    # Strategy 3: OCR (expensive, last resort)
    text = extract_with_ocr(pdf_path)
    if text:
        return text, "ocr"

    logger.warning("All extraction methods failed for %s", pdf_path.name)
    return None, "failed"


def extract_all(
    raw_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Extract text from all PDFs in the raw data directory.

    Saves extracted text as .txt files in the interim directory.

    Returns:
        Stats dict with counts per method.
    """
    raw_dir = raw_dir or settings.data.raw_dir
    output_dir = output_dir or settings.data.interim_dir

    stats = {"pymupdf": 0, "pdfminer": 0, "ocr": 0, "failed": 0, "skipped": 0}
    pdf_paths = sorted(raw_dir.rglob("*.pdf"))

    logger.info("Found %d PDFs to extract", len(pdf_paths))

    for pdf_path in pdf_paths:
        # Maintain year/case_id structure
        relative = pdf_path.relative_to(raw_dir)
        txt_path = output_dir / relative.with_suffix(".txt")

        if txt_path.exists() and txt_path.stat().st_size > 0:
            stats["skipped"] += 1
            continue

        text, method = extract_text(pdf_path)
        stats[method] += 1

        if text:
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text(text, encoding="utf-8")

    logger.info("Extraction complete: %s", stats)
    return stats


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    extract_all()
