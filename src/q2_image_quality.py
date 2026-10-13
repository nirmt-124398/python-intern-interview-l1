"""Question 2 - Image quality gate for OCR preprocessing.

Decides whether a document image is sharp enough and correctly exposed to be
fed into an OCR pipeline.  Everything works on in-memory ``numpy`` arrays so the
module has no dependency on cameras, files or tesseract.

Threshold rationale
-------------------
* **Blur** - ``cv2.Laplacian(gray, cv2.CV_64F).var()`` is the standard
  "focus measure" used across the OCR literature (Pezeshki et al., document
  image quality assessment).  Sharp document text typically scores
  ``> 100-150``; visibly blurred scans fall ``< 50``.  ``100.0`` is therefore a
  sensible default cutoff for document OCR.
* **Brightness** - keeping the mean grayscale in ``(60, 200)`` preserves enough
  separation between dark text and light background for adaptive/Otsu
  thresholding.  Below ~60 the ink and paper collapse together; above ~200 the
  background saturates and thin strokes are lost.
* All three thresholds are *tunable* and should be re-calibrated on a labelled
  validation set for each camera / lighting setup.
"""

from __future__ import annotations

import cv2
import numpy as np

# Standard document-OCR defaults; see module docstring for justification.
DEFAULT_BLUR_THRESHOLD = 100.0
DEFAULT_DARK_THRESHOLD = 60.0
DEFAULT_BRIGHT_THRESHOLD = 200.0

_IMAGE_SHAPE = (480, 640)  # (height, width)


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Return a single-channel uint8 grayscale view of ``image``.

    Accepts a 2-D grayscale array, a 3-channel BGR array (OpenCV default) or a
    3-channel RGB array.  Colour arrays are collapsed with ``cv2.cvtColor``;
    because only luminance statistics are computed, the minor BGR/RGB channel
    ordering difference does not change the outcome.
    """
    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(
            f"Unsupported image shape {image.shape}; expected (H, W) or (H, W, 3)"
        )
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def blur_score_laplacian(gray: np.ndarray) -> float:
    """Focus measure via variance of the Laplacian (higher = sharper)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_mean(gray: np.ndarray) -> float:
    """Mean grayscale intensity in the 0-255 range."""
    return float(np.mean(gray))


def contrast_std(gray: np.ndarray) -> float:
    """Standard deviation of grayscale intensity (extra diagnostic)."""
    return float(np.std(gray))


def check_image_quality(
    image: np.ndarray,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    dark_threshold: float = DEFAULT_DARK_THRESHOLD,
    bright_threshold: float = DEFAULT_BRIGHT_THRESHOLD,
) -> dict[str, object]:
    """Assess whether ``image`` is suitable for OCR.

    Args:
        image: BGR, RGB or grayscale ``numpy`` array.
        blur_threshold: Minimum Laplacian variance considered sharp.
        dark_threshold: Brightness below this is treated as underexposed.
        bright_threshold: Brightness at/above this is treated as overexposed.

    Returns:
        dict with keys ``is_usable`` (bool), ``blur_score``, ``brightness``,
        ``contrast`` (floats, rounded to 1 decimal) and ``reason`` (str).
    """
    gray = _to_gray(image)

    blur_score = blur_score_laplacian(gray)
    brightness = brightness_mean(gray)
    contrast = contrast_std(gray)

    is_sharp = blur_score >= blur_threshold
    is_exposed = dark_threshold < brightness < bright_threshold
    is_usable = is_sharp and is_exposed

    return {
        "is_usable": bool(is_usable),
        "blur_score": round(blur_score, 1),
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
        "reason": _build_reason(blur_score, brightness, blur_threshold,
                                dark_threshold, bright_threshold),
    }


def _build_reason(
    blur_score: float,
    brightness: float,
    blur_threshold: float,
    dark_threshold: float,
    bright_threshold: float,
) -> str:
    """Compose a human-readable verdict from the raw measurements."""
    issues: list[str] = []
    if blur_score < blur_threshold:
        issues.append("blurred")
    if brightness <= dark_threshold:
        issues.append("underexposed")
    if brightness >= bright_threshold:
        issues.append("overexposed")

    if not issues:
        return "Image suitable for OCR"
    if set(issues) == {"blurred", "underexposed"}:
        return "Image may be blurred or underexposed"
    return "Image is " + " and ".join(issues)


# ---------------------------------------------------------------------------
# Deterministic synthetic image generators (no files required).
# ---------------------------------------------------------------------------

def _blank_canvas() -> np.ndarray:
    """White 3-channel canvas matching ``_IMAGE_SHAPE``."""
    return np.full((*_IMAGE_SHAPE, 3), 255, dtype=np.uint8)


def _draw_text_bars(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Paint black bars + text onto ``image`` to emulate a text document.

    Filled rectangles carry most of the ink (realistic ~35-45% page coverage);
    a rendered text line adds genuine glyph edges so the Laplacian focus
    measure reacts to text sharpness rather than only to block edges.
    """
    h, w = image.shape[:2]
    margin = 48
    bar_h = 30
    gap = 10
    usable = w - 2 * margin
    y = margin
    row = 0
    while y + bar_h <= h - margin:
        if row == 1:
            cv2.putText(
                image,
                "Document OCR sample line",
                (margin, y + bar_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            # Uneven line lengths look like natural justified paragraphs.
            line_w = int(usable * rng.uniform(0.75, 0.95))
            cv2.rectangle(image, (margin, y), (margin + line_w, y + bar_h),
                          (0, 0, 0), thickness=-1)
        y += bar_h + gap
        row += 1
    return image


def make_clear_image() -> np.ndarray:
    """Sharply rendered, well-exposed document image (should be usable)."""
    rng = np.random.default_rng(42)
    return _draw_text_bars(_blank_canvas(), rng)


def make_blurred_image() -> np.ndarray:
    """Clear image degraded with a 15x15 Gaussian blur (should be unusable)."""
    clear = make_clear_image()
    return cv2.GaussianBlur(clear, (15, 15), 0)


def make_dark_image() -> np.ndarray:
    """Clear image scaled to 25% brightness (should be unusable)."""
    clear = make_clear_image().astype(np.float32) * 0.25
    return np.clip(clear, 0, 255).astype(np.uint8)


if __name__ == "__main__":  # pragma: no cover - manual demo
    for name, factory in (
        ("clear", make_clear_image),
        ("blurred", make_blurred_image),
        ("dark", make_dark_image),
    ):
        result = check_image_quality(factory())
        print(f"{name:>8}: {result}")
