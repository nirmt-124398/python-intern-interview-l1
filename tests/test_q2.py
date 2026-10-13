"""Tests for the Question 2 image-quality gate (synthetic, deterministic)."""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from q2_image_quality import (  # noqa: E402
    blur_score_laplacian,
    brightness_mean,
    check_image_quality,
    contrast_std,
    make_blurred_image,
    make_clear_image,
    make_dark_image,
)


def test_clear_image_is_usable():
    result = check_image_quality(make_clear_image())
    print("clear   :", result)
    assert result["is_usable"] is True
    assert result["blur_score"] >= 100.0
    assert result["reason"] == "Image suitable for OCR"


def test_blurred_image_is_not_usable():
    result = check_image_quality(make_blurred_image())
    print("blurred :", result)
    assert result["is_usable"] is False
    assert result["blur_score"] < 100.0


def test_dark_image_is_not_usable():
    result = check_image_quality(make_dark_image())
    print("dark    :", result)
    assert result["is_usable"] is False
    assert result["brightness"] <= 60.0


def test_grayscale_input_path():
    gray = cv2.cvtColor(make_clear_image(), cv2.COLOR_BGR2GRAY)
    assert gray.ndim == 2
    result = check_image_quality(gray)
    print("gray    :", result)
    assert result["is_usable"] is True


def test_metric_helpers_return_floats():
    gray = cv2.cvtColor(make_clear_image(), cv2.COLOR_BGR2GRAY)
    assert isinstance(blur_score_laplacian(gray), float)
    assert isinstance(brightness_mean(gray), float)
    assert isinstance(contrast_std(gray), float)


def test_generators_are_deterministic():
    assert np.array_equal(make_clear_image(), make_clear_image())
    assert np.array_equal(make_blurred_image(), make_blurred_image())
    assert np.array_equal(make_dark_image(), make_dark_image())


def test_underexposed_reason_for_dark_image():
    reason = check_image_quality(make_dark_image())["reason"]
    assert "underexposed" in reason
