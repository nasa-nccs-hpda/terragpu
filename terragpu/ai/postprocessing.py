import cv2
from scipy.ndimage import median_filter, binary_fill_holes


def _grow(merged_mask, eps=120):
    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
    return cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, struct)


def _denoise(merged_mask, eps=30):
    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
    return cv2.morphologyEx(merged_mask, cv2.MORPH_OPEN, struct)


def _binary_fill(merged_mask):
    return binary_fill_holes(merged_mask).astype(int)
