"""Classical spatial, spectral and stereo workloads; no learned models."""
import numpy as np


def _filters(xp):
    if xp.__name__ == 'cupy':
        from cupyx.scipy import ndimage
    else:
        from scipy import ndimage
    return ndimage


def focal_mean(data, size=15, *, xp=np):
    """NaN-aware square mean, clipped at image boundaries (no reflected pixels)."""
    if data.ndim != 2 or not isinstance(size, int) or size < 1 or size % 2 != 1:
        raise ValueError('Expected a 2-D image and positive odd window size')
    filt = _filters(xp)
    valid = xp.isfinite(data)
    total = filt.uniform_filter(xp.where(valid, data, 0).astype(xp.float32), size=size, mode='constant')
    count = filt.uniform_filter(valid.astype(xp.float32), size=size, mode='constant')
    # The count is an integer. Round away sliding-filter cancellation residue,
    # which could otherwise turn an all-invalid neighborhood into a valid one.
    count = xp.rint(count * (size*size))
    total = total * (size*size)
    out = xp.full(data.shape, xp.nan, dtype=xp.float32)
    xp.divide(total, count, out=out, where=count > 0)
    return out


def spectral_angle(cube, reference, *, xp=np):
    """Angle in radians to one supplied spectrum; not a classification model."""
    if cube.ndim != 3 or reference.shape != (cube.shape[-1],):
        raise ValueError('Expected y,x,wavelength cube and matching reference')
    dot = xp.sum(cube * reference, axis=-1)
    norm = xp.sqrt(xp.sum(cube*cube, axis=-1) * xp.sum(reference*reference))
    cosine = xp.full(dot.shape, xp.nan, dtype=xp.float32)
    xp.divide(dot, norm, out=cosine, where=norm > 0)
    return xp.arccos(xp.clip(cosine, -1, 1))


def _census(image, xp):
    h, w = image.shape
    desc = xp.zeros((h, w), dtype=xp.uint32)
    center = image[2:-2, 2:-2]
    for y in range(-2, 3):
        for x in range(-2, 3):
            if y or x:
                desc[2:-2, 2:-2] = ((desc[2:-2, 2:-2] << xp.uint32(1)) |
                                         (image[2+y:h-2+y, 2+x:w-2+x] < center).astype(xp.uint32))
    return desc


def stereo_census(left, right, disparity_min=-128, disparity_max=128, *, xp=np):
    """5x5 Census + 5x5 Hamming aggregation, integer winner-takes-all stereo.

    Convention: x_right = x_left + disparity. Ties choose the lowest disparity.
    Full support is required for both Census and aggregation windows. Costs are
    processed one disparity at a time: O(HW) memory, not an HWD cost volume.
    No rectification, RPC triangulation, subpixel fit or left-right check is done.
    """
    if left.ndim != 2 or left.shape != right.shape or min(left.shape) < 9:
        raise ValueError('Expected matching 2-D images at least 9x9')
    if not isinstance(disparity_min, int) or not isinstance(disparity_max, int) or disparity_min > disparity_max:
        raise ValueError('Invalid integer disparity interval')
    filt = _filters(xp)
    a, b = _census(left, xp), _census(right, xp)
    valid_left = filt.minimum_filter(xp.isfinite(left).astype(xp.uint8), size=5, mode='constant')
    valid_right = filt.minimum_filter(xp.isfinite(right).astype(xp.uint8), size=5, mode='constant')
    valid_left[:2] = valid_left[-2:] = 0
    valid_left[:, :2] = valid_left[:, -2:] = 0
    valid_right[:2] = valid_right[-2:] = 0
    valid_right[:, :2] = valid_right[:, -2:] = 0
    best = xp.full(left.shape, xp.inf, dtype=xp.float32)
    result = xp.full(left.shape, xp.nan, dtype=xp.float32)
    width = left.shape[1]
    for disparity in range(disparity_min, disparity_max+1):
        lo, hi = max(0, -disparity), min(width, width-disparity)
        if lo >= hi:
            continue
        cost = xp.zeros(left.shape, dtype=xp.float32)
        valid = xp.zeros(left.shape, dtype=xp.uint8)
        bits = a[:, lo:hi] ^ b[:, lo+disparity:hi+disparity]
        # Exact uint32 parallel population count, shared by NumPy and CuPy.
        bits = bits - ((bits >> 1) & xp.uint32(0x55555555))
        bits = (bits & xp.uint32(0x33333333)) + ((bits >> 2) & xp.uint32(0x33333333))
        bits = (bits + (bits >> 4)) & xp.uint32(0x0F0F0F0F)
        cost[:, lo:hi] = ((bits * xp.uint32(0x01010101)) >> 24).astype(xp.float32)
        valid[:, lo:hi] = valid_left[:, lo:hi] & valid_right[:, lo+disparity:hi+disparity]
        # Integer-valued aggregation avoids CPU/GPU floating-point tie drift.
        total = filt.convolve(cost, xp.ones((5, 5), dtype=xp.float32), mode='constant')
        support = filt.minimum_filter(valid, size=5, mode='constant') != 0
        improve = support & (total < best)
        best = xp.where(improve, total, best)
        result = xp.where(improve, xp.float32(disparity), result)
    return result


def disparity_metrics(prediction, truth, mask=None):
    """Report coverage separately from error; never silently discard failures."""
    eligible = np.isfinite(truth)
    if mask is not None:
        eligible &= mask
    valid = eligible & np.isfinite(prediction)
    error = np.abs(prediction[valid]-truth[valid])
    return dict(reference_pixels=int(eligible.sum()), evaluated_pixels=int(valid.sum()),
                coverage=float(valid.sum()/eligible.sum()) if eligible.any() else None,
                mae_pixels=float(error.mean()) if error.size else None,
                bad3_percent=float(100*np.mean(error > 3)) if error.size else None)
