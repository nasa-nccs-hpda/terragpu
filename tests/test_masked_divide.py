import numpy as np
import pytest

from terragpu.engine import masked_divide


class StrictUfuncNamespace:
    """CPU regression double for CuPy's restricted ufunc keyword interface."""
    where = staticmethod(np.where)
    isfinite = staticmethod(np.isfinite)
    nan = np.nan

    @staticmethod
    def divide(a, b):
        return np.divide(a, b)


def test_masked_divide_without_where_keyword():
    a = np.array([6, 0, np.inf, np.nan, 8, 1], dtype='float32')
    b = np.array([3, 0, 0, 2, 4, np.inf], dtype='float32')
    valid = np.array([True, True, False, True, False, True])
    with np.errstate(divide='raise', invalid='raise'):
        result = masked_divide(a, b, valid, xp=StrictUfuncNamespace)
    np.testing.assert_equal(result, [2, np.nan, np.nan, np.nan, np.nan, np.nan])
    assert result.dtype == np.float32


@pytest.mark.gpu
def test_masked_divide_gpu():
    cp = pytest.importorskip('cupy')
    a = cp.asarray([6, 0, np.inf, np.nan], dtype=cp.float32)
    b = cp.asarray([3, 0, 0, 2], dtype=cp.float32)
    result = masked_divide(a, b, cp.ones(4, dtype=cp.bool_), xp=cp)
    np.testing.assert_equal(cp.asnumpy(result), [2, np.nan, np.nan, np.nan])
