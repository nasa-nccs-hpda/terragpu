import os
import sys
import pytest

from terragpu import io


@pytest.mark.parametrize('filename', ['data/ProvoSuperCubeNNref.tif'])
def test_imread(filename):
    data = io.imread(filename)
    # assert filename
    # assert 
    # self.assertEqual(raster.data.shape[0], 8)


# def test_hls_init(self):
#    raster = Raster(filename=TIF_FILENAME, bands=BANDS)
#    self.assertEqual(raster.data.shape[0], 8)

# def test_xx_init(self):
#    raster = Raster(filename=TIF_FILENAME, bands=BANDS)
#    self.assertEqual(raster.data.shape[0], 8)

