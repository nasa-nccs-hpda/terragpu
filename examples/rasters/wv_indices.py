import os
import sys
import time

#os.environ["CUDA_VISIBLE_DEVICES"] = ""
#os.environ["ARRAY_MODULE"] = "numpy"

from terragpu import io
from terragpu import engine
from terragpu.array.raster import Raster
from terragpu.indices.wv_indices import add_indices


def main(filename, bands):

    # Read TIF Imagery

    # GPUtil.showUtilization()

    # Read imagery
    t1 = time.perf_counter()
    
    raster = io.imread(filename, bands)
    
    t2 = time.perf_counter()

    print("Elapsed time read in seconds:", t2 - t1)

    t1 = time.perf_counter()

    # Calculate some indices
    raster = add_indices(raster, indices=[
        'dvi', 'ndvi', 'cs1', 'cs2', 'si', 'fdi', 'dwi', 'ndwi'])

    t2 = time.perf_counter()
    
    print("Elapsed time calculate in seconds:", t2 - t1)

    print(raster)

    # Save to directory
    io.imsave(raster, "/lscratch/jacaraba/output.tif")

    return

if __name__ == '__main__':

    # filename to open
    #filename = '/att/nobackup/jacaraba/AGU2021/terragpu/terragpu/test/data/WV02_Gonji.tif'
    #filename = '/adapt/nobackup/projects/ilab/projects/Senegal/LCLUC_Senegal_Cloud/data/WV02_20101020_0-5000_data.tif'
    #filename = '/att/nobackup/jacaraba/AGU2021/terragpu/terragpu/test/data/ProvoSuperCubeNNref.tif'
    filename = '/att/nobackup/walemu/NASA_NPP/CRPld_Map_Pred_and_Forec/EVHR/Gonji/M1BS/WV02_20111207_M1BS_103001000FB04600-toa.tif'
    
    bands = [
        'CoastalBlue',
        'Blue',
        'Green',
        'Yellow',
        'Red',
        'RedEdge',
        'NIR1',
        'NIR2'
    ]

    # Start dask cluster - dask scheduler must be started from main
    engine.configure_dask(local_directory='/lscratch/jacaraba')
    
    # Execute main function and calculate indices
    main(filename, bands)
