import os
import sys
import time
import multiprocessing
from dask.distributed import performance_report


os.environ["ARRAY_MODULE"] = "numpy"

from terragpu import io
from terragpu import engine
from terragpu.array.raster import Raster
from terragpu.indices.wv_indices import add_indices
from dask.distributed import wait, progress


def main(filename, bands):

    # Read TIF Imagery

    # GPUtil.showUtilization()

    # with performance_report(filename="/att/nobackup/jacaraba/dask-report-gpu4-persistsconcat.html"):

    ### AS DATASET

    # Read imagery
    t1 = time.perf_counter()
    raster = io.imread(filename, bands)
    print(raster)
    t2 = time.perf_counter()
    print("Elapsed time read in seconds:", t2 - t1)

    # Calculate some indices
    t1 = time.perf_counter()
    raster = add_indices(raster, indices=[
        'dvi', 'ndvi', 'cs1', 'cs2', 'si', 'fdi', 'dwi', 'ndwi'])
    print(raster)
    #wait(raster)
    t2 = time.perf_counter()
    print("Elapsed time calculate in seconds:", t2 - t1)

    # Save to directory
    t1 = time.perf_counter()
    io.imsave(raster, "/lscratch/jacaraba/output.tif")
    t2 = time.perf_counter()
    print("Elapsed time save in seconds:", t2 - t1)

    ### AS DATARRAY
    #raster = io.read_tif_new(filename, bands)
    #print(raster.data)
    #raster = add_indices_new(raster, indices=['dvi'])

    return

if __name__ == '__main__':

    # filename to open
    #filename = '/att/nobackup/jacaraba/AGU2021/terragpu/terragpu/test/data/WV02_Gonji.tif'
    #filename = '/adapt/nobackup/projects/ilab/projects/Senegal/LCLUC_Senegal_Cloud/data/WV02_20101020_0-5000_data.tif'
    #filename = '/att/nobackup/jacaraba/AGU2021/terragpu/terragpu/test/data/ProvoSuperCubeNNref.tif'
    #filename = '/att/nobackup/walemu/NASA_NPP/CRPld_Map_Pred_and_Forec/EVHR/Gonji/M1BS/WV02_20111207_M1BS_103001000FB04600-toa.tif'
    #filename = '/lscratch/jacaraba/WV02_20111207_M1BS_103001000FB04600-toa.tif'
    filename = '/att/nobackup/mwooten3/Senegal_LCLUC/VHR/priority-tiles/Aki-tiles-ETZ/M1BS/WV02_20170126_M1BS_103001006391D100-toa.tif'

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
    #engine.configure_dask(
    #    device='gpu',
    #    local_directory='/lscratch/jacaraba')

    # Execute main function and calculate indices
    main(filename, bands)
