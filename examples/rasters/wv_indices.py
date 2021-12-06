import sys
from terragpu import io
from terragpu import engine

def main(filename):
    
    # Read TIF Imagery
    raster = io.imread(filename)
    print(raster)


if __name__ == '__main__':

    # filename to open
    filename = '/att/nobackup/jacaraba/AGU2021/terragpu/terragpu/test/data/ProvoSuperCubeNNref.tif'

    # Start dask cluster - dask scheduler must be started from main
    engine.configure_dask()

    main(filename)
