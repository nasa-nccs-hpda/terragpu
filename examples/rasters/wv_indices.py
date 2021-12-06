from terragpu import io
from terragpu import engine
from terragpu.array.raster import Raster
from terragpu.indices.wv_indices import add_indices

def main(filename, bands):

    # Read TIF Imagery
    # GPUtil.showUtilization()

    # option #1 - TerraGPU Raster object
    # raster = Raster(filename, bands)
    # print(raster.raster)

    # option #2 - Built-in backend object
    raster = io.imread(filename, bands)
    raster.attrs['band_names'] = [b.lower() for b in bands]

    # Calculate some indices
    raster = add_indices(raster, indices=[
        'dvi', 'ndvi', 'cs1', 'cs2', 'si', 'fdi', 'dwi', 'ndwi'])
    print(raster)


if __name__ == '__main__':

    # filename to open
    # filename = '/att/nobackup/jacaraba/AGU2021/terragpu/terragpu/test/data/ProvoSuperCubeNNref.tif'
    filename = '/att/nobackup/jacaraba/AGU2021/terragpu/terragpu/test/data/WV02_Gonji.tif'
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
