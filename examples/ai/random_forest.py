from terragpu import io
from terragpu import engine
from terragpu.array.raster import Raster
from terragpu.indices.wv_indices import add_indices


def main(filename, bands):

    # Read imagery
    raster = io.imread(filename, bands)
    print(raster)

    # Calculate some indices
    raster = add_indices(raster, indices=[
        'dvi', 'ndvi', 'cs1', 'cs2', 'si', 'fdi', 'dwi',
        'ndwi', 'gndvi', 'sr'])
    print(raster)

    # Save to directory
    io.imsave(raster, "/lscratch/jacaraba/output.tif", crs="EPSG:32618")

    return

if __name__ == '__main__':

    # Define your variables

    
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

    dataset_metadata = {
        'WV03_20141024_data.tif': {
            '0': 50000,
            '1': 50000,
            '2': 0,
            '3': 825,
            '4': 65000,
            '5': 0,
        },
        'WV03_20141118_data.tif': {
            '0': 50000,
            '1': 50000,
            '2': 65000,
            '3': 96703,
            '4': 65000,
            '5': 0,
        },
        'WV03_20141207_data.tif': {
            '0': 50000,
            '1': 50000,
            '2': 65000,
            '3': 95703,
            '4': 65000,
            '5': 0,
        },
        'WV03_20180114_data.tif': {
            '0': 50000,
            '1': 50000,
            '2': 65000,
            '3': 7769,
            '4': 0,
            '5': 186796,
        },
    }


    # Start dask cluster - dask scheduler must be started from main
    #engine.configure_dask(
    #    device='gpu',
    #    n_workers=4,
    #    local_directory='/lscratch/jacaraba')

    # Execute main function and calculate indices
    main(filename, bands)
