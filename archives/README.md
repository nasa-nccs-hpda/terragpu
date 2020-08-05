# purpose: To build a Random Forest model with an input csv and then save/apply model to a raster
# Inputs: input CSV, Raster Stack

### Train model
python randomforest_andrew.py -dir results -t /att/nobackup/aweis/forMaggie/cloud_training.csv
python rasterRF.py -w results -m newmodel -b 1 2 3 4 5 6 7 8 9 10 11 -bn 'Coastal Blue' Blue Green Yellow Red Red Edge Near-IR1 Near-IR2 DVI FDI SI -c 'results/TrainingData/cloud_training.csv'

### predict cloud
python randomforest_andrew_v2.py -dir results -a results/Models/model_5Images_0002.pkl -i /att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/8-band/WV02_20140716_M1BS_103001003328DB00-toa.tif

python rasterRF.py -w results -m results/Models/model_20_log2.pkl -b 1 2 3 4 5 6 7 8 9 10 11 -bn 'Coastal Blue' Blue Green Yellow Red Red Edge Near-IR1 Near-IR2 DVI FDI SI


### Test images
/att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/RandomForests/VHR-Stacks/WV02_20140716_M1BS_103001003328DB00-toa_pansharpen.tif - 21G
/att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/8-band - couple of images - small images
/att/nobackup/jacaraba/deep-rsensing-data/data/Vietnam_2019228_data_pansharp.tif - 98GB image
/att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/8-band/WV02_20140716_M1BS_103001003328DB00-toa.tif - 1.2GB

## Formal testing
python rasterRF.py -w results -m results/Models/model_20_log2.pkl -b 1 2 3 4 5 6 7 8 9 10 11 -i /att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/RandomForests/VHR-Stacks/WV02_20140716_M1BS_103001003328DB00-toa_pansharpen.tif

window 1000x1000 
    CPU - About 45 minutes to finish.
    GPU - About 11 minutes to finish.
window 5000x5000
    GPU - About 8 minutes to finish.


## simple testing
training
python rasterRF.py -w results -c /att/nobackup/aweis/forMaggie/cloud_training.csv -b 1 2 3 4 5 6 7 8 9 10 11

prediction
python rasterRF.py -w results -m results/Models/model_20_log2.pkl -b 1 2 3 4 5 6 7 8 9 10 11 -i /att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/RandomForests/VHR-Stacks/WV02_20140716_M1BS_103001003328DB00-toa_pansharpen.tif



(vhr-cloudmask) [jacaraba@gpu001 random_forest]$ python rasterRF.py -w results -m results/Models/model_20_log2.pkl -b 1 2 3 4 5 6 7 8 9 10 11 -i /att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/8-band/WV02_20140716_M1BS_103001003328DB00-toa.tif
Initializing Random Forest script with the following parameters
Working Directory: results
n_trees:           20
max features:      log2
Command used:  ['rasterRF.py', '-w', 'results', '-m', 'results/Models/model_20_log2.pkl', '-b', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '-i', '/att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/8-band/WV02_20140716_M1BS_103001003328DB00-toa.tif']
Loaded model results/Models/model_20_log2.pkl into cuda:0.
<xarray.DataArray (band: 11, y: 9831, x: 10374)>
dask.array<concatenate, shape=(11, 9831, 10374), dtype=int16, chunksize=(1, 2048, 2048), chunktype=numpy.ndarray>
Coordinates:
  * x        (x) float64 5.48e+05 5.48e+05 5.48e+05 ... 5.687e+05 5.687e+05
  * y        (y) float64 1.197e+06 1.197e+06 1.197e+06 ... 1.177e+06 1.177e+06
  * band     (band) int64 1 2 3 4 5 6 7 8 9 10 11
Attributes:
    transform:      (2.0, 0.0, 547977.0, 0.0, -2.0, 1196515.0)
    crs:            +init=epsg:32648
    res:            (2.0, 2.0)
    is_tiled:       0
    nodatavals:     (-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9...
    scales:         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    offsets:        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    AREA_OR_POINT:  Area
Final prediction initial shape:  (9831, 10374)
{'driver': 'GTiff', 'dtype': 'int16', 'nodata': -9999.0, 'width': 10374, 'height': 9831, 'count': 8, 'crs': CRS.from_epsg(32648), 'transform': Affine(2.0, 0.0, 547977.0,
       0.0, -2.0, 1196515.0), 'tiled': False, 'compress': 'lzw', 'interleave': 'pixel'}
Elapsed Time:  0.5118443250656128

(vhr-cloudmask) jacaraba@crane101:/att/gpfsfs/briskfs01/ppl/jacaraba/vhr-cloudmask/random_forest$ python rasterRF.py -w results -m results/Models/model_20_log2.pkl -b 1 2 3 4 5 6 7 8 9 10 11 -i /att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/8-band/WV02_20140716_M1BS_103001003328DB00-toa.tif
Initializing Random Forest script with the following parameters
Working Directory: results
n_trees:           20
max features:      log2
Command used:  ['rasterRF.py', '-w', 'results', '-m', 'results/Models/model_20_log2.pkl', '-b', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '-i', '/att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/8-band/WV02_20140716_M1BS_103001003328DB00-toa.tif']
Loaded model results/Models/model_20_log2.pkl into cpu.
<xarray.DataArray (band: 11, y: 9831, x: 10374)>
dask.array<concatenate, shape=(11, 9831, 10374), dtype=int16, chunksize=(1, 2048, 2048), chunktype=numpy.ndarray>
Coordinates:
  * y        (y) float64 1.197e+06 1.197e+06 1.197e+06 ... 1.177e+06 1.177e+06
  * x        (x) float64 5.48e+05 5.48e+05 5.48e+05 ... 5.687e+05 5.687e+05
  * band     (band) int64 1 2 3 4 5 6 7 8 9 10 11
Attributes:
    transform:      (2.0, 0.0, 547977.0, 0.0, -2.0, 1196515.0)
    crs:            +init=epsg:32648
    res:            (2.0, 2.0)
    is_tiled:       0
    nodatavals:     (-9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9999.0, -9...
    scales:         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    offsets:        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    AREA_OR_POINT:  Area
Final prediction initial shape:  (9831, 10374)
{'driver': 'GTiff', 'dtype': 'int16', 'nodata': -9999.0, 'width': 10374, 'height': 9831, 'count': 8, 'crs': CRS.from_epsg(32648), 'transform': Affine(2.0, 0.0, 547977.0,
       0.0, -2.0, 1196515.0), 'tiled': False, 'compress': 'lzw', 'interleave': 'pixel'}
Elapsed Time:  2.4667730728785195
