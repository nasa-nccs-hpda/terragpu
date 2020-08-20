# vhr-cloudmask

Generalized Random Forest models for processing very high-resolution rasters. 
Use case included in this repository aims to perform cloud/shadow masking.

### Objectives

- Generalized Random forest script for pixel-wise classification of very high-resolution rasters.
- Application of random forest script to generate cloud masks.
- Geometry based approach to generate cloud masks.

### Prerequisites

#### Conda
```
conda env create -f environment.yml
```

#### Pip            
```
pip install -r requirements.txt
```
Note: PIP installations do not include CUDA libraries for GPU support. Make sure 
NVIDIA libraries are installed locally in the system.

### Training

The CSV file generated for training has the format of n x bands, which implies that there
is a column per band, and each row represents a point in the raster.

Adding additional description of arguments here.

Simple training command. 
```
python rasterRF.py -w results -c cloud_training.csv -b 1 2 3 4 5 6 7 8 9 10 11
```

### Inference

Adding additional description of arguments here.

Simple prediction command.
```
python rasterRF.py -w results -m results/Models/model_20_log2.pkl -b 1 2 3 4 5 6 7 8 9 10 11 -i /somewhere/toa_pansharpen.tif
```

### Performance Statistics
rasterRF.py has both CPU and GPU options to perform inference. Some performance statistics have been included below based
on our current use cases. GPU system running one V100 GPU, while CPU system running 24 cores. Memory consumption will 
depend greatly on the window size. GPU provides exponential speed ups for inferences. A window size of 6000 x 6000
exceeds GPU memory of 32GBs.

Raster Size: 1.2GB, 8 bands; dimensions: y: 9831, x: 10374

| Window Size (px) | RAM Usage  | CPU (elap time) | GPU (elap time) |
| :--------------: |:----------:| :-------------: | :-------------: |
| Full Raster      | ~40 GB     | ~2.55 min       | ~0.72 min       |
| 1000 x 1000      | ~4 GB      | ~2.40 min       | ~0.64 min       |
| 5000 x 5000      | ~16 GB     | ~2.29 min       | ~0.49 min       |

Raster Size: 21GB, 8 bands; dimensions: y: 47751, x: 39324

| Window Size (px) | RAM Usage  | CPU (elap time) | GPU (elap time) |
| :--------------: |:----------:| :-------------: | :-------------: |
| Full Raster      | ~200 GB    | Out of RAM      | Out of RAM      |
| 1000 x 1000      | ~4 GB      | ~45.00 min      | ~11.00 min      |
| 5000 x 5000      | ~16 6GB    | ~40.00 min      | ~8.00 min       |


### Authors

* Margaret Wooten, margaret.wooten@nasa.gov
* Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
* Andrew Weis, aweis1998@icloud.com

### References

[1] Segal-Rozenhaimer, M., Li, A., Das, K., & Chirayath, V. (2020). Cloud detection algorithm for 
multi-modal satellite imagery using convolutional neural-networks (CNN). Remote Sensing of Environment, 237, 111446.

[2] Sun, L., Liu, X., Yang, Y., Chen, T., Wang, Q., & Zhou, X. (2018). A cloud shadow detection method combined with 
cloud height iteration and spectral analysis for Landsat 8 OLI data. ISPRS Journal of Photogrammetry and Remote Sensing, 138, 193-207.

### Additional Notes

#### rasterRF.py usage output
```
(vhr-cloudmask)$ python rasterRF.py -h
usage: rasterRF.py [-h] -w WORKDIR [-m MODEL] -b [BANDS [BANDS ...]] [-bn [BAND_NAMES [BAND_NAMES ...]]] [-c TRAINCSV] [-t N_TREES] [-f MAX_FEAT] [-ts TESTSIZE]
                   [-i [RASTERS [RASTERS ...]]] [-ws WINDOWSIZE WINDOWSIZE]

  -h, --help            show this help message and exit
  -w WORKDIR, --work-directory WORKDIR
                        Specify working directory
  -m MODEL, --model MODEL
                        Specify model filename that will be saved or evaluated
  -b [BANDS [BANDS ...]], --bands [BANDS [BANDS ...]]
                        Specify number of bands.
  -bn [BAND_NAMES [BAND_NAMES ...]], --band-names [BAND_NAMES [BAND_NAMES ...]]
                        Specify number of bands.
  -c TRAINCSV, --csv TRAINCSV
                        Specify CSV file to train the model.
  -t N_TREES, --n-trees N_TREES
                        Specify number of trees for random forest model.
  -f MAX_FEAT, --max-features MAX_FEAT
                        Specify random forest max features.
  -ts TESTSIZE, --test-size TESTSIZE
                        Size of test data (e.g: .30)
  -i [RASTERS [RASTERS ...]], --rasters [RASTERS [RASTERS ...]]
                        Image or pattern to evaluate images.
  -ws WINDOWSIZE WINDOWSIZE, --window-size WINDOWSIZE WINDOWSIZE
                        Specify window size to perform sliding predictions.
```
#### Installing GDAL on Ubuntu 18.04
This scriplet has been included for documentation purposes only. This project does not require
GDAL libraries in order to work.
```
#!/usr/bin/env bash
sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update
sudo apt-get update
sudo apt-get install gdal-bin
sudo apt-get install libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
pip install GDAL
```
