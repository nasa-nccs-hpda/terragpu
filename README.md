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
Adding description of arguments here.

Simple training command. 
```
python rasterRF.py -w results -c /att/nobackup/aweis/forMaggie/cloud_training.csv -b 1 2 3 4 5 6 7 8 9 10 11
```

### Inference
Adding description of arguments here.

Simple prediction command.
```
python rasterRF.py -w results -m results/Models/model_20_log2.pkl -b 1 2 3 4 5 6 7 8 9 10 11 -i /att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/RandomForests/VHR-Stacks/WV02_20140716_M1BS_103001003328DB00-toa_pansharpen.tif
```

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


