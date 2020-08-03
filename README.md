# vhr-cloudmask

Random Forest models for cloud/shadow masking of very high resolution images.

### Objectives

- Random forest script for the pixel-wise classification of very high-resolution rasters.
- Application of random forest script to identify clouds.
- Geometry based approach to identify clouds.

### Prerequisites

#### Conda
```
conda env create -f environment.yml
```

#### Pip            
```
pip install -r requirements.txt
```

## Authors

* Margaret Wooten
* Jordan Alexis Caraballo-Vega

## References

[1] TBD.

#!/usr/bin/env bash

sudo add-apt-repository ppa:ubuntugis/ppa && sudo apt-get update
sudo apt-get update
sudo apt-get install gdal-bin
sudo apt-get install libgdal-dev
export CPLUS_INCLUDE_PATH=/usr/include/gdal
export C_INCLUDE_PATH=/usr/include/gdal
pip install GDAL
