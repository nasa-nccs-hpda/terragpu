# vhr-cloudmask

Cloud masking of very high-resolution imagery using Random Forest. 

## Objectives

- Generalized Random forest script for pixel-wise classification of very high-resolution rasters.
- Application of random forest script to generate cloud masks.

## Using this project at the NCCS

The following is a step by step guide to classify rasters using the [NASA Center for Climate Simulation (NCCS)](https://www.nccs.nasa.gov/) 
GPU cluster. Steps should be similar in any other working station or HPC cluster. This project assumes that an initial CSV with the
training data has been given. There is a script included that modifies the calculated indices if necessary. 

<!--ts-->
  * [Login to the GPU cluster](#Login to the GPU cluster)
  * [Installation](#Installation)
    * [Configuring anaconda](#Configuring anaconda)
    * [Installing anaconda environment](#Installing anaconda environment)
<!--te-->

### Login to the GPU cluster
```
ssh username@adaptlogin.nccs.nasa.gov
ssh username@gpulogin1
```

### Installation

#### Configuring anaconda
You will only need to do this the first time you login, or if you want to create a new environment. 
The following steps let you configure a symbolic link to your $NOBACKUP space since your $HOME 
quota is limited. Replace username with your auid.
```
ssh username@adaptlogin.nccs.nasa.gov
module load ml/anaconda3
mkdir /att/nobackup/username/.conda; ln -s /att/nobackup/username/.conda /home/username/.conda;
```
#### Installing anaconda environment
Now we will create an anaconda environment to execute the software included in this project.
This environment can also be included as a kernel when using Jupyter Notebooks.
```
cd $NOBACKUP
git clone https://github.com/jordancaraballo/xrasterlib.git
cd xrasterlib
conda create --name xrasterlib
conda activate xrasterlib
conda install -c anaconda pip cudatoolkit cudnn
pip install -r requirements/requirements.txt
python setup.py install
```

### Training

The CSV file generated for training has the format of n x bands, which implies that there
is a column per band, and each row represents a point in the raster.

Adding additional description of arguments here.

Simple training command. 
```
python rasterRF.py -w results -c cloud_training.csv -b 1 2 3 4 5 6 7 8 9 10 11

python rfdriver.py -o results -c data/cloud_training.csv
python rfdriver.py -o results -c data/cloud_training.csv -l
python rfdriver.py -o results -c data/cloud_training.csv -m newmodel.pkl
```
Prediction
Assuming default bands are ['Coastal Blue', 'Blue', 'Green', 'Yellow', 'Red', 'Red Edge',
                                                           'Near-IR1', 'Near-IR2']
                                                        
```
python rfdriver.py -o results -m newmodel.pkl -i /Users/jacaraba/Desktop/cloud-mask-data/WV02_*.tif
python rfdriver.py -o results -m newmodel.pkl -i /Users/jacaraba/Desktop/cloud-mask-data/WV02_20140716_M1BS_103001003328DB00-toa.tif  
```
else specify bands with 
add it here


        """    
            # support keelin squares - 4band model
            #rfobj.data = rfobj.data[:4, :, :]
            #rfobj.initbands = 4
            
            ## support 8 band rasters  - 4 band model
            #red = rfobj.data[4, :, :]
            #green = rfobj.data[2, :, :]
            #rfobj.data = rfobj.data.drop(dim="band", labels=[1, 3, 4, 5, 6, 8], drop=True)
            #rfobj.data = xr.concat([red, green, rfobj.data], dim="band")  # concat new band
            #rfobj.data = rfobj.data.transpose('band', 'y', 'x')
            #rfobj.initbands = 4

            print("Size of raster: ", raster_obj.data.shape, raster_obj.bands)
            print(raster_obj.data)
            
            #rfobj.addindices([indices.dvi, indices.fdi, indices.si], factor=1.0)
            #rfobj.addindices([indices.dvi, indices.fdi, indices.si, indices.cs1], factor=1.0)
            raster_obj.addindices([indices.dvi, indices.fdi, indices.si, indices.cs2], factor=1.0)
            #rfobj.addindices([indices.dvi, indices.fdi, indices.si, indices.cs1, indices.cs2], factor=1.0)

            print("Size of raster: ", rfobj.data.shape)

            raster_obj.predictrf(rastfile=rast, ws=args.windowsize)
            raster_obj.sieve(raster_obj.prediction, raster_obj.prediction, size=800, mask=None, connectivity=8)
            output_name = "{}/cm_{}".format(raster_obj.resultsdir, rast.split('/')[-1])  # model name

            #rfobj.prediction = ndimage.median_filter(rfobj.prediction, size=20)

            raster_obj.toraster(rast, raster_obj.prediction, output_name)
            """
            
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

#### Location of the Data
Location of Vietnam MS data:
4-band MS @ 2 m resolution: /att/gpfsfs/briskfs01/ppl/user/Vietnam_LCLUC/TOA/M1BS
8-band MS @ 2 m resolution: /att/gpfsfs/briskfs01/ppl/user/Vietnam_LCLUC/TOA/M1BS/8-band
4-band MS pansharpened to 0.5 m resolution: /att/gpfsfs/briskfs01/ppl/user/Vietnam_LCLUC/TOA/M1BS/pansharpen
8-band MS pansharpened to 0.5 m resolution: /att/gpfsfs/briskfs01/ppl/user/Vietnam_LCLUC/TOA/M1BS/8-band/pansharpen

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
