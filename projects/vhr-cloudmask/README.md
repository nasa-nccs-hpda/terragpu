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
  1. [Login to the GPU cluster](#Login to the GPU cluster)  
  2. [Installation](#Installation)  
    * [Configuring anaconda](#Configuring anaconda)  
    * [Installing anaconda environment](#Installing anaconda environment)  
    * [Common errors when installing anaconda environment](#Common errors when installing anaconda environment)  
  3. [Training](#Training)  
    * [Training Background](#Training Background)  
    * [Training Data](#Training Data)  
    * [Training a Model](#Training a Model)  
    * [Trained Models](#Trained Models)  
  4. [Classification](#Classification)  
    * [Classification Background](#Classification Background)  
    * [Classification of Rasters](#Classification of Rasters)  
  4. [Performance Statistics](#Performance Statistics)  
  5. [Things to test at some point](#Things to test at some point)  


<!--te-->

### 1. Login to the GPU cluster
```
ssh username@adaptlogin.nccs.nasa.gov
ssh username@gpulogin1
```

### 2. Installation

#### Configuring anaconda
You will only need to do this the first time you login, or if you want to create a new environment. 
The following steps let you configure a symbolic link to your $NOBACKUP space since your $HOME 
quota is limited. Replace username with your auid.
```
module load anaconda
mkdir /att/nobackup/username/.conda; ln -s /att/nobackup/username/.conda /home/username/.conda;
chmod 755 /att/nobackup/username/.conda
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

#### Common errors when installing anaconda environment
1. Permission denied when running pip command
```
chmod 775 /att/gpfsfs/briskfs01/ppl/jacaraba/.conda/envs/xrasterlib/bin/pip
```
2. Permission denied when installing a particular package (example bokeh)
```
find /home/username/.conda/envs/ -type d -exec chmod 755 {} \;
```

### 3. Training

#### Background

The CSV file generated for training has the format of rows x bands, which implies that there
is a column per band, and each row represents a point in the raster. The last column of each
training CSV file includes a binary mask for determining if the point is cloud or cloud free. 
```
Example:
2306,2086,2005,1914,1916,2273,2959,2462,1043,-1897,-2147483648,1
2310,2097,2002,1921,1936,2288,2953,2484,1017,-1901,-2147483648,0
```
Wooten's team has concluded that the use of 3 additional indices improves the classification of 
cloud pixels. The indices calculated at his time are FDI, SI, and NDWI. The order of the bands that 
are being studied in this project depend on the number of bands included in the rasters. 
```
8 band imagery - ['CoastalBlue', 'Blue', 'Green', 'Yellow', 'Red', 'RedEdge','NIR1', 'NIR2']
4 band imagery - ['Blue', 'Green', 'Red', 'NIR1'] or ['Red', 'Green', 'Blue', 'NIR1']
```

#### Training Data

A couple of files have been located in ADAPT to ease finding data for training. The files with
their description are listed below, the path is /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/training.
This data might be moved to /att/pubrepo/ILAB/projects/Vietnam/cloudmask_data at some point.

| Filename                                  | Description     | 
| :---------------------------------------- |:----------------|
| cloud_training.csv                        | training data using all 8 bands from imagery and the 3 indices calculated using all of the bands (FDI, SI, DVI). | 
| cloud_training_8band_fdi_si_ndwi.csv      | training data using all 8 bands from imagery and the 3 indices calculated using all of the bands.  | 
| cloud_training_4band_fdi_si_ndwi.csv      | training data using only 4 bands from imagery and the 3 indices calculated using only 4 bands. The order of the bands goes accordingly to 8 band imagery (B-G-R-NIR).  |
| cloud_training_4band_rgb_fdi_si_ndwi.csv  | training data using only 4 bands from imagery and the 3 indices calculated using only 4 bands. The order of the bands was fixed to match (R-G-B-NIR).  |

#### Training a Model

Given a training CSV file, a new model can be generated with the following command.
```
cd projects/vhr-cloudmask
python rfdriver.py -o results -c data/cloud_training.csv
```
where -o is the output directory to store the model, and -c is the training csv. If 
you which to generate a log file with all of the output of the model, you should add 
the -l option to the command line. If you wish to specify a particular name for the model
you may train the model with the following command:
```
python rfdriver.py -o results -c data/cloud_training.csv -m newmodel.pkl
```
To get all of the available options you can execute the command below.
```
python rfdriver.py -h
```
The following commands were used to train the 4 models available for this project.
```
salloc # from gpulogin1, gets a session in a GPU system
python rfdriver.py -o /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/models -c /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/training/cloud_training.csv -l
python rfdriver.py -o /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/models -c /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/training/cloud_training_8band_fdi_si_ndwi.csv -l -m model_20_log2_8band_fdi_si_ndwi.pkl
python rfdriver.py -o /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/models -c /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/training/cloud_training_4band_fdi_si_ndwi.csv -l -m model_20_log2_4band_fdi_si_ndwi.pkl
python rfdriver.py -o /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/models -c /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/training/cloud_training_4band_rgb_fdi_si_ndwi.csv -l -m model_20_log2_4band_rgb_fdi_si_ndwi.pkl
```

#### Trained Models


| Training Data                             | Trained Model                                                                                    | 
| :---------------------------------------- |:-------------------------------------------------------------------------------------------------|
| cloud_training.csv                        | /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/models/model_20_log2.pkl                       | 
| cloud_training_8band_fdi_si_ndwi.csv      | /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/models/model_20_log2_8band_fdi_si_ndwi.pkl     | 
| cloud_training_4band_fdi_si_ndwi.csv      | /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/models/model_20_log2_4band_fdi_si_ndwi.pkl     |
| cloud_training_4band_rgb_fdi_si_ndwi.csv  | /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/models/model_20_log2_4band_rgb_fdi_si_ndwi.pkl |

### 4. Classification

#### Classification Background

Once the models have been trained, is time to classify the rasters. The imagery for this work is located in the following paths. For 8 band imagery
we can use 8 band models or 4 band models after dropping the unnecessary bands. In this case, we will use the 8 band model for 8 band imagery,
and 4 band model for 4 band imagery. This can be discussed further in a later discussion.
```
Location of Vietnam MS data:
8-band MS @ 2 m resolution: /att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/8-band
8-band MS pansharpened to 0.5 m resolution: /att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/8-band/pansharpen
4-band MS @ 2 m resolution: /att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS
4-band MS pansharpened to 0.5 m resolution: /att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/pansharpen
```

#### Classification of Rasters

Given a model, classification can be performed with the following command.
```
python rfdriver.py -o results -m model.pkl -i data/WV02_20140716_M1BS_103001003328DB00-toa.tif  # predict single raster
python rfdriver.py -o results -m model.pkl -i data/WV02_*.tif  # uses wildcards to predict all rasters matching pattern
```
The driver by default assumes that the order and number of bands included in the raster is 
['CoastalBlue', 'Blue', 'Green', 'Yellow', 'Red', 'RedEdge', 'NIR1', 'NIR2']. That said, 8-band imagery can 
be classified with the following command.

8-band 2m: bands ['CoastalBlue', 'Blue', 'Green', 'Yellow', 'Red', 'RedEdge', 'NIR1', 'NIR2']
```
salloc # from gpulogin1, gets a session in a GPU system
python rfdriver.py -l -o /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/classified/classified_8band2m -m /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/models/model_20_log2_8band_fdi_si_ndwi.pkl -i /att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/8-band/*.tif --sieve
```
8-band 0.5m: bands ['CoastalBlue', 'Blue', 'Green', 'Yellow', 'Red', 'RedEdge', 'NIR1', 'NIR2']
```
salloc # from gpulogin1, gets a session in a GPU system
python rfdriver.py -l -o /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/classified/classified_8band05m -m /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/models/model_20_log2_8band_fdi_si_ndwi.pkl -i /att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/8-band/pansharpen/*.tif --sieve
```
Since 4-band imagery has a different set of bands, you may specify the available bands using the -b option. 
Below are some examples for classifying 4-band imagery for this project. We split some of the patterns into two 
calls to speed up classification by using more resources and systems.

4-band 2m: bands ['Blue', 'Green', 'Red', 'NIR1'] 
```
# session #1
salloc # from gpulogin1, gets a session in a GPU system
python rfdriver.py -l -o /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/classified/classified_4band2m -m /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/models/model_20_log2_4band_fdi_si_ndwi.pkl -i /att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/WV02*.tif -b Blue Green Red NIR1 --sieve
```
```
# session #2
salloc # from gpulogin1, gets a session in a GPU system
python rfdriver.py -l -o /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/classified/classified_4band2m -m /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/models/model_20_log2_4band_fdi_si_ndwi.pkl -i /att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/WV03*.tif -b Blue Green Red NIR1 --sieve
```
4-band 0.5m: bands ['Blue', 'Green', 'Red', 'NIR1'] 
```
# session #1
salloc # from gpulogin1, gets a session in a GPU system
python rfdriver.py -l -o /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/classified/classified_4band2m -m /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/models/model_20_log2_4band_fdi_si_ndwi.pkl -i /att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/pansharpen/WV02*.tif -b Blue Green Red NIR1 --sieve
```
```
# session #2
salloc # from gpulogin1, gets a session in a GPU system
python rfdriver.py -l -o /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/classified/classified_4band2m -m /att/gpfsfs/briskfs01/ppl/jacaraba/cloudmask_data/models/model_20_log2_4band_fdi_si_ndwi.pkl -i /att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/pansharpen/WV03*.tif -b Blue Green Red NIR1 --sieve
```
Note: All of these predictions have been performed using salloc. A script to submit slurm allocations is included
but not recommended at this time. For some reason there is something odd in the way the ADAPT GPU cluster is 
allocating GPU memory, so only small windows can be used. Thus it is recommended to use direct salloc to allocate
your resources and use a window size of up to 5000 x 5000. This issue will be reported to NCCS User Services
so they can look into it. I always recommend using screen or tmux sessions so if something happens with your network
connection, your work will keep running at the NCCS.

### Performance Statistics
rasterRF.py has both CPU and GPU options to perform inference. Some performance statistics have been included below based
on our current use cases. GPU system running one V100 GPU, while CPU system running 24 cores. Memory consumption will 
depend greatly on the window size. GPU provides exponential speed ups for inferences. A window size of 6000 x 6000
exceeds GPU memory of 32GBs.

Raster Size: 1.2GB, 8 bands; dimensions: y: 9831, x: 10374

| Window Size (px) | RAM Usage  | CPU (elap time) | GPU (elap time) |
| :--------------: |:----------:| :-------------: | :-------------: |
| Full Raster      | 40 GB     | ~2.55 min       | ~0.72 min       |
| 1000 x 1000      | 4 GB      | ~2.40 min       | ~0.64 min       |
| 5000 x 5000      | 16 GB     | ~2.29 min       | ~0.49 min       |

Raster Size: 21GB, 8 bands; dimensions: y: 47751, x: 39324

| Window Size (px) | RAM Usage  | CPU (elap time) | GPU (elap time) |
| :--------------: |:----------:| :-------------: | :-------------: |
| Full Raster      | 200 GB    | Out of RAM      | Out of RAM      |
| 1000 x 1000      | 4 GB      | ~45.00 min      | ~11.00 min      |
| 5000 x 5000      | 16 6GB    | ~40.00 min      | ~8.00 min       |

### Things to test at some point

Below are a couple of things that should be tested and implemented at some point.
1. Apply median filter after cloud mask, and then apply sieve filter: this is supposed to smooth cloud mask borders.
2. Make sure we can the correct sieve filter size. Currently using 800.
3. Adding more parallelization to the indices addition function (calculate indices in parallel, might save some seconds).
4. Adding more parallelization to the prediction process (make it multi-gpu, parallel windows predicted and returned).
5. Add a module for pandas data fram indices calculation. This will speed up the generation of training.

### Authors

* Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
* Margaret Wooten, margaret.wooten@nasa.gov

### Contributors

* Andrew Weis, aweis1998@icloud.com
* Brian Lee, brianlee52@bren.ucsb.edu

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
