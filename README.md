# xrasterlib

Python library to process and classify remote sensing imagery

<img src="images/nccslogo.png" height="150" width="300">

### Objectives

- Library to process remote sensing imagery using memory efficient libraries.
- Machine Learning and Deep Learning image classification.
- Provide GPU accelaration for the evaluation of sckit models.

### Installation

The following library is intended to be used to accelerate the development
of data science products for remote sensing satellite imagery. xrasterlib can 
be installed by itself, but instructions for installing the full environments 
are listed below so projects, examples, and notebooks can be run.

#### Build Conda Environment - Single Step (GPU support)
```
git clone https://github.com/jordancaraballo/xrasterlib.git
cd xrasterlib; conda env create -f requirements/environment.yml;
conda activate xrasterlib
python setup.py install
```

#### Build Conda Environment - Multi Step
```
git clone https://github.com/jordancaraballo/xrasterlib.git
cd xrasterlib; conda create --name xrasterlib;
conda activate xrasterlib
conda install -c anaconda pip 
conda install -c anaconda cudatoolkit=10.1 cudnn # if GPU available
pip install --upgrade -r requirements/requirements.txt
python setup.py install
```

#### Pip only
```
git clone https://github.com/jordancaraballo/xrasterlib.git
pip install -r requirements/requirements.txt
python setup.py install
```
Note: PIP installations do not include CUDA libraries for GPU support. Make sure
NVIDIA libraries are installed locally in the system if not using conda.

## Getting Started

```
├── archives              <- Legacy code stored to historical reference
├── docs                  <- Default documentation for working with this project
├── images                <- Store project images
├── notebooks             <- Jupyter notebooks
├── projects              <- Current projects utilizing the library
├── requirements          <- Requirements for installing the dependencies
├── xrasterlib            <- Library source code
├── README.md             <- The top-level README for developers using this project
├── CHANGELOG.md          <- Releases documentation
└── LICENSE
```

## Background

Library to process and classify remote sensing imagery. This is work in progress currently supporting
Random Forest classification and merging Convolutional Neural Networks from the deep-rsensing project.
Each particular project includes its own README with information. When these projects are concluded,
they will be migrated to their own repository.

Raster processing relies in xarray and rasterio for memory mapping operations.

## Short Term Enhancements

- [ ] Create and publish PIP package
- [X] Add additional indices
- [ ] Document Random Forest

## Ongoing Projects

- projects/cloudmask  - cloud masking of VHR imagery using Random Forest
- projects/shadowmask - shadow masking of VHR imagery using geometry properties.

## Authors

* Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
* Margaret Wooten, margaret.wooten@nasa.gov

## Contributors

* Andrew Weis, aweis1998@icloud.com

## References

[1] Chollet, François; et all, Keras, (2015), GitHub repository, https://github.com/keras-team/keras. Accessed 13 February 2020.

[2] Paszke, Adam; Gross, Sam; Chintala, Soumith; Chanan, Gregory; et all, PyTorch, (2016), GitHub repository, https://github.com/pytorch/pytorch. Accessed 13 February 2020.

[3] Google Brain Team; et all, TensorFlow, (2015), GitHub repository, https://github.com/tensorflow/tensorflow. Accessed 13 February 2020.