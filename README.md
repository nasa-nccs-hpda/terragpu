# xrasterlib: Python library to process and classify remote sensing imagery

![nccs-logo](images/nccslogo.png | width=100)

### Objectives

- Library to process remote sensing imagery using memory efficient libraries.
- Machine Learning and Deep Learning image classification.
- CPU and GPU support.

### Installation

#### Build Conda Environment
```
conda env create -f requirements/environment.yml
```

#### Build PyEnv
```
pip install -r requirements/requirements.txt
```
Note: PIP installations do not include CUDA libraries for GPU support. Make sure
NVIDIA libraries are installed locally in the system.

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
└── LICENSE
```

## Background

Library to process and classify remote sensing imagery. This is work in progress currently supporting
Random Forest classification and merging Convolutional Neural Networks from the deep-rsensing project.
Each particular project includes its own README with information. When these projects are concluded,
they will be migrated to their own repository.

Raster processing relies in xarray and rasterio for memory mapping operations.

## Authors

* Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
* Margaret Wooten, margaret.wooten@nasa.gov

## Contributors

* Andrew Weis, aweis1998@icloud.com

## References

[1] Chollet, François; et all, Keras, (2015), GitHub repository, https://github.com/keras-team/keras. Accessed 13 February 2020.

[2] Paszke, Adam; Gross, Sam; Chintala, Soumith; Chanan, Gregory; et all, PyTorch, (2016), GitHub repository, https://github.com/pytorch/pytorch. Accessed 13 February 2020.

[3] Google Brain Team; et all, TensorFlow, (2015), GitHub repository, https://github.com/tensorflow/tensorflow. Accessed 13 February 2020.