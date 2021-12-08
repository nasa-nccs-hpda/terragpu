# terragpu

Python library to process and classify remote sensing imagery by means of GPUs and AI/ML

[![DOI](https://zenodo.org/badge/295528915.svg)](https://zenodo.org/badge/latestdoi/295528915)
![Pipeline Status](https://github.com/nasa-cisto-ai/terragpu/actions/workflows/main.yml/badge.svg)


<img src="images/nccslogo.png" height="150" width="300">

## Objectives

- Library to process remote sensing imagery using memory efficient libraries.
- Machine Learning and Deep Learning image classification.
- Agnostic built-in GPU accelaration.
- Example notebooks for quick AI/ML start with your own data.

### Installation

The following library is intended to be used to accelerate the development of data science products for remote sensing satellite imagery. terragpu can be installed by itself, but instructions for installing the full environments are listed under the requirements directory so projects, examples, and notebooks can be run.

Note: PIP installations do not include CUDA libraries for GPU support. Make sure
NVIDIA libraries are installed locally in the system if not using conda.

## Getting Started

``` bash
├── archives              <- Legacy code stored to historical reference
├── docs                  <- Default documentation for working with this project
├── images                <- Store project images
├── notebooks             <- Jupyter notebooks
├── examples              <- Examples for utilizing the library
├── requirements          <- Requirements for installing the dependencies
├── scripts               <- Utility scripts for analysis
├── terragpu              <- Library source code
├── README.md             <- The top-level README for developers using this project
├── CHANGELOG.md          <- Releases documentation
├── LICENSE               <- License documentation
└── setup.py              <- Script to install library
```

## Background

Library to process and classify remote sensing imagery. This is work in progress currently supporting
Random Forest classification and merging Convolutional Neural Networks from the deep-rsensing project.
Each particular project includes its own README with information.

Raster processing relies in xarray and rasterio for memory mapping operations, using Dask as the backend.
PyTorch is implemented for GPU accelaration of Sckitlearn models. GPU acceleration is provided
by the NVIDIA RAPIDS environment and we are in the development phase to support AMD GPUs.

## Authors

- Jordan Alexis Caraballo-Vega, jordan.a.caraballo-vega@nasa.gov
- Margaret Wooten, margaret.wooten@nasa.gov

## Contributors

- Andrew Weis, aweis1998@icloud.com
- Brian Lee, brianlee52@bren.ucsb.edu

## Installation
See the build [guide](requirements/README.md).

## Contributing

Please see our [guide for contributing to terragpu](CONTRIBUTING.md).

## References

Tutorials will be published under [Medium](https://medium.com/@jordan.caraballo/) for additional support
and development, including how to use the library or any upcoming releases.

Please consider citing this when using terragpu in a project. You can use the citation BibTeX:

```bibtex
@software{
  A_Caraballo-Vega_TerraGPU_2020,
  author = {A Caraballo-Vega, Jordan},
  doi = {10.5281/zenodo.5765917},
  license = {Apache-2.0},
  month = {7},
  title = {{TerraGPU}},
  url = {TBD},
  version = {2021.11},
  year = {2020}
}
```

## References

[1] Raschka, S., Patterson, J., & Nolet, C. (2020). Machine learning in python: Main developments and technology trends in data science, machine learning, and artificial intelligence. Information, 11(4), 193.

[2] Paszke, Adam; Gross, Sam; Chintala, Soumith; Chanan, Gregory; et all, PyTorch, (2016), GitHub repository, <https://github.com/pytorch/pytorch>. Accessed 13 February 2020.

[3] Caraballo-Vega, J., Carroll, M., Li, J., & Duffy, D. (2021, December). Towards Scalable & GPU Accelerated Earth Science Imagery Processing: An AI/ML Case Study. In AGU Fall Meeting 2021. AGU.
