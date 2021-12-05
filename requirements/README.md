# Requirements

TerraGPU can be installe and used via anaconda environments and containers.
A Docker container is provided and this same container can be converted
to a Singularity container without loosing any functionalities.
A demo cpu based conda environment is provided with limited capabilities for
CPU-only users.

CPU support is limited and the author does not provide any guarantee of usability.
Limited functionalities have been adapted for CPU in order to provide demo support
or local testing before migrating to production environments testing.

## Architecture

The container is built on top of NGC NVIDIA RAPIDS containers. It provides a built-in
anaconda environment called rapids which includes all other dependencies for software
development of Earth Science applications on GPUS and AI/ML frameworks.

This application is powered by PyTorch and PyTorch Lighning AI/ML backends, including
CUML for the development of machine learning models support by NVIDIA RAPIDS.

## Example to Download the Container via Singularity

A Dockerhub public extension will be provided as soon as permissions
are granted from the software release process.

```bash
module load singularity
singularity build --sandbox terragpu docker://gitlab.nccs.nasa.gov:5050/nccs-lcluc/terragpu/terragpu:latest
```

## Example to Install Anaconda Environment

``` bash
git clone https://github.com/nasa-cisto-ai/terragpu.git
cd terragpu; conda env create -f requirements/environment_gpu.yml;
conda activate terragpu
python setup.py install
```

## Container Usage

As an example, you can shell into the container:

```bash
singularity shell terragpu
```

Then activate the Anaconda environment:

```
source active rapids
```

And start developing your Python applications:

```bash
python --version
```