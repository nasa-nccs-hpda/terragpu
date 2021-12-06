# Additional changes to include Earth science libraries
# TerraGPU Container Environment

# Arguments to pass to the image
ARG CUDA_VER=11.2
ARG TORCH_VER=
ARG RAPIDS_VER=21.10
ARG LINUX_VER=ubuntu20.04
ARG FROM_IMAGE=nvcr.io/nvidia/rapidsai/rapidsai

# Import RAPIDS container as the BASE Image (cuda base image)
FROM ${FROM_IMAGE}:${RAPIDS_VER}-cuda${CUDA_VER}-base-${LINUX_VER}

# Adding ENV variables
ENV RAPIDS_DIR=/rapids

# Ubuntu needs noninteractive to be forced
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV DEBIAN_FRONTEND noninteractive

# Install Jax with CUDA support.
RUN apt-get -y update \
    && DEBIAN_FRONTEND=noninteractive apt-get -y install build-essential libssl-dev cmake \
        autoconf automake \
    && source activate rapids \
    && pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html \
    && pip install pytorch-toolbelt tifffile rasterio opencv-python webcolors Pillow seaborn Sphinx sphinx-rtd-theme \
        rioxarray xgboost pytorch-lightning lightning-bolts omegaconf jsonargparse[signatures] \
        geopandas pytest \
    # git clone https://github.com/jordancaraballo/senegal_lcluc.git /app && \
    # python /app/setup.py install
    # pip install from URL
    && pip cache purge \
    && apt-get -y autoremove \
    && rm -rf /var/cache/apt

HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
