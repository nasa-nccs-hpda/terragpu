"""
Created: 01/29/2016, Refactored: 07/20/2020

Purpose: Build, save and apply random forest model for the pixel classification 
         of raster data. Usage requirements are referenced in README.
         
Data Source: This script has been tested with very high-resolution NGA data.
             Additional testing will be required to validate the applicability 
             of this model for other datasets.

Original Author: Margaret Wooten, SCIENTIFIC PROGRAMMER/ANALYST, Code 610
Refactored: Jordan A Caraballo-Vega, Science Data Processing Branch, Code 587
"""
#--------------------------------------------------------------------------------
# Import System Libraries
#--------------------------------------------------------------------------------
import sys, os, glob, argparse  # system modifications
import joblib                   # joblib for parallel jobs
from time import time           # tracking time
from datetime import datetime   # datetime library
import numpy as np              # for arrays modifications
import pandas as pd             # csv data frame modifications
import xarray as xr             # read rasters
import skimage.io as io         # managing images
import matplotlib.pyplot as plt # visualizations

from sklearn.model_selection import train_test_split # train/test data split
from sklearn.ensemble import RandomForestClassifier  # random forest classifier
from hummingbird.ml import convert                   # support GPU training

# Fix seed reproducibility.
seed = 42
np.random.seed = seed

