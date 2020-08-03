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
from datetime import datetime   # tracking date
import numpy as np              # for arrays modifications
import pandas as pd             # csv data frame modifications
import xarray as xr             # read rasters

from sklearn.model_selection import train_test_split # train/test data split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from hummingbird.ml import convert # support GPU inference

# Fix seed reproducibility.
seed = 21
np.random.seed = seed


def train_model(x, y, modelDir, n_trees, max_feat):

    labels = np.unique(y) # now it's the unique values in y array from text file
    print(f'The training data include {labels.size} classes.')
    print(f'Our X matrix is sized: {x.shape}') # shape of x data
    print(f'Our y array is sized:  {y.shape}') # shape of y data

    print ('Initializing model...')
    if '.' not in labels[0]: # if labels are integers, check first value from y (come as string)
        rf = RandomForestClassifier(n_estimators=n_trees, max_features=max_feat, oob_score=True) 
    else: # if labels are floats, use random forest regressor
        rf = RandomForestRegressor(n_estimators=n_trees, max_features=max_feat, oob_score=True)

    print("Training model...")
    rf.fit(x, y) # fit model to training data
    print('Score:', rf.oob_score_)

    try: # export model to file
        model_save = os.path.join(modelDir, f'model_{n_trees}_{max_feat}.pkl')
        joblib.dump(rf, model_save)
    except Exception as e:
        print(f'Error: {e}')

    return model_save # Return output model for application and validation


def get_test_training_sets(traincsv, testsize=0.30):

    df = pd.read_csv(traincsv, header=None, sep=',') # generate pd dataframe
    data = df.values # get values, TODO: maybe remove this line and add it on top
    x = data.T[0:-1].T.astype(str)
    y = data.T[-1].astype(str)

    # return x_train, x_test, y_train, y_test
    return train_test_split(x, y, test_size=testsize, random_state=seed)


def create_logfile(args, logdir='results'):
    logfile = os.path.join(logdir, '{}_log_{}trees_{}.txt'.format(
        datetime.now().strftime("%Y%m%d-%H%M%S"), args.n_trees, args.max_feat))
    print('See ', logfile)
    so = se = open(logfile, 'w')  # open our log file
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')  # re-open stdout without buffering
    os.dup2(so.fileno(), sys.stdout.fileno())  # redirect stdout and stderr to the log file opened above
    os.dup2(se.fileno(), sys.stderr.fileno())
    return logfile


def create_directories(work_dir='results'):
    """
    :param work_dir: working directory, comes from argparser option -w.
    :return: dictionary with keys and real paths to working subdirs.
    """
    dir_dictionary = dict()
    for directory in ['TrainingData', 'Models', 'Results', 'Logs']:
        dir_dictionary[directory] = os.path.join(work_dir, directory)
        os.system(f'mkdir -p {dir_dictionary[directory]}')
    return dir_dictionary


def getparser():
    """
    :return: argparser object with CLI commands.
    """
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("-w", "--work-directory", type=str, required=True, dest='workdir',
                        default="", help="Specify working directory")
    parser.add_argument("-m", "--model", type=str, required=True, dest='model',
                        default="", help="Specify model filename that will be saved or evaluated")
    parser.add_argument('-b', '--bands', nargs='*', dest='bands', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        help='Specify number of bands.', required=True, type=int)
    parser.add_argument('-bn', '--band-names', nargs='*', dest='band_names', help='Specify number of bands.',
                        required=True, type=str, default=['Coastal Blue', 'Blue', 'Green', 'Yellow', 'Red', 'Red Edge',
                        'Near-IR1', 'Near-IR2', 'DVI', 'FDI', 'SI'])
    # Train
    parser.add_argument("-c", "--csv", type=str, required=False, dest='traincsv',
                        default="", help="Specify CSV file to train the model.")
    parser.add_argument("-t", "--n-trees", type=str, required=False, dest='n_trees',
                        default=20, help="Specify number of trees for random forest model.")
    parser.add_argument("-f", "--max-features", type=str, required=False, dest='max_feat',
                        default='log2', help="Specify random forest max features.")
    parser.add_argument("-ts", "--test-size", type=float, required=False, dest='testsize',
                        default='0.30', help="Size of test data (e.g: .30)")
    # Evaluate
    parser.add_argument("-i", "--rasters", type=str, nargs='*', required=False, dest='rasters',
                        default=['*.tif'], help="Image or pattern to evaluate images.")

    return parser.parse_args()


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
def main():

    start_time = time()  # record start time
    args = getparser()  # initialize arguments parser

    print('Initializing script with the following parameters')
    print(f'Working Directory: {args.workdir}')
    print(f'n_trees:           {args.n_trees}')
    print(f'max features:      {args.max_feat}')

    # 1. set working and results directories
    dir_dict = create_directories(args.workdir)

    # 2. set log file for script - enable after developing
    logfile = create_logfile(args, logdir=dir_dict['Logs'])
    print ("Command used: ", sys.argv) # saving command into log file

    # 3a. if does not exist, proceed and train
    if os.path.isfile(args.traincsv):

        # 3a1. read CSV training data and split into train and test sets
        print(f'Input train CSV: {args.traincsv}')
        x_train, x_test, y_train, y_test = get_test_training_sets(args.traincsv, args.testsize)
        print(f'y_train lenght: {len(y_train)} and y_test lenght: {len(y_test)}')

        # 3a2. train and save the model
        print(f'Building model with n_trees={args.n_trees} and max_feat={args.max_feat}...')
        model_save = train_model(x_train, y_train, dir_dict['Models'], args.n_trees, args.max_feat)
        print(f'Model has been saved as {model_save}')

    # 3b. evaluate images from model
    elif os.path.isfile(args.model):
        print ("Performing inference to given images.")
        apply_model(args.rasters, dir_dict['Classified'], args.model)
        ### TBD: add apply model, gpu support, parallelization, xarray rasterio

    else:
        sys.exit("ERROR: You should specify a train csv or a model to load. Refer to python " +
                 "rasterRF.py -h for more options.")

    print("Elapsed Time: ", (time() - start_time) / 60.0)  # output program run time


if __name__ == "__main__":

    main()

    # simple test:
    # python rasterRF.py -w results -m test -b 1 2 3 -bn red green blue
