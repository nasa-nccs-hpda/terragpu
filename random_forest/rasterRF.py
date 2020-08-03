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
#import joblib                   # joblib for parallel jobs
from time import time           # tracking time
from datetime import datetime
import numpy as np              # for arrays modifications
import pandas as pd             # csv data frame modifications
import xarray as xr             # read rasters
#import skimage.io as io         # managing images

#from sklearn.model_selection import train_test_split # train/test data split
#from sklearn.ensemble import RandomForestClassifier  # random forest classifier
#from hummingbird.ml import convert                   # support GPU training

# Fix seed reproducibility.
seed = 21
np.random.seed = seed


def get_test_training_sets(traincsv, testsize=0.30):

    df = pandas.read_csv(traincsv, header=None, sep=',') # generate pd dataframe
    data = df.values # get values, TODO: maybe remove this line and add it on top
    x = data.T[0:-1].T.astype(str)
    y = data.T[-1].astype(str)

    # Now we have X and y, but this is not split into validation and training. Do that here:
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=testsize, random_state=seed)
    return x_train, x_test, y_train, y_test


def create_logfile(logdir, args):
    logfile = os.path.join(logdir, '{}_log_{}trees_{}.txt'.format(
        datetime.now().strftime("%Y%m%d-%H%M%S"), args.n_trees, args.max_feat))
    print('See ', logfile)
    so = se = open(logfile, 'w')  # open our log file
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')  # re-open stdout without buffering
    os.dup2(so.fileno(), sys.stdout.fileno())  # redirect stdout and stderr to the log file opened above
    os.dup2(se.fileno(), sys.stderr.fileno())
    return logfile


def create_directories(work_dir):
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
                        default='log2', help="Size of test data (e.g: .30)")
    # Evaluate

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
    #logfile = create_logfile(dir_dict['Logs'], args)
    print ("Command used: ", sys.argv) # saving command into log file

    # 3. if does not exist, proceed and train
    if os.path.isfile(args.traincsv):

        # 3a. read CSV training data and split into train and test sets
        print(f'Input train CSV: {args.traincsv}')
        (X_train, X_test, y_train, y_test) = get_test_training_sets(train_csv, args.testsize)
        print('Y_TRAIN LENGTH: {}\nY_TEST LENGTH: {}'.format(len(y_train), len(y_test)))

        # TRAIN MODEL:
        #print("Building model with n_trees={} and max_feat={}...".format(n_trees, max_feat))
        #model_save = train_model(X_train, y_train, dir_dict['Models'], n_trees, max_feat, catalogid)
        #print(model_save)

    elif os.path.isfile(args.model):
        print ("Performing inference.")

    else:
        sys.exit("ERROR: You should specify a train csv or a model to load. Refer to python " +
                 "rasterRF.py -h for more options.")

    print("Elapsed Time: ", (time() - start_time) / 60.0)  # output program run time


if __name__ == "__main__":

    main()

    # simple test:
    # python rasterRF.py -w results -m test -b 1 2 3 -bn red green blue