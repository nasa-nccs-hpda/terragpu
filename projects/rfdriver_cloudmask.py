"""
Purpose: Build, save and apply random forest model for the pixel classification
         of raster data. Usage requirements are referenced in README.

Data Source: This script has been tested with very high-resolution NGA data.
             Additional testing will be required to validate the applicability
             of this model for other datasets.

Original Author: Margaret Wooten, SCIENTIFIC PROGRAMMER/ANALYST, Code 610
Refactored: Jordan A Caraballo-Vega, Science Data Processing Branch, Code 587
"""
# --------------------------------------------------------------------------------
# Import System Libraries
# --------------------------------------------------------------------------------
from datetime import datetime  # tracking date
from time import time  # tracking time
import sys, os, argparse  # system libraries
import numpy as np  # for arrays modifications

# Fix seed for reproducibility.
seed = 21
np.random.seed(seed)

# Add src path to system PATH
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/src')

import indices
from RandomForest import RandomForest
# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------
def create_logfile(args, logdir='results'):
    """
    :param args: argparser object
    :param logdir: log directory to store log file
    :return: logfile instance, stdour and stderr being logged to file
    """
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
    parser.add_argument("-m", "--model", type=str, required=False, dest='model',
                        default=None, help="Specify model filename that will be saved or evaluated")
    parser.add_argument('-b', '--bands', nargs='*', dest='bands', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        help='Specify number of bands.', required=True, type=int)
    parser.add_argument('-bn', '--band-names', nargs='*', dest='band_names', help='Specify number of bands.',
                        required=False, type=str, default=['Coastal Blue', 'Blue', 'Green', 'Yellow', 'Red', 'Red Edge',
                                                           'Near-IR1', 'Near-IR2', 'DVI', 'FDI', 'SI'])
    # Train
    parser.add_argument("-c", "--csv", type=str, required=False, dest='traincsv',
                        default=None, help="Specify CSV file to train the model.")
    parser.add_argument("-t", "--n-trees", type=int, required=False, dest='n_trees',
                        default=20, help="Specify number of trees for random forest model.")
    parser.add_argument("-f", "--max-features", type=str, required=False, dest='max_feat',
                        default='log2', help="Specify random forest max features.")
    parser.add_argument("-ts", "--test-size", type=float, required=False, dest='testsize',
                        default='0.30', help="Size of test data (e.g: .30)")
    # Evaluate
    parser.add_argument("-i", "--rasters", type=str, nargs='*', required=False, dest='rasters',
                        default='*.tif', help="Image or pattern to evaluate images.")
    parser.add_argument("-ws", "--window-size", nargs=2, type=int, required=False, dest='windowsize',
                        default=[5000, 5000], help="Specify window size to perform sliding predictions.")
    return parser.parse_args()


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
def main():

    start_time = time()  # record start time
    args = getparser()  # initialize arguments parser

    print('Initializing Random Forest script with the following parameters')
    print(f'Working Directory: {args.workdir}')
    print(f'n_trees:           {args.n_trees}')
    print(f'max features:      {args.max_feat}')

    # 1. set working, logs, and results directories, return dictionary
    # this is not necessary, just doing it to keep things organized
    dir_dict = create_directories(args.workdir)

    # 2. set log file for script - you may disable this when developing
    #logfile = create_logfile(args, logdir=dir_dict['Logs'])
    #print("Command used: ", sys.argv)  # saving command into log file

    # 3. Instantiate RandomForest object
    rfobj = RandomForest(args.traincsv, args.model)

    # 3a. if training csv exists, proceed and train
    if rfobj.traincsvfile is not None:

        # 3a1. read CSV training data and split into train and test sets
        # returns four numpy arrays to train the model on
        rfobj.splitdata(testsize=args.testsize, seed=seed)
        print(f'y_train lenght: {len(rfobj.y_train)} and y_test lenght: {len(rfobj.y_test)}')

        # 3a2. train and save the model
        print(f'Building model with n_trees={rfobj.n_trees} and max_feat={rfobj.max_feat}...')
        rfobj.trainrf()
        print(f'Model has been saved as {rfobj.modelfile}')

    # 3b. evaluate images from model
    elif rfobj.modelfile is not None:

        # 3b1. load model - CPU or GPU bound
        rfobj.loadrf()

        if not args.rasters or args.rasters == '*.tif':  # if raster -i variable is empty, stop process and log.
            sys.exit("ERROR: No images to predict. Refer to python rfdriver.py -h for options.")

        # 3b3. apply model and get predictions
        for rast in args.rasters:  # iterate over each raster
            rfobj.readraster(rast)
            rfobj.addindices([indices.dvi, indices.fdi, indices.si], factor=1.0)

            rfobj.predictrf(rastfile=rast, ws=args.windowsize)
            rfobj.sieve(rfobj.prediction, rfobj.prediction, size=800, mask=None, connectivity=8)
            output_name = "{}/cm_{}".format(rfobj.resultsdir, rast.split('/')[-1])  # model name
            rfobj.toraster(rast, rfobj.prediction, output_name)

    # 3c. exit if csv or model are not present or given
    else:
        sys.exit("ERROR: You should specify a train csv or a model to load. Refer to python " +
                 "rfdriver.py -h for more options.")

    print("Elapsed Time: ", (time() - start_time) / 60.0)  # output program run time in minutes


if __name__ == "__main__":
    main()

    # python rfdriver_cloudmask.py -w results -c ../cloudmask/cloud_training.csv -b 1 2 3 4 5 6 7 8 9 10 11
    # python rfdriver_cloudmask.py -w results -m model_20_log2.pkl -b 1 2 3 4 5 6 7 8 9 10 11 -i /Users/jacaraba/Desktop/cloud-mask-data/WV02_20140716_M1BS_103001003328DB00-toa.tif
