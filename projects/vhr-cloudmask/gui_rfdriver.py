"""
Purpose: Build, save and apply random forest model for the pixel classification
         of raster data. Usage requirements are referenced in README.

Data Source: This script has been tested with very high-resolution NGA data.
             Additional testing will be required to validate the applicability
             of this model for other datasets.

Original Author: Margaret Wooten, SCIENTIFIC PROGRAMMER/ANALYST, Code 610
Refactored: Jordan A Caraballo-Vega, Science Data Processing Branch, Code 587
"""

# Installing
# conda install -c conda-forge gooey
# Running
# Mac pythonw gui_rfdriver.py
# --------------------------------------------------------------------------------
# Import System Libraries
# --------------------------------------------------------------------------------
import sys
import os
import gc
import warnings
from datetime import datetime  # tracking date
from time import time  # tracking time
import argparse  # system libraries
import numpy as np  # for arrays modifications
from gooey import Gooey, GooeyParser
from functools import reduce

from xrasterlib.rf import RF
import xrasterlib.indices as indices

# Fix seed for reproducibility.
seed = 21
np.random.seed(seed)

# Ignoring true_divide errors since we know they are expected
warnings.filterwarnings("ignore", "invalid value encountered in true_divide", RuntimeWarning)
# --------------------------------------------------------------------------------
# methods
# --------------------------------------------------------------------------------


def create_logfile(args, logdir='results'):
    """
    :param args: argparser object
    :param logdir: log directory to store log file
    :return: logfile instance, stdour and stderr being logged to file
    """
    logfile = os.path.join(logdir, '{}_log_{}trees_{}.txt'.format(
        datetime.now().strftime("%Y%m%d-%H%M%S"), args.ntrees, args.maxfeat))
    print('See ', logfile)
    so = se = open(logfile, 'w')  # open our log file
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')  # re-open stdout without buffering
    os.dup2(so.fileno(), sys.stdout.fileno())  # redirect stdout and stderr to the log file opened above
    os.dup2(se.fileno(), sys.stderr.fileno())
    return logfile


def getparser():

    # Initialize GUI object
    parser = GooeyParser()
    subs = parser.add_subparsers(help="commands", dest="command")

    # Train subparser
    train = subs.add_parser("train", prog="Train").add_argument_group("")

    train.add_argument("-c", "--csv", type=str, required=True, dest='traincsv', widget="FileChooser",
                       gooey_options=dict(wildcard="CSV files (*.csv)|*.csv", full_width=True),
                        default=None, help="Specify CSV file to train the model.")
    train.add_argument('-b', '--bands', nargs='*', dest='bands', help='Specify bands.', required=False, type=str,
                       default=['CoastalBlue', 'Blue', 'Green', 'Yellow', 'Red', 'RedEdge', 'NIR1', 'NIR2'])
    train.add_argument("-o", "--out-directory", type=str, required=True, dest='outdir',
                       default='results', help="Specify output directory.")
    train.add_argument("-m", "--model", type=str, required=True, dest='model',
                        default='model.pkl', help="Specify model filename that will be saved or evaluated.")
    train.add_argument("-t", "--n-trees", type=int, required=False, dest='ntrees',
                        default=20, help="Specify number of trees.")
    train.add_argument("-f", "--max-features", type=str, required=False, dest='maxfeat',
                        default='log2', help="Specify max features.")
    train.add_argument("-ts", "--test-size", type=float, required=False, dest='testsize',
                        default='0.30', help="Size of test data.")
    train.add_argument("-l", "--log", required=False, dest='logbool', action='store_true', help="Set logging.")

    # Predict subparser
    classify = subs.add_parser("classify", prog="Classify").add_argument_group("")
    classify.add_argument("-m", "--model", type=str, required=True, dest='model',widget="FileChooser",
                       gooey_options=dict(wildcard="PKL files (*.pkl)|*.pkl", full_width=True),
                        default='results/model.pkl', help="Specify model filename that will be saved or evaluated.")
    classify.add_argument("-i", "--rasters", type=str, nargs='*', required=False, dest='rasters',
                          widget="FileChooser", gooey_options=dict(wildcard="TIF files (*.tif)|*.tif", full_width=True),
                        default='*.tif', help="Image or pattern to evaluate images.")
    classify.add_argument("-o", "--out-directory", type=str, required=True, dest='outdir',
                       default='results', help="Specify output directory.")
    classify.add_argument('-b', '--bands', nargs='*', dest='bands', help='Specify bands.', required=False, type=str,
                       default=['CoastalBlue', 'Blue', 'Green', 'Yellow', 'Red', 'RedEdge', 'NIR1', 'NIR2'])
    classify.add_argument('-d', '--drop-bands', nargs='*', dest='dropbands', help='Specify bands to remove.',
                        required=False, type=str, default=['HOM1', 'HOM2'])
    classify.add_argument("-toaf", "--toa-factor", type=float, required=False, dest='toaf',
                        default=10000.0, help="Specify TOA factor for indices calculation.")
    classify.add_argument("-ws", "--window-size", nargs=2, type=int, required=False, dest='windowsize',
                        default=[5000, 5000], help="Specify window size to perform sliding predictions.")
    classify.add_argument("-ps", "--sieve", required=False, dest='sievebool', action='store_true', help="Apply sieve.")
    classify.add_argument("-pm", "--median", required=False, dest='medianbool', action='store_true', help="Apply median.")
    classify.add_argument("-ss", "--sieve-size", type=int, required=False, dest='sieve_sz',
                        default=800, help="Specify size for sieve filter.")
    classify.add_argument("-ms", "--median-size", type=int, required=False, dest='median_sz',
                        default=20, help="Specify size for median filter.")

    return parser.parse_args()

# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
@Gooey(
    program_name="Random Forest",
    program_description="Build and classify raster imagery.",
    default_size=(600, 720),
    navigation="TABBED",
    progress_regex=r"^Progress (\d+)$",
)
def main():

    start_time = time()  # record start time
    args = getparser()  # initialize arguments parser

    print('Initializing Random Forest script with the following parameters')
    print(f'Working Directory: {args.outdir}')
    print(f'ntrees:            {args.ntrees}')
    print(f'max features:      {args.maxfeat}')
    print(f'model filename:    {args.model}')

    # --------------------------------------------------------------------------------
    # 1. set log file for script if requested (-l command line option)
    # --------------------------------------------------------------------------------
    os.system(f'mkdir -p {args.outdir}')  # create output dir
    if args.logbool:  # if command line option -l was given
        logfile = create_logfile(args, logdir=args.outdir)  # create logfile for std
    print("Command line executed: ", sys.argv)  # saving command into log file

    # --------------------------------------------------------------------------------
    # 2. Instantiate RandomForest object
    # --------------------------------------------------------------------------------
    raster_obj = RF(traincsvfile=args.traincsv, modelfile=args.model,
                    outdir=args.outdir, ntrees=args.ntrees, maxfeat=args.maxfeat)

    # --------------------------------------------------------------------------------
    # 3a. if training csv exists, train
    # --------------------------------------------------------------------------------
    if raster_obj.traincsvfile is not None:
        raster_obj.splitdata(testsize=args.testsize, seed=seed)  # 3a1. read CSV split train/test
        raster_obj.train()  # 3a2. train and save RF model

    ## TODO: Classify piece on the GUI.
    ## TODO: Document GUI and how to access it.

if __name__ == "__main__":
    main()
