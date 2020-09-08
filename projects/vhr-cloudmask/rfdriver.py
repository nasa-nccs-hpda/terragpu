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
import sys
import os
import warnings
from datetime import datetime  # tracking date
from time import time  # tracking time
import argparse  # system libraries
import numpy as np  # for arrays modifications

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
    """
    :return: argparser object with CLI commands.
    """
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("-l", "--log", required=False, dest='logbool', action='store_true', help="Set logging.")
    parser.add_argument("-o", "--out-directory", type=str, required=True, dest='outdir',
                        default="", help="Specify output directory.")
    parser.add_argument("-m", "--model", type=str, required=False, dest='model',
                        default=None, help="Specify model filename that will be saved or evaluated.")
    parser.add_argument('-b', '--bands', nargs='*', dest='bands', help='Specify bands.',
                        required=False, type=str, default=['CoastalBlue', 'Blue', 'Green', 'Yellow', 'Red', 'RedEdge',
                                                           'NIR1', 'NIR2'])
    # Train
    parser.add_argument("-c", "--csv", type=str, required=False, dest='traincsv',
                        default=None, help="Specify CSV file to train the model.")
    parser.add_argument("-t", "--n-trees", type=int, required=False, dest='ntrees',
                        default=20, help="Specify number of trees for random forest model.")
    parser.add_argument("-f", "--max-features", type=str, required=False, dest='maxfeat',
                        default='log2', help="Specify random forest max features.")
    parser.add_argument("-ts", "--test-size", type=float, required=False, dest='testsize',
                        default='0.30', help="Size of test data (e.g: .30)")
    # Inference
    parser.add_argument("-i", "--rasters", type=str, nargs='*', required=False, dest='rasters',
                        default='*.tif', help="Image or pattern to evaluate images.")
    parser.add_argument('-d', '--drop-bands', nargs='*', dest='dropbands', help='Specify bands to remove.',
                        required=False, type=str, default=['HOM1', 'HOM2'])
    parser.add_argument("-toaf", "--toa-factor", type=float, required=False, dest='toaf',
                        default=10000.0, help="Specify TOA factor for indices calculation.")
    parser.add_argument("-ws", "--window-size", nargs=2, type=int, required=False, dest='windowsize',
                        default=[5000, 5000], help="Specify window size to perform sliding predictions.")
    parser.add_argument("-ps", "--sieve", required=False, dest='sievebool', action='store_true', help="Apply sieve.")
    parser.add_argument("-pm", "--median", required=False, dest='medianbool', action='store_true', help="Apply median.")
    parser.add_argument("-ss", "--sieve-size", type=int, required=False, dest='sieve_sz',
                        default=800, help="Specify size for sieve filter.")
    parser.add_argument("-ms", "--median-size", type=int, required=False, dest='median_sz',
                        default=20, help="Specify size for median filter.")
    return parser.parse_args()


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
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

    # --------------------------------------------------------------------------------
    # 3b. if model exists, predict
    # --------------------------------------------------------------------------------
    elif raster_obj.modelfile is not None:
        raster_obj.load()  # 3b1. load model - CPU or GPU bound
        assert (args.rasters and args.rasters != '*.tif'), "No raster to predict, python rfdriver.py -h for options."

        # 3b3. apply model and get predictions
        for rast in args.rasters:  # iterate over each raster
            raster_obj.readraster(rast, args.bands)  # read raster

            # preprocess raster to remove anomalous pixels, make boundaries (0, 10000)
            raster_obj.preprocess(op='>', boundary=0, replace=0)
            raster_obj.preprocess(op='<', boundary=10000, replace=10000)
            assert (raster_obj.data.min().values == 0 and raster_obj.data.max().values == 10000), \
                "min and max should be (0, 10000). Verify preprocess."

            # add additional indices if necessary
            if raster_obj.model_nfeat != raster_obj.data.shape[0]:
                raster_obj.addindices([indices.fdi, indices.si, indices.ndwi], factor=args.toaf)

            # drop unnecessary bands if necessary
            if raster_obj.model_nfeat != raster_obj.data.shape[0]:
                raster_obj.dropindices(args.dropbands)

            raster_obj.predict(ws=args.windowsize)  # predict

            # TODO: check if order matters between sive and median

            # apply sieve filter if necessary
            if args.sievebool:
                raster_obj.sieve(raster_obj.prediction, raster_obj.prediction, size=args.sieve_sz)  # apply sieve

            # apply median filter if necessary
            if args.medianbool:
                raster_obj.prediction = raster_obj.median(raster_obj.prediction, ksize=args.median_sz)  # apply median

            output_name = "{}/cm_{}".format(raster_obj.outdir, rast.split('/')[-1])  # out mask name
            raster_obj.toraster(rast, raster_obj.prediction, output_name)  # save raster with mask

    # --------------------------------------------------------------------------------
    # 3c. exit if csv or model are not present or given
    # --------------------------------------------------------------------------------
    else:
        sys.exit("ERROR: Specify a train csv or model to load, python rfdriver.py -h for more options.")
    print("Elapsed Time: ", (time() - start_time) / 60.0)  # output program run time in minutes


if __name__ == "__main__":
    main()
