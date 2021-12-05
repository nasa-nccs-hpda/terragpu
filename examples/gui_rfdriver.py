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

import sys
import os
import gc
import logging
import warnings
from time import time  # tracking time
import argparse  # system libraries
import numpy as np  # for arrays modifications
import cupy as cp
import torch  # use for GPU acceleration

from xrasterlib.ml.rf import RF
from xrasterlib.utils import create_logfile
import xrasterlib.indices as indices

import dask.array as da
#from dask_cuda import LocalCUDACluster
#from dask.distributed import Client

#cluster = LocalCUDACluster()
#client = Client(cluster)

try:
    import gooey
except ImportError:
    gooey = None

# Fix seed for reproducibility.
seed = 21
np.random.seed(seed)
torch.cuda.empty_cache()

# Ignoring true_divide errors since we know they are expected
warnings.filterwarnings(
    "ignore", "invalid value encountered in true_divide", RuntimeWarning
)

# --------------------------------------------------------------------------------
# methods
# --------------------------------------------------------------------------------

# Do not run GUI if it is not available or if command-line arguments are given.
if gooey is None or len(sys.argv) > 1:
    def gui_decorator(f):
        return f
    gui = False

# initialize Gooey decorator and set warning if this was not intended
else:
    logging.warning(
        "Initializing GUI since no arguments were given. Refer to python " +
        "rfdriver.py -h for usage options if this was not intended"
    )
    gui_decorator = gooey.Gooey(
        suppress_gooey_flag=True,
        program_name="Random Forest Cloud Mask Classifier",
        program_description="Train and Classify raster imagery using RF.",
        default_size=(980, 720),
        navigation="TABBED",
        progress_regex=r"^Progress (\d+)$",
    )
    gui = True


def getparser(gui):

    # if GUI is not available, initialize plain arg parser object
    if gui:
        parser = gooey.GooeyParser(description="RF driver")
    else:
        parser = argparse.ArgumentParser(description="RF driver")

    # defining subparsers, one for training one for classification
    subs = parser.add_subparsers(dest="command")
    train = subs.add_parser("train", prog="Train")
    classify = subs.add_parser("classify", prog="Classify")

    # Train
    if gui:
        train.add_argument(
            "-c", "--csv", type=str, required=False, dest='traincsv',
            default=None, help="CSV file to train from mask",
            widget="FileChooser",
            gooey_options=dict(
                wildcard="CSV files (*.csv)|*.csv", full_width=True
            ),
        )
    else:
        train.add_argument(
            "-c", "--csv", type=str, required=False, dest='traincsv',
            default=None, help="CSV file to train from mask"
        )
    train.add_argument(
        '-b', '--bands', nargs='*', dest='bands',
        help="Raster bands ids (shortname)",
        default=['CoastalBlue', 'Blue', 'Green', 'Yellow', 'Red', 'RedEdge',
                 'NIR1', 'NIR2'], required=False, type=str
    )
    train.add_argument(
        "-o", "--out-directory", type=str, required=True,
        dest='outdir', default="", help="Output directory artifacts"
    )
    train.add_argument(
        "-m", "--model", type=str, required=False, dest='model',
        default=None, help="Model filename (.pkl)"
    )
    train.add_argument(
        "-t", "--n-trees", type=int, required=False, dest='ntrees',
        default=20, help="n trees for RF"
    )
    train.add_argument(
        "-f", "--max-features", type=str, required=False, dest='maxfeat',
        default='log2', help="RF max features"
    )
    train.add_argument(
        "-ts", "--test-size", type=float, required=False, dest='testsize',
        default='0.30', help="Ratio of test data"
    )
    train.add_argument(
        "-l", "--log", required=False, dest='logbool',
        action='store_true', help="Set logging"
    )

    # Predict subparser
    if gui:
        classify.add_argument(
            "-m", "--model", type=str, required=False, dest='model',
            widget="FileChooser", default=None, help="Model filename (.pkl)",
            gooey_options=dict(
                wildcard="PKL files (*.pkl)|*.pkl",
                full_width=True
            ),
        )
        classify.add_argument(
            "-i", "--rasters", type=str, nargs='*', required=False,
            dest='rasters', default='*.tif', help="Name or pattern to imagery",
            widget="FileChooser", gooey_options=dict(
                wildcard="TIF files (*.tif)|*.tif", full_width=True
            )
        )
    else:
        classify.add_argument(
            "-m", "--model", type=str, required=False, dest='model',
            default=None, help="Model filename (.pkl)"
        )
        classify.add_argument(
            "-i", "--rasters", type=str, nargs='*', required=False,
            dest='rasters', default='*.tif', help="Image or pattern to rasters"
        )
    classify.add_argument(
        "-o", "--out-directory", type=str, required=True,
        dest='outdir', default="", help="Specify output directory"
    )
    classify.add_argument(
        '-b', '--bands', nargs='*', dest='bands', required=False, type=str,
        help='Raster bands ids (shortname)',
        default=['CoastalBlue', 'Blue', 'Green', 'Yellow',
                 'Red', 'RedEdge', 'NIR1', 'NIR2']
    )
    classify.add_argument(
        '-d', '--drop-bands', nargs='*', dest='dropbands', required=False,
        help='Bands to remove', type=str, default=['HOM1', 'HOM2']
    )
    classify.add_argument(
        "-toaf", "--toa-factor", type=float, required=False, dest='toaf',
        default=10000.0, help="TOA factor for indices calculation."
    )
    classify.add_argument(
        "-ws", "--window-size", nargs=2, type=int, required=False,
        dest='windowsize', default=[5000, 5000],
        help="Window size to perform sliding predictions"
    )
    classify.add_argument(
        "-ps", "--sieve", required=False, dest='sievebool',
        action='store_true', help="Apply sieve"
    )
    classify.add_argument(
        "-pm", "--median", required=False, dest='medianbool',
        action='store_true', help="Apply median"
    )
    classify.add_argument(
        "-l", "--log", required=False, dest='logbool',
        action='store_true', help="Set logging"
    )
    classify.add_argument(
        "-ss", "--sieve-size", type=int, required=False, dest='sieve_sz',
        default=800, help="Size for sieve filter"
    )
    classify.add_argument(
        "-ms", "--median-size", type=int, required=False, dest='median_sz',
        default=20, help="Size for median filter"
    )

    return parser.parse_args()


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------
@gui_decorator
def main():

    start_time = time()  # record start time
    args = getparser(gui)  # initialize arguments parser

    # --------------------------------------------------------------------------------
    # 1. set log file for script if requested (-l command line option)
    # --------------------------------------------------------------------------------
    os.system(f'mkdir -p {args.outdir}')  # create output dir
    if args.logbool:  # if command line option -l was given
        create_logfile(logdir=args.outdir)  # create logfile for std
    print("Command line executed: ", sys.argv)  # saving command into log file

    # --------------------------------------------------------------------------------
    # 3a. train
    # --------------------------------------------------------------------------------
    if args.command == "train":
        # --------------------------------------------------------------------------------
        # 2. Instantiate RandomForest object
        # --------------------------------------------------------------------------------
        raster_obj = RF(
            traincsvfile=args.traincsv, modelfile=args.model,
            outdir=args.outdir, ntrees=args.ntrees, maxfeat=args.maxfeat
        )

        print('Initializing RF script with the following parameters')
        print(f'Working Directory: {args.outdir}')
        print(f'ntrees:            {args.ntrees}')
        print(f'max features:      {args.maxfeat}')
        print(f'model filename:    {args.model}')
        raster_obj.splitdata(testsize=args.testsize, seed=seed)  # read, split
        raster_obj.train()  # train and save RF model

    # --------------------------------------------------------------------------------
    # 3b. if model exists, predict
    # --------------------------------------------------------------------------------
    elif args.command == "classify":
        raster_obj = RF(modelfile=args.model, outdir=args.outdir)
        raster_obj.load()  # 3b1. load model - CPU or GPU bound
        assert (args.rasters and args.rasters != '*.tif'), \
            "No raster to predict, python rfdriver.py -h for options."

        # 3b3. apply model and get predictions
        for rast in args.rasters:  # iterate over each raster

            gc.collect()  # clean garbage
            print(f"Starting new prediction...{rast}")
            raster_obj.readraster(rast, args.bands)  # read raster

            print(raster_obj.data)

            #cp.cuda.Device(1).use()
            raster_obj.data = raster_obj.data.data.map_blocks(cp.asarray)
            print(raster_obj.data.shape[0])

            # remove anomalous pixels, make boundaries (0, 10000)
            print("starting preprocess 1")
            raster_obj.preprocess(op='>', boundary=0, subs=0)
            print("starting preprocess 2")
            raster_obj.preprocess(op='<', boundary=10000, subs=10000)
            print("done with preprocess")

            """
            assert (raster_obj.data.min().values >= 0 and
                    raster_obj.data.max().values <= 10000), \
                "min and max should be (0, 10000). Verify preprocess."
            """

            # add additional indices if necessary
            print(f"Size of raster {raster_obj.data.shape[0]} before indices")
            if raster_obj.model_nfeat != raster_obj.data.shape[0]:
                raster_obj.addindices(
                    [indices.fdi, indices.si, indices.ndwi], factor=args.toaf
                )

            print(raster_obj.data)

            """
            # drop unnecessary bands if necessary
            print(f"Size of raster {raster_obj.data.shape[0]} after indices")
            if raster_obj.model_nfeat != raster_obj.data.shape[0]:
                raster_obj.dropindices(args.dropbands)

            raster_obj.predict(ws=args.windowsize)  # predict

            # TODO: check if order matters between sive and median

            # apply median filter if necessary
            print("starting median filter")
            if args.medianbool:
                raster_obj.prediction = raster_obj.median(
                    raster_obj.prediction,
                    ksize=args.median_sz
                )

            # apply sieve filter if necessary
            print("starting sieve filter")
            if args.sievebool:
                raster_obj.sieve(
                    raster_obj.prediction,
                    raster_obj.prediction,
                    size=args.sieve_sz
                )

            # apply median filter if necessary
            #if args.medianbool:
            #    raster_obj.prediction = raster_obj.median(
            #        raster_obj.prediction,
            #        ksize=args.median_sz
            #    )

            # out mask name, save raster
            output_name = "{}/cm_{}".format(
                raster_obj.outdir, rast.split('/')[-1]
            )
            raster_obj.toraster(rast, raster_obj.prediction, output_name)
            raster_obj.prediction = None  # unload between each iteration
            """
    # --------------------------------------------------------------------------------
    # 3c. exit if csv or model are not present or given
    # --------------------------------------------------------------------------------
    else:
        sys.exit("ERROR: Specify a train csv or model to load." +
                 "Refer to python rfdriver.py -h for more options.")
    print("Elapsed Time: ", (time() - start_time) / 60.0)  # time in min

    # TODO: Document GUI and how to access it.


if __name__ == "__main__":
    main()
