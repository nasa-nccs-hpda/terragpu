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
from datetime import datetime  # tracking date
from time import time          # tracking time
import sys, os, argparse       # system libraries
import joblib                  # joblib for parallel jobs
import numpy as np             # for arrays modifications
import pandas as pd            # csv data frame modifications
import xarray as xr            # read rasters
import rasterio as rio         # geotiff manipulation

from sklearn.model_selection import train_test_split  # train/test data split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from hummingbird.ml import convert  # support GPU inference
import torch  # import torch to verify available devices

# Fix seed for reproducibility.
seed = 21
np.random.seed = seed

#--------------------------------------------------------------------------------
# Functions
#--------------------------------------------------------------------------------
def add_DVI(data):
    # Difference Vegetation Index (DVI) = B7 - B5, for 8 bands images
    return (data[6, :, :] - data[4, :, :]).expand_dims(dim="band", axis=0)

def add_FDI(data):
    # Forest Discrimination Index (FDI) = (B8 - (B6 + B2)), for 8 bands images
    return (data[7, :, :] - (data[5, :, :] + data[1, :, :])).expand_dims(dim="band", axis=0)

def add_SI(data):
    # Shadow Index (SI) = (1-Blue)*(1-Green)*(1-Red), for 6 bands images, (SI) = (1-B2)*(1-B3)*(1-B5)
    return ((1 - data[1, :, :]) * (1 - data[2, :, :]) * (1 - data[4, :, :])).expand_dims(dim="band", axis=0)


def add_bands(rastarr):
    """
    Add indices if they are missing
    :param rastarr: xarray raster
    :return: xarray raster with three additional bands
    """
    nbands = rastarr.shape[0]  # get number of bands
    for band in [add_DVI(rastarr), add_FDI(rastarr), add_SI(rastarr)]:
        nbands = nbands + 1
        band.coords['band'] = [nbands]  # add band indices to raster
        rastarr = xr.concat([rastarr, band], dim='band')  # concat new band
    rastarr.attrs['scales'] = [rastarr.attrs['scales'][0]] * nbands
    rastarr.attrs['offsets'] = [rastarr.attrs['offsets'][0]] * nbands
    return rastarr  # return xarray with new bands


def to_raster(rast, prediction, output='rfmask.tif'):
    """
    :param rast: raster name to get metadata from
    :param prediction: numpy array with prediction output
    :param output: raster name to save on
    :return: tif file saved to disk
    """
    # get meta features from raster
    with rio.open(rast) as src:
        meta = src.profile
        print(meta)

    out_meta = meta  # modify profile based on numpy array
    out_meta['count'] = 1  # output is single band
    out_meta['dtype'] = 'int16'  # data type is float64

    # write to a raster
    with rio.open(output, 'w', **out_meta) as dst:
        dst.write(prediction, 1)
    print(f'Prediction saved at {output}.')


def apply_model(rasters, model, ws=[5120, 5120], bands=[1, 2, 3, 4, 5, 6, 7, 8], resultsdir='results/Results'):
    """
    :param rasters: list of raster names
    :param model: model object to predict with
    :param ws: window size to predict
    :param bands: list with band ids
    :param resultsdir: where to store results
    :return: tif prediction files save to disk
    """
    # open rasters and get both data and coordinates
    for rast in rasters:  # iterate over each raster
        rastarr = xr.open_rasterio(rast, chunks={'band': 1, 'x': 2048, 'y': 2048})  # open raster into xarray
        if rastarr.shape[0] != len(bands):  # add additional bands if required, preferibly not
            rastarr = add_bands(rastarr)  # add bands here
        print(rastarr)  # print raster information
        
        rast_shape = rastarr[0, :, :].shape  # getting the shape of the wider scene
        wsx, wsy = ws[0], ws[1]  # chunking and doing in memory sliding window predictions

        # if the window size is bigger than the image, ignore and predict full image
        if wsx > rast_shape[0]:
            wsx = rast_shape[0]
        if wsy > rast_shape[1]:
            wsy = rast_shape[1]

        final_prediction = np.zeros(rast_shape)  # crop out the window for prediction
        print("Final prediction initial shape: ", final_prediction.shape)

        for sx in range(0, rast_shape[0], wsx):  # iterate over x-axis
            for sy in range(0, rast_shape[1], wsy):  # iterate over y-axis
                x0, x1, y0, y1 = sx, sx+wsx, sy, sy+wsy  # assign window indices
                if x1 > rast_shape[0]:  # if selected x-indices exceeds boundary
                    x1 = rast_shape[0]  # assign boundary to x-window
                if y1 > rast_shape[1]:  # if selected y-indices exceeds boundary
                    y1 = rast_shape[1]  # assign boundary to y-window

                window = rastarr[:, x0:x1, y0:y1]  # get window
                window = window.stack(z=('y', 'x'))  # stack y and x axis
                window = window.transpose("z", "band").values  # reshape xarray, return numpy arr
                final_prediction[x0:x1, y0:y1] = (model.predict(window)).reshape((x1-x0, y1-y0))

        # save raster
        output_name = "{}/cm_{}".format(resultsdir, rast.split('/')[-1])  # model name
        final_prediction = final_prediction.astype(np.int16)  # change type of prediction to int16
        to_raster(rast, final_prediction, output=output_name)  # load prediction to raster


def train_model(x, y, modeldir='results/Models', modelname='rfmodel', n_trees=20, max_feat='log2'):
    """
    :param x: dataframe with x data
    :param y: dataframe with y data
    :param modeldir: directory to save model on
    :param modelname: name for the model to store
    :param n_trees: number of trees for random forest
    :param max_feat: max features for random forest
    :return: saved model object
    """
    labels = np.unique(y)  # now it's the unique values in y array from text file
    print(f'The training data include {labels.size} classes.')
    print(f'Our X matrix is sized: {x.shape}')  # shape of x data
    print(f'Our y array is sized:  {y.shape}')  # shape of y data

    print('Initializing model...')
    if '.' not in labels[0]:  # if labels are integers, check first value from y (come as string)
        rf = RandomForestClassifier(n_estimators=n_trees, max_features=max_feat, oob_score=True) 
        y = y.astype(np.int)
    else:  # if labels are floats, use random forest regressor
        rf = RandomForestRegressor(n_estimators=n_trees, max_features=max_feat, oob_score=True)
        y = y.astype(np.float)

    print("Training model...")
    rf.fit(x, y)  # fit model to training data
    print('Score:', rf.oob_score_)

    try:  # export model to file
        model_save = os.path.join(modeldir, f'{modelname}.pkl')
        joblib.dump(rf, model_save)
    except Exception as e:
        print(f'Error: {e}')
    return model_save  # Return output model for application and validation


def get_test_training_sets(traincsv, testsize=0.30):
    """
    :param traincsv: csv filename to create dataframe
    :param testsize: size of testing features
    :return: 4 arrays with train and test data respectively
    """
    df = (pd.read_csv(traincsv, header=None, sep=',')).values  # generate pd dataframe
    x = df.T[0:-1].T.astype(str)
    y = df.T[-1].astype(str)
    # returns 4 numpy arrays with train and test data
    return train_test_split(x, y, test_size=testsize, random_state=seed)


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
                        default="", help="Specify model filename that will be saved or evaluated")
    parser.add_argument('-b', '--bands', nargs='*', dest='bands', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        help='Specify number of bands.', required=True, type=int)
    parser.add_argument('-bn', '--band-names', nargs='*', dest='band_names', help='Specify number of bands.',
                        required=False, type=str, default=['Coastal Blue', 'Blue', 'Green', 'Yellow', 'Red', 'Red Edge',
                        'Near-IR1', 'Near-IR2', 'DVI', 'FDI', 'SI'])
    # Train
    parser.add_argument("-c", "--csv", type=str, required=False, dest='traincsv',
                        default="", help="Specify CSV file to train the model.")
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

    start_time = time() # record start time
    args = getparser()  # initialize arguments parser

    print('Initializing Random Forest script with the following parameters')
    print(f'Working Directory: {args.workdir}')
    print(f'n_trees:           {args.n_trees}')
    print(f'max features:      {args.max_feat}')

    # 1. set working, logs,  and results directories, return dictionary
    dir_dict = create_directories(args.workdir)

    # 2. set log file for script - you may disable this when developing
    logfile = create_logfile(args, logdir=dir_dict['Logs'])
    print("Command used: ", sys.argv)  # saving command into log file

    # 3a. if training csv does not exist, proceed and train
    if os.path.isfile(args.traincsv):

        # 3a1. read CSV training data and split into train and test sets
        # returns four numpy arrays to train the model on
        print(f'Input train CSV: {args.traincsv}')
        x_train, x_test, y_train, y_test = get_test_training_sets(args.traincsv, args.testsize)
        print(f'y_train lenght: {len(y_train)} and y_test lenght: {len(y_test)}')

        # 3a2. train and save the model
        print(f'Building model with n_trees={args.n_trees} and max_feat={args.max_feat}...')
        if not args.model:  # if model argument was not given, set default model name
            args.model = 'model_{}_{}'.format(args.n_trees, args.max_feat)
        # training and saving the model in this step
        model_save = train_model(x_train, y_train, modelDir=dir_dict['Models'],
                                 modelname=args.model, n_trees=args.n_trees, max_feat=args.max_feat)
        print(f'Model has been saved as {model_save}')

    # 3b. evaluate images from model
    elif os.path.isfile(args.model):

        # 3b1. load model - CPU or GPU bound
        rfmodel = joblib.load(args.model)  # loading the model in parallel
        device = 'cpu'  # set cpu as default device
        if torch.cuda.is_available():  # if cuda is available, assign model to GPU
            device = torch.device('cuda:0')  # assign device
            rfmodel = convert(rfmodel, 'pytorch')  # convert model to tensors for GPU
            rfmodel.to(device)  # assign model to GPU
        print(f'Loaded model {args.model} into {device}.')
        
        if not args.rasters:  # if raster -i variable is empty, stop process and log.
            sys.exit("ERROR: No images to predict. Refer to python rasterRF.py -h for options.")

        # 3b2. apply model and save predictions
        apply_model(rasters=args.rasters, model=rfmodel, ws=args.windowsize,
                    bands=args.bands, resultsdir=dir_dict['Results'])
    else:
        sys.exit("ERROR: You should specify a train csv or a model to load. Refer to python " +
                 "rasterRF.py -h for more options.")

    print("Elapsed Time: ", (time() - start_time) / 60.0)  # output program run time in minutes


if __name__ == "__main__":

    main()
