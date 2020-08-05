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
import rasterio as rio          # import rasterio for geotiff manipulation

from sklearn.model_selection import train_test_split # train/test data split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from hummingbird.ml import convert # support GPU inference
import torch # import torch to verify available devices

# Fix seed reproducibility.
seed = 21
np.random.seed = seed

def add_DVI(data):
    """
    Difference Vegetation Index (DVI) = (NIR - Red), for 6 bands images, (DVI) = B7 - B5, for 8 bands images
    """
    return ((data[6,:,:] - data[4,:,:])).expand_dims(dim="band", axis=0)

def add_FDI(data):
    """
    Forest Discrimination Index (FDI) = (NIR - (Red + Blue)), for 6 bands images, (FDI) = (B8 - (B6 + B2)), for 8 bands images
    """
    return ((data[7,:,:] - (data[5,:,:] + data[1,:,:]))).expand_dims(dim="band", axis=0)

def add_SI(data):
    """
    Shadow Index (SI) = (1-Blue)*(1-Green)*(1-Red), for 6 bands images, (SI) = (1-B2)*(1-B3)*(1-B5)
    """
    return ((1 - data[1,:,:]) * (1 - data[2,:,:]) * (1 - data[4,:,:])).expand_dims(dim="band", axis=0)

def add_bands(rastarr, bands):
    nbands = rastarr.shape[0]
    for band in [add_DVI(rastarr), add_FDI(rastarr), add_SI(rastarr)]:
        nbands = nbands + 1
        band.coords['band'] = [nbands]
        rastarr = xr.concat([rastarr, band], dim='band')
    rastarr.attrs['scales']  = [rastarr.attrs['scales'][0]] * nbands
    rastarr.attrs['offsets'] = [rastarr.attrs['offsets'][0]] * nbands
    return rastarr

def to_raster(rast, prediction, output='rfmask.tif'):
    # get meta features from raster
    with rio.open(rast) as src:
        meta = src.profile
        print(meta)

    out_meta = meta # modify profile based on numpy array
    out_meta['count'] = 1 # output is single band
    out_meta['dtype'] = 'int16' # data type is float64

    # write to a raster
    with rio.open(output, 'w', **out_meta) as dst:
        dst.write(prediction, 1)


def apply_model(rasters, model, bands=[1,2,3,4,5,6,7,8], resultsdir='results/Results'): 
   
    # open rasters and get both data and coordinates
    for rast in rasters:
        rastarr = xr.open_rasterio(rast, chunks={'band': 1, 'x': 2048, 'y': 2048}) # open raster into xarray
        if rastarr.shape[0] != len(bands): # add additional bands if required, preferibly not
            rastarr = add_bands(rastarr, bands) # add bands here
        print (rastarr)
        
        # getting the shape of the wider scene
        rast_shape = rastarr[0,:,:].shape

        # chunking and doing in memory sliding window predictions
        # size of the scene shape=(11, 9831, 10374), dtype=int16, chunksize=(1, 2048, 2048)
        # <xarray.DataArray (band: 11, y: 39324, x: 47751)
        # window size selected of 10000x10000
        wsx, wsy = 1000, 1000
        
        # crop out the window for prediction
        final_prediction = np.zeros(rast_shape)
        print ("Final prediction initial shape: ", final_prediction.shape)

        for sx in range(0, rast_shape[0], wsx):
            for sy in range(0, rast_shape[1], wsy):
                x0, x1, y0, y1 = sx, sx+wsx, sy, sy+wsy
                if x1 > rast_shape[0]:
                    x1 = rast_shape[0]
                if y1 > rast_shape[1]:
                    y1 = rast_shape[1]
                print (x0, x1, y0, y1)

                window = rastarr[:, x0:x1, y0:y1]
                print ("window type: ", type(window), window.shape)

                # testing the speed of two methods
                # method #1 - numpy array after reshape
                window = window.stack(z=('y', 'x'))
                window = window.transpose("z", "band")
                #window = window.values
                
                # method #2
                #window = window.values
                #window = np.transpose(window, (1, 2, 0))
                #window = window.reshape(window.shape[0] * window.shape[1], window.shape[2])
                
                print (window.shape, type(window))
                
                prediction = model.predict(window)
                prediction = prediction.reshape((x1-x0, y1-y0))
                print (prediction.shape)

                final_prediction[x0:x1, y0:y1] = prediction
                print ("in between final prediction: ", final_prediction.shape)

        # save raster
        output_name = "{}/cm_{}".format(resultsdir, rast.split('/')[-1])
        final_prediction = final_prediction.astype(np.int16)
        to_raster(rast, final_prediction, output=output_name)

        # reshape into new format to be feed into model - long 2D array (nrow * ncol, nband)
        #rastarr = rastarr.stack(z=('y', 'x')) # merge together x-y dimensions.
        #rastarr = rastarr.transpose("z", "band") # change from channel-first to channel last format
        #print (rastarr)

        ### TODO 
        ### A. CHUNK PIECE OF ARRAY
        ### B. stack PIECE OF ARRAY
        ### C. PREDICT PIECE OF ARRAY
        ### D. PLACE INTO FINAL PRODUCT

        """
        # print model information
        print (f'Performing prediction of {rast}...')
        print (rastarr.shape)
        prediction = model.predict(rastarr)
        print (prediction.shape, type(prediction), np.unique(prediction))
        #prediction = np.expand_dims(prediction, axis=1)
        print (rast_shape)
        prediction = prediction.reshape(rast_shape).astype(np.int16)
        print (prediction.shape, np.unique(prediction))
        
        ## TODO: ADD NODATA VALUES TO PREDICTION HERE
        # class_prediction[img[:, :, 0] == ndval] = ndval 

        # save raster
        output_name = "{}/cm_{}".format(resultsdir, rast.split('/')[-1])
        to_raster(rast, prediction, output=output_name)
        """

def train_model(x, y, modelDir, n_trees, max_feat):

    labels = np.unique(y) # now it's the unique values in y array from text file
    print(f'The training data include {labels.size} classes.')
    print(f'Our X matrix is sized: {x.shape}') # shape of x data
    print(f'Our y array is sized:  {y.shape}') # shape of y data

    print ('Initializing model...')
    if '.' not in labels[0]: # if labels are integers, check first value from y (come as string)
        rf = RandomForestClassifier(n_estimators=n_trees, max_features=max_feat, oob_score=True) 
        y = y.astype(np.int)
    else: # if labels are floats, use random forest regressor
        rf = RandomForestRegressor(n_estimators=n_trees, max_features=max_feat, oob_score=True)
        y = y.astype(np.float)

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
                        required=False, type=str, default=['Coastal Blue', 'Blue', 'Green', 'Yellow', 'Red', 'Red Edge',
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
    #logfile = create_logfile(args, logdir=dir_dict['Logs'])
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

        # 3b1. load model - CPU or GPU bound
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        rfmodel = joblib.load(args.model)
        # TODO: GPU support - not accepting dask arrays
        #rfmodel = convert(joblib.load(args.model), 'pytorch')        
        #rfmodel.to(device)
        print (f'Loaded model {args.model} into {device}.')

        # 3b2. apply model and save predictions
        apply_model(rasters=args.rasters, model=rfmodel, bands=args.bands, resultsdir=dir_dict['Results'])
        ### TBD: add apply model, gpu support, parallelization, xarray rasterio

    else:
        sys.exit("ERROR: You should specify a train csv or a model to load. Refer to python " +
                 "rasterRF.py -h for more options.")

    print("Elapsed Time: ", (time() - start_time) / 60.0)  # output program run time


if __name__ == "__main__":

    main()

    # simple test:
    # python rasterRF.py -w results -m test -b 1 2 3 -bn red green blue
