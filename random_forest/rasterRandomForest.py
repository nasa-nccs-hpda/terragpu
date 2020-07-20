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
import skimage.io as io         # managing images
import matplotlib.pyplot as plt # visualizations

from osgeo import gdal, gdal_array                   # gdal for raster changes
from sklearn.model_selection import train_test_split # train/test data split
from sklearn.ensemble import RandomForestClassifier  # random forest classifier
from hummingbird.ml import convert                   # support GPU training

# Fix seed reproducibility.
seed = 42
np.random.seed = seed

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()                    # enable exceptions to report errors
gdal.AllRegister()                      # GDAL register all drivers
drvtif = gdal.GetDriverByName("GTiff")  # load GTiff driver to open rasters

n_trees = 20
max_feat = 'log2'
modelName = '{}_{}'.format(n_trees, max_feat)# 'try3' # to distinguish between parameters of each model


extentName = '5Images' #sys.argv[1]#'qA' # model number or name, so we can save to different directories
if True:
    bands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] # bands of the image stack #ANDREW
    band_names = ['Coastal Blue', 'Blue', 'Green', 'Yellow', 'Red', 'Red Edge', 'Near-IR1', 'Near-IR2', 'DVI', 'FDI', 'SI'] # the order of the stack #ANDREW


#############################################################################


"""Function to read data stack into img object"""
def stack_to_obj(VHRstack):

    img_ds = gdal.Open(VHRstack, gdal.GA_ReadOnly) # GDAL dataset

    gt = img_ds.GetGeoTransform()
    proj = img_ds.GetProjection()
    ncols = img_ds.RasterXSize
    nrows = img_ds.RasterYSize
    ndval = img_ds.GetRasterBand(1).GetNoDataValue() # should be -999 for all layers, unless using scene as input

    imgProperties = (gt, proj, ncols, nrows, ndval)

    """ Read data stack into array """
    #MAGGIE - NOTE THE ADDITION OF 3 Layers to np array to account for indicies
    img = np.zeros((nrows, ncols, (img_ds.RasterCount + 3)), gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType)) # + 3 for indicies
    #img = np.zeros((nrows, ncols, 4), gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    for b in range(img.shape[2]): # the 3rd index of img.shape gives us the number of bands in the stack
        if b >= 8: # Skipping anything above band 8
            continue
        print('\nb: {}'.format(b))
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray() # GDAL is 1-based while Python is 0-based

    return (img, imgProperties)


#############################################################################


"""Function to write final classification to tiff"""
def array_to_tif(inarr, outfile, imgProperties):

    # get properties from input
    (gt, proj, ncols, nrows, ndval) = imgProperties
    print(ndval)

    drv = drvtif.Create(outfile, ncols, nrows, 1, 3, options = [ 'COMPRESS=LZW' ]) # 1= number of bands (i think) and 3 = Data Type (16 bit signed)
    drv.SetGeoTransform(gt)
    drv.SetProjection(proj)
    drv.GetRasterBand(1).SetNoDataValue(ndval)
    drv.GetRasterBand(1).WriteArray(inarr)

    return outfile


#############################################################################


def add_index_band(img, band, indexcalc):
    b1 = img[:,:,0]
    b2 = img[:,:,1]
    b3 = img[:,:,2]
    b4 = img[:,:,3]
    b5 = img[:,:,4]
    b6 = img[:,:,5]
    b7 = img[:,:,6]
    b8 = img[:,:,7]

    calc = eval(indexcalc)

    b1, b2, b3, b4, b5, b6, b7, b8 = None, None, None, None, None, None, None, None

    img[:, :, band] = calc

    calc = None

    return img


#############################################################################


"""Function to run diagnostics on model"""
def run_diagnostics(model_save, X, y): # where model is the model object, X and y are training sets

    # load model for use:
    print("\nLoading model from {} for cross-val".format(model_save))
    model_load = joblib.load(model_save) # nd load

    print("\n\nDIAGNOSTICS:\n")

    try:
        print("n_trees = {}".format(n_trees))
        print("max_features = {}\n".format(max_feat))
    except Exception as e:
        print("ERROR: {}\n".format(e))

    # check Out of Bag (OOB) prediction score
    print('Our OOB prediction of accuracy is: {}\n'.format(model_load.oob_score_ * 100))
    print("OOB error: {}\n".format(1 - model_load.oob_score_))



    # check the importance of the bands:
    for b, imp in zip(bands, model_load.feature_importances_):
        print('Band {b} ({name}) importance: {imp}'.format(b=b, name=band_names[b-1], imp=imp))
    print('')


    """
    # see http://scikit-learn.org/stable/modules/cross_validation.html for how to use rf.score etc
    """

    # dont really know if this is applicable for 2 classes but try it anyway:
    # Setup a dataframe -- just like R
    df = pd.DataFrame() #**** Need to create a new y with validation points, like we did with y in the function below (make roi be valid sites array instead of training)
    df['truth'] = y
    df['predict'] = model_load.predict(X)

    # Cross-tabulate predictions
    print(pd.crosstab(df['truth'], df['predict'], margins=True))

##    print "Other:"
##    print model.criterion
##    print model.estimator_params
##    print model.score
##    print model.feature_importances_
##    print ''


#############################################################################


def apply_model(VHRstack, classDir, model_save): # VHR stack we are applying model to, output dir, and saved model

    bname =os.path.basename(VHRstack).strip('.tif')
    (img, imgProperties) = stack_to_obj(VHRstack)
    (gt, proj, ncols, nrows, ndval) = imgProperties # ndval is nodata val of image stack not sample points

    # Add DVI Index
    img = add_index_band(img, 8, 'b7-b5') #DVI

    # Add FDI Index
    img = add_index_band(img, 9, '(b8 - (b6 + b2))') #FDI 

    # Add SI Index
    img = add_index_band(img, 10, '(1-b2)*(1-b3)*(1-b5)') #SI

    print(img)
    print(img.shape)
    print(np.unique(img))

    # Classification of img array and save as image (5 refers to the number of bands in the stack)
    # reshape into long 2d array (nrow * ncol, nband) for classification
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    img_as_array = img[:, :, :img.shape[2]].reshape(new_shape) # 5 is number of layers
##    print img_as_array.shape # (192515625, 5)
##    print np.unique(img_as_array) # [ -999  -149  -146 ..., 14425 14530 14563]

    print('Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape))

    print("\nLoading model from {}".format(model_save))
    model_load = joblib.load(model_save) # nd load
    print("\nModel information:\n{}".format(model_load))

    # Now predict for each pixel
    class_prediction = model_load.predict(img_as_array)

    #* at some point may need to convert values that were -999 in img array back to -999, depending on what rf does to those areas

##    print img[:, :, 0].shape # (13875, 13875)
##    print img[:, :, 0]

    # Reshape our classification map and convert to 16-bit signed
    class_prediction = class_prediction.reshape(img[:, :, 0].shape).astype(np.float32)

##    print class_prediction # numpy array? what?
##    print class_prediction.shape # (13875, 13875)
##    print class_prediction.dtype #uint8
##    print np.unique(class_prediction) # [1 2]
##    print img.shape # (13875, 13875, 5)
##    print np.unique(img) # [ -999  -149  -146 ..., 14425 14530 14563]

    # Now we need to convert existing NoData values back to NoData (-999, or 0 if at scene-level)
    class_prediction[img[:, :, 0] == ndval] = ndval # just chose the 0th index to find where noData values are (should be the same for all MS layers, not ure about PAn)
    # use from old method to save to tif
##    print np.unique(class_prediction)
##    print ndval

    img = None

    # export classificaiton to tif
    classification = os.path.join(classDir, "{}__{}__classified.tif".format(modelName,bname))
    array_to_tif(class_prediction, classification, imgProperties)
##    io.imsave(classification, class_prediction)
    print("\nWrote map output to {}".format(classification))



    return


#############################################################################


"""Function for training the model using training data"""
# To train the model, you need: input text file/csv, model output location, model parameters
def train_model(X, y, modelDir, n_trees, max_feat, catalogid):
    #import pdb;pdb.set_trace()

    n_samples = np.shape(X)[0] # the first index of the shape of X will tell us how many sample points
    print('\nWe have {} samples'.format(n_samples))

    labels = np.unique(y) # now it's the unique values in y array from text file
    print('The training data include {n} classes: {classes}'.format(n=labels.size, classes=labels))


    print('Our X matrix is sized: {sz}'.format(sz=X.shape)) # hopefully will be (?, 4)
    print('Our y array is sized: {sz}'.format(sz=y.shape))


    """ Now train the model """
    print("\nInitializing model...")
    rf = RandomForestClassifier(n_estimators=n_trees, max_features=max_feat, oob_score=True) # can change/add other settings later

    print("\nTraining model...")
    rf.fit(X, y) # fit model to training data

    print('score:', rf.oob_score_)

    # Export model:
    try:
        model_save = os.path.join(modelDir, "model_{}_{}.pkl".format(extentName, catalogid))
        joblib.dump(rf, model_save)
    except Exception as e:
        print("Error: {}".format(e))


    return model_save # Return output model for application and validation


#############################################################################


def get_test_training_sets(inputText):

    df = pandas.read_csv(inputText, header=None, sep=',')
    data = df.values
    X = data.T[0:-1].T.astype(str)
    print(X)
    y = data.T[-1].astype(str)
    print(y)

    # Now we have X and y, but this is not split into validation and training. Do that here:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 21)

    return (X_train, X_test, y_train, y_test)


#############################################################################


"""Function to convert stack and sample imagery into X and y [not yet split into training/validation]"""
def convert_img_to_Xy_data(VHRstack, sampleTif): # Given a data stack and sample tiff, we can convert these into X and y for Random Forests

    """ Read in the raster stack and training data to numpy array: """
    img_ds = gdal.Open(VHRstack, gdal.GA_ReadOnly) # GDAL dataset
    roi_ds = gdal.Open(trainTif, gdal.GA_ReadOnly)

    """ get geo metadata so we can write the classification back to tiff """
    gt = img_ds.GetGeoTransform()
    proj = img_ds.GetProjection()
    ncols = img_ds.RasterXSize
    nrows = img_ds.RasterYSize
    ndval = img_ds.GetRasterBand(1).GetNoDataValue() # should be -999 for all layers, unless using scene as input

    imgProperties = (gt, proj, ncols, nrows, ndval)

    """ Read data stack into array """
    img = np.zeros((nrows, ncols, img_ds.RasterCount), gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    for b in range(img.shape[2]): # the 3rd index of img.shape gives us the number of bands in the stack
        print('\nb: {}'.format(b))
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray() # GDAL is 1-based while Python is 0-based

    """ Read Training dataset into numpy array """
    roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    roi_nd = roi_ds.GetRasterBand(1).GetNoDataValue() # get no data value of training dataset
    roi[roi==roi_nd] = 0 # Convert the No Data pixels in raster to 0 for the model

    X = img[roi > 0, :]  # Data Stack pixels
    y = roi[roi > 0]     # Training pixels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 21)

    return (X_train, X_test, y_train, y_test)


#############################################################################


def create_directories(main_dir):
    VHRdir = '/att/gpfsfs/briskfs01/ppl/mwooten3/Vietnam_LCLUC/TOA/M1BS/8-band/pansharpen'#os.path.join(main_dir, 'VHR-Stacks') # Required Before Run
    trainDir = os.path.join(main_dir, 'TrainingData') # Required Before Run
    modelDir = os.path.join(main_dir, 'Models')
    classifiedDir = os.path.join(main_dir, 'Classified')
    logDir = os.path.join(main_dir, 'Logs')

    dir_dictionary = {}

    for req in [trainDir]:#[VHRdir, trainDir]:
        garbage, dirname = req.rsplit('/', 1)
        if os.path.isdir(req):
            dir_dictionary[dirname] = req
            continue
        print("MISSING REQUIRED DIRECTORY: {}".format(req))
        sys.exit()
    dir_dictionary['VHR-Stacks'] = VHRdir

    for directory in [modelDir, classifiedDir ,logDir]:
        os.system("mkdir -p {}".format(directory))
        garbage, dirname = directory.rsplit('/', 1)
        dir_dictionary[dirname] = directory


    return dir_dictionary


#############################################################################


def getparser():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument("-w", "--work-directory", type=str, required=True, dest='workdir',
                        default="", help="Specify working directory")
    parser.add_argument("-m", "--model", type=str, required=True, dest='model',
                        default="", help="Specify model filename that will be saved or evaluated")
    parser.add_argument('-b', '--bands', nargs='*', dest='bands', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        help='Specify number of bands.', required=True, type=int)
    parser.add_argument('-bn', '--band-names', nargs='*', dest='band_names', help='Specify number of bands.', 
                        required=True, type=int, ['Coastal Blue', 'Blue', 'Green', 'Yellow', 'Red', 'Red Edge', 
                        'Near-IR1', 'Near-IR2', 'DVI', 'FDI', 'SI'])
    # Train
    parser.add_argument("-c", "--csv", type=str, required=False, dest='csvdata',
                        default="", help="Specify CSV file to train the model.")
    parser.add_argument("-t", "--n-trees", type=int, required=False, dest='n_trees',
                        default=20, help="Specify number of trees for random forest model.")
    parser.add_argument("-f", "--max-features", type=str, required=False, dest='max_feat',
                        default='log2', help="Specify random forest max features.")
    # Evaluate
    parser.add_argument("-c", "--csv", type=str, required=False, dest='csvdata',
                        default="", help="CSV file to train the model.")
                               

    return parser.parse_args()
    


#############################################################################


def main():

    start_time = time() # record start time
    args = getparser()  # initialize arguments parser
    
    print ('Initializing script with the following parameters')
    print(f'Working Directory: {args.workdir}')
    print(f'Working Directory: {args.workdir}')
    
    """
    parser.add_argument("-w", "--work-directory", type=str, required=True, dest='workdir',
                        default="", help="Specify working directory")
    parser.add_argument("-m", "--model", type=str, required=True, dest='model',
                        default="", help="Specify model filename that will be saved or evaluated")
    parser.add_argument('-b', '--bands', nargs='*', dest='bands', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                        help='Specify number of bands.', required=True, type=int)
    parser.add_argument('-bn', '--band-names', nargs='*', dest='band_names', help='Specify number of bands.', 
                        required=True, type=int, ['Coastal Blue', 'Blue', 'Green', 'Yellow', 'Red', 'Red Edge', 
                        'Near-IR1', 'Near-IR2', 'DVI', 'FDI', 'SI'])
    # Train
    parser.add_argument("-c", "--csv", type=str, required=False, dest='csvdata',
                        default="", help="Specify CSV file to train the model.")
    parser.add_argument("-t", "--n-trees", type=int, required=False, dest='n_trees',
                        default=20, help="Specify number of trees for random forest model.")
    parser.add_argument("-f", "--max-features", type=str, required=False, dest='max_feat',
                        default='log2', help="Specify random forest max features.")
    # Evaluate
    parser.add_argument("-c", "--csv", type=str, required=False, dest='csvdata',
                        default="", help="CSV file to train the model.")
    
    """
    print("Elapsed Time: ", (time() - start_time) / 60.0) # output program run time
    
    """
    
    args = arg_parser_train()
    print('Initializing model with parameters:')
    print('# GPUs: {}'              .format(args.gpu_devs))
    print('Data filenames: {}'      .format(args.x))
    print('Labels filenames: {}'    .format(args.y))

    # Get Arguments
    parser = getparser()
    args = parser.parse_args()

    main_dir = args.directory
    train_csv = args.train
    model = args.apply

    # Set Up Directories
    dir_dict = create_directories(main_dir)
      
    # Log processing:
    logfile = os.path.join(dir_dict['Logs'], 'Log_{}trees_{}.txt'.format(n_trees, max_feat))
    print('See ', logfile)
    so = se = open(logfile, 'w') # open our log file
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w') # re-open stdout without buffering
    os.dup2(so.fileno(), sys.stdout.fileno()) # redirect stdout and stderr to the log file opened above
    os.dup2(se.fileno(), sys.stderr.fileno())
    
    # TRAIN MODEL FROM CSV
    if train_csv:

        #catalogid = csv.strip('.csv').strip('trainingdata_')
        catalogid = '0002'
        print('TRAINING MODEL FOR IMAGE: {}\n\n\n'.format(catalogid))
        
        #Before calling the model train or apply, read and configure the inputs into test and train 
        print('Input Text: ', train_csv)

        (X_train, X_test, y_train, y_test) = get_test_training_sets(train_csv)
        print('Y_TRAIN LENGTH: {}\nY_TEST LENGTH: {}'.format(len(y_train), len(y_test)))
        
        # TRAIN MODEL:
        print("Building model with n_trees={} and max_feat={}...".format(n_trees, max_feat))
        model_save = train_model(X_train, y_train, dir_dict['Models'], n_trees, max_feat, catalogid)
        print(model_save)
        
    #model_save = os.path.join(modelDir, 'model_{}_20_log2.pkl'.format(extentName))


    # APPLY MODEL
    if model:
        print("\nApplying model to rest of imagery\n")
        image_list = glob.glob(os.path.join(dir_dict["VHR-Stacks"], '*tif'))#os.listdir(dir_dict["VHR-Stacks"])
        
        for stack in image_list:
            #catalogid = stack[19:35]
            #model = 'model_{}_{}.pkl'.format(extentName, catalogid)
            #model = 'model_{}_{}.pkl'.format(extentName, 'all')
            #model_save = os.path.join(modelDir, model)

            print('Stack: {}\nModel: {}\n\n '.format(stack, model))
            VHRstack = os.path.join(dir_dict['VHR-Stacks'], stack) # Stack of VHR data (Multispec and Pan)
            apply_model(VHRstack, dir_dict['Classified'], model)
    
    #run_diagnostics(model_save, X_test, y_test)

    """


# -----------------------------------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":

    main()
