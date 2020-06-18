# created 1/29/2016

# purpose: To build a Random Forest model with an input csv and then save/apply model to a raster
# Inputs: input CSV, Raster Stack



# Import GDAL, NumPy, and matplotlib
import sys, os
from osgeo import gdal, gdal_array
import numpy as np
#import matplotlib.pyplot as plt
##%matplotlib inline # IPython
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
#import skimage.io as io
from timeit import default_timer as timer

# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()

##import gdal
##from osgeo.gdal import *
gdal.UseExceptions() # enable exceptions to report errors
drvtif = gdal.GetDriverByName("GTiff")

n_trees = 20
max_feat = 'log2'
modelName = '{}_{}'.format(n_trees, max_feat) # to distinguish between parameters of each model/output


extentName = sys.argv[1] # model number or name (can be blank) so we can save to different directories

# Hardcode for now:
bands = [1, 2, 3, 4, 5] # bands of the image stack #ANDREW
band_names = ['Blue', 'Green', 'Red', 'NIR', 'NDVI'] # the order of the stack #ANDREW


def find_elapsed_time(start, end): # example time = round(find_elapsed_time(start, end),3) where start and end = timer()
    elapsed_min = (end-start)/60
    return float(elapsed_min)

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
    img = np.zeros((nrows, ncols, img_ds.RasterCount), gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    for b in range(img.shape[2]): # the 3rd index of img.shape gives us the number of bands in the stack
        print '\nb: {}'.format(b)
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray() # GDAL is 1-based while Python is 0-based

    return (img, imgProperties)

"""Function to write final classification to tiff"""
def array_to_tif(inarr, outfile, imgProperties):

    # get properties from input
    (gt, proj, ncols, nrows, ndval) = imgProperties
    print ndval

    drv = drvtif.Create(outfile, ncols, nrows, 1, 3, options = [ 'COMPRESS=LZW' ]) # 1= number of bands (i think) and 3 = Data Type (16 bit signed)
    drv.SetGeoTransform(gt)
    drv.SetProjection(proj)
    drv.GetRasterBand(1).SetNoDataValue(ndval)
    drv.GetRasterBand(1).WriteArray(inarr)

    return outfile


"""Function to run diagnostics on model"""
def run_diagnostics(model_save, X, y): # where model is the model object, X and y are training sets

    # load model for use:
    print "\nLoading model from {} for cross-val".format(model_save)
    model_load = joblib.load(model_save) # nd load

    print "\n\nDIAGNOSTICS:\n"

    try:
        print "n_trees = {}".format(n_trees)
        print "max_features = {}\n".format(max_feat)
    except Exception as e:
        print "ERROR: {}\n".format(e)

    # check Out of Bag (OOB) prediction score
    print 'Our OOB prediction of accuracy is: {}\n'.format(model_load.oob_score_ * 100)
    print "OOB error: {}\n".format(1 - model_load.oob_score_)



    # check the importance of the bands:
    for b, imp in zip(bands, model_load.feature_importances_):
        print 'Band {b} ({name}) importance: {imp}'.format(b=b, name=band_names[b-1], imp=imp)
    print ''


    """
    # see http://scikit-learn.org/stable/modules/cross_validation.html for how to use rf.score etc
    """

    # dont really know if this is applicable for 2 classes but try it anyway:
    # Setup a dataframe -- just like R
    df = pd.DataFrame() #**** Need to create a new y with validation points, like we did with y in the function below (make roi be valid sites array instead of training)
    df['truth'] = y
    df['predict'] = model_load.predict(X)

    # Cross-tabulate predictions
    print pd.crosstab(df['truth'], df['predict'], margins=True)

##    print "Other:"
##    print model.criterion
##    print model.estimator_params
##    print model.score
##    print model.feature_importances_
##    print ''


def apply_model(VHRstack, classDir, model_save): # VHR stack we are applying model to, output dir, and saved model


    (img, imgProperties) = stack_to_obj(VHRstack)
    (gt, proj, ncols, nrows, ndval) = imgProperties # ndval is nodata val of image stack not sample points
    
    """
    print img
    print img.shape
    print np.unique(img)
    """
    
    # Classification of img array and save as image (5 refers to the number of bands in the stack)
    # reshape into long 2d array (nrow * ncol, nband) for classification
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])
    img_as_array = img[:, :, :img.shape[2]].reshape(new_shape) # 5 is number of layers
##    print img_as_array.shape # (192515625, 5)
##    print np.unique(img_as_array) # [ -999  -149  -146 ..., 14425 14530 14563]

    print 'Reshaped from {o} to {n}'.format(o=img.shape, n=img_as_array.shape)

    print "\nLoading model from {}".format(model_save)
    model_load = joblib.load(model_save) # nd load
    print "\nModel information:\n{}".format(model_load)

    # Now predict for each pixel
    print "\nPredicting model on image array"
    class_prediction = model_load.predict(img_as_array)

    #* at some point may need to convert values that were -999 in img array back to -999, depending on what rf does to those areas

##    print img[:, :, 0].shape # (13875, 13875)
##    print img[:, :, 0]

    # Reshape our classification map and convert to 16-bit signed
    class_prediction = class_prediction.reshape(img[:, :, 0].shape).astype(np.int16)

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

    # export classificaiton to tif
    classification = os.path.join(classDir, "{}__{}__classified.tif".format(extentName, modelName))
    array_to_tif(class_prediction, classification, imgProperties)
##    io.imsave(classification, class_prediction)
    print "\nWrote map output to {}".format(classification)



    return


"""Function for training the model using training data"""
# To train the model, you need: input text file/csv, model output location, model parameters
def train_model(X, y, modelDir, n_trees, max_feat):


    n_samples = np.shape(X)[0] # the first index of the shape of X will tell us how many sample points
    print '\nWe have {} samples'.format(n_samples)

    labels = np.unique(y) # now it's the unique values in y array from text file
    print 'The training data include {n} classes: {classes}'.format(n=labels.size, classes=labels)


    print 'Our X matrix is sized: {sz}'.format(sz=X.shape) # hopefully will be (?, 4)
    print 'Our y array is sized: {sz}'.format(sz=y.shape)


    """ Now train the model """
    print "\nInitializing model..."
    rf = RandomForestClassifier(n_estimators=n_trees, max_features=max_feat, oob_score=True) # can change/add other settings later

    print "\nTraining model..."
    rf.fit(X, y) # fit model to training data

    print 'score:', rf.oob_score_

    # Export model:
    try:
        model_save = os.path.join(modelDir, "model_{}_{}.pkl".format(extentName, modelName))
        joblib.dump(rf, model_save)
    except Exception as e:
        print "Error: {}".format(e)


    return model_save # Return output model for application and validation


def get_test_training_sets(inputCsv):

    with open(inputCsv, 'r') as it:
        cnt = 0
        for r in it.readlines():
            cnt += 1
            rline = r.strip('\n').strip().split(',')

            xx = np.array(rline[0:-1]) # xx is the line except the last entry (class)
            yy = np.array(rline[-1]) # yy is the last entry in the line

            if cnt == 1:
                X = [xx]
                y = yy
            else:
                X = np.vstack((X, xx))
                y = np.append(y, yy)

    # Convert y array to integer type
    y = y.astype(float).astype(int)
    
    # Now we have X and y, but this is not split into validation and training. Do that here:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state = 21)

    return (X_train, X_test, y_train, y_test)

"""Function to convert stack and sample imagery into X and y [not yet split into training/validation]"""
def convert_img_to_Xy_data(VHRstack, sampleTif): # Given a data stack and sample tiff, we can convert these into X and y for Random Forests

    """ Read in the raster stack and training data to numpy array: """
    img_ds = gdal.Open(VHRstack, gdal.GA_ReadOnly) # GDAL dataset
    roi_ds = gdal.Open(sampleTif, gdal.GA_ReadOnly)

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
        print '\nb: {}'.format(b)
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray() # GDAL is 1-based while Python is 0-based

    """ Read Training dataset into numpy array """
    roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
    roi_nd = roi_ds.GetRasterBand(1).GetNoDataValue() # get no data value of training dataset
    roi[roi==roi_nd] = 0 # Convert the No Data pixels in raster to 0 for the model

    X = img[roi > 0, :]  # Data Stack pixels
    y = roi[roi > 0]     # Training pixels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 21)

    return (X_train, X_test, y_train, y_test)

#def main():

start = timer()
print "Running"
# Set up directories:
ddir = '/att/gpfsfs/briskfs01/ppl/mwooten3/MCC/RandomForests'
testdir = os.path.join(ddir, 'test')
VHRdir = testdir
trainDir = testdir
modelDir = testdir
classDir = testdir
logDir   = testdir

"""
for d in [modelDir, classDir, logDir]:
    os.system('mkdir -p {}'.format(d))
print "Running"

# Log processing:
logfile = os.path.join(logDir, 'Log_{}trees_{}.txt'.format(n_trees, max_feat))
so = se = open(logfile, 'w', 0) # open our log file
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0) # re-open stdout without buffering
os.dup2(so.fileno(), sys.stdout.fileno()) # redirect stdout and stderr to the log file opened above
os.dup2(se.fileno(), sys.stderr.fileno())
"""


# Get the raster stack and sample data
VHRstack = os.path.join(VHRdir, 'WV02_20170427_M1BS_103001006855B300-toa__stack.tif') # Stack of VHR data (Multispec and Pan)
##    trainTif = os.path.join(trainDir, 'training__{}.tif'.format(extentName)) # Training Data
##    validTif = os.path.join(trainDir, 'valid__{}.tif'.format(extentName)) # Validation Data



# IF input is csv file, use this method
"""Before calling the model train or apply, read and configure the inputs into test and train """
inCsv = os.path.join(trainDir, 'WV02_20170427_M1BS_103001006855B300-toa__training.csv')
(X_train, X_test, y_train, y_test) = get_test_training_sets(inCsv)


# Train and apply models:
print "Building model with n_trees={} and max_feat={}...".format(n_trees, max_feat)
model_save = train_model(X_train, y_train, modelDir, n_trees, max_feat)

print "\nApplying model to rest of imagery"
apply_model(VHRstack, classDir, model_save)
run_diagnostics(model_save, X_test, y_test)

elapsed = round(find_elapsed_time(start, timer()),3)

print "\n\nElapsed time = {}".format(elapsed)

#main()





