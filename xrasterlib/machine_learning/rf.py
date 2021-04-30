import os
import joblib
import logging
from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split  # train/test data split
from sklearn.ensemble import RandomForestClassifier as sklRFC
from sklearn.ensemble import RandomForestRegressor as sklRFR
from sklearn.metrics import accuracy_score

try:
    import cupy as cp
    import cudf as cf
    from cuml.ensemble import RandomForestClassifier as cumlRFC
    from cuml.ensemble import RandomForestRegressor as cumlRFR
    cp.random.seed(seed=None)
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

from xrasterlib.raster import Raster

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"

# -------------------------------------------------------------------------------
# class RF
# This class performs training and classification of satellite imagery using a
# Random Forest classifier.
# -------------------------------------------------------------------------------


class RF(Raster):

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(self, traincsvfile=None, modelfile=None, outdir='results',
                 ntrees=20, maxfeat='log2'
                 ):
        super().__init__()

        # working directory to store result artifacts
        self.outdir = outdir

        # training csv filename
        if traincsvfile is not None and not os.path.isfile(traincsvfile):
            raise RuntimeError('{} does not exist'.format(traincsvfile))
        self.traincsvfile = traincsvfile

        # training parameters
        self.ntrees = ntrees
        self.maxfeat = maxfeat

        # training and test data variables, initialize them as empty
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        # trained model filename
        # TODO: update this with args.command option (train vs. classify)
        if traincsvfile is None and modelfile is not None \
                and not os.path.isfile(modelfile):
            raise RuntimeError('{} does not exist'.format(modelfile))

        elif modelfile is None and self.traincsvfile is not None:
            self.modelfile = 'model_{}_{}.pkl'.format(
                self.ntrees, self.maxfeat
            )
        else:  # if a model name is given
            self.modelfile = modelfile

        # store loaded model
        self.model = None
        self.model_nfeat = None

        # store prediction if required
        self.prediction = None

    # ---------------------------------------------------------------------------
    # methods
    # ---------------------------------------------------------------------------
    def splitdata(self, testsize=0.30, seed=21):
        """
        :param testsize: size of testing features
        :param seed: random state integer for reproducibility
        :return: 4 arrays with train and test data respectively
        """
        # read data and split into data and labels
        df = pd.read_csv(self.traincsvfile, header=None, sep=',')
        x = df.iloc[:, :-1].astype(np.float32)
        y = df.iloc[:, -1]
        # modify type of labels (int or float)
        if 'int' in str(y.dtypes):
            y = y.astype(np.int8)
        else:
            y = y.astype(np.float32)
        # split data into training and test
        self.x_train, self.x_test, \
            self.y_train, self.y_test = train_test_split(
                x, y, test_size=testsize, random_state=seed
            )
        del df, x, y

    def train(self):
        # TODO: Consider moving these print statements to logging
        print(f'Train data includes {np.unique(self.y_train).size} classes.')
        print(f'X matrix is sized: {self.x_train.shape}')  # shape of x data
        print(f'Y array is sized:  {self.y_train.shape}')  # shape of y data
        print(f'Training with ntrees={self.ntrees} and maxfeat={self.maxfeat}')

        if self.has_gpu:  # run using RAPIDS library
            # initialize cudf data and log into GPU memory
            print('Training model via RAPIDS.')
            self.x_train = cf.DataFrame.from_pandas(self.x_train)
            self.x_test = cf.DataFrame.from_pandas(self.x_test)
            self.y_train = cf.Series(self.y_train.values)

            # if labels are integers, use RF Classifier
            if 'int' in str(self.y_test.dtypes):
                rf_funct = cumlRFC
            else:  # if labels are floats, use RF Regressor
                rf_funct = cumlRFR

        # run only using CPU resources and Sklearn
        else:
            print('Training model via SKLearn.')
            # if labels are integers, use RF Classifier
            if 'int' in str(self.y_test.dtypes):
                rf_funct = sklRFC
            else:  # if labels are floats, use RF Regressor
                rf_funct = sklRFR

        # Initialize model
        rf_model = rf_funct(
            n_estimators=self.ntrees,
            max_features=self.maxfeat
        )

        # fit model to training data and predict for accuracy score
        rf_model.fit(self.x_train, self.y_train)

        if self.has_gpu:
            score = accuracy_score(
                self.y_test, rf_model.predict(self.x_test).to_array()
            )
        else:
            score = accuracy_score(
                self.y_test, rf_model.predict(self.x_test)
            )
        print(f'Training accuracy: {score}')

        try:  # export model to file
            outmodel = self.outdir + '/' + self.modelfile
            joblib.dump(rf_model, outmodel)
            # print(f'Model has been saved as {outmodel}')
            logging.info(f'Model has been saved as {outmodel}')
        except Exception as e:
            logging.error(f'ERROR: {e}')

    def load(self):
        self.model = joblib.load(self.modelfile)  # loading pkl in parallel
        # self.model_nfeat = self.model.n_features_  # model features
        print(f'Loaded model {self.modelfile}.')

    def predict(self, ws=[5120, 5120]):
        # open rasters and get both data and coordinates
        rast_shape = self.data[0, :, :].shape  # shape of the wider scene
        wsx, wsy = ws[0], ws[1]  # in memory sliding window predictions

        # if the window size is bigger than the image, predict full image
        if wsx > rast_shape[0]:
            wsx = rast_shape[0]
        if wsy > rast_shape[1]:
            wsy = rast_shape[1]

        self.prediction = np.zeros(rast_shape)  # crop out the window
        print(f'wsize: {wsx}x{wsy}. Prediction shape: {self.prediction.shape}')

        for sx in tqdm(range(0, rast_shape[0], wsx)):  # iterate over x-axis
            for sy in range(0, rast_shape[1], wsy):  # iterate over y-axis
                x0, x1, y0, y1 = sx, sx + wsx, sy, sy + wsy  # assign window
                if x1 > rast_shape[0]:  # if selected x exceeds boundary
                    x1 = rast_shape[0]  # assign boundary to x-window
                if y1 > rast_shape[1]:  # if selected y exceeds boundary
                    y1 = rast_shape[1]  # assign boundary to y-window

                window = self.data[:, x0:x1, y0:y1]  # get window
                window = window.stack(z=('y', 'x'))  # stack y and x axis
                window = window.transpose("z", "band").values  # reshape
                # perform sliding window prediction
                self.prediction[x0:x1, y0:y1] = \
                    self.model.predict(window).reshape((x1 - x0, y1 - y0))
        # save raster
        self.prediction = self.prediction.astype(np.int8)  # type to int16


# -------------------------------------------------------------------------------
# class RF Unit Tests
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    # Running Unit Tests
    print("Unit tests below")
