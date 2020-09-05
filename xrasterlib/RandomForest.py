# -------------------------------------------------------------------------------
# Import System Libraries
# -------------------------------------------------------------------------------
import os
import joblib
from tqdm import tqdm
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split  # train/test data split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from hummingbird.ml import convert  # support GPU inference
import torch  # import torch to verify available devices

from Raster import Raster

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch, Code 587"
__email__  = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"
# -------------------------------------------------------------------------------
# class Indices
# This class calculates remote sensing indices given xarray or numpy objects.
# -------------------------------------------------------------------------------


class RandomForest(Raster):

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(self, traincsvfile=None, modelfile=None):
        super().__init__()

        # working directory to store result artifacts
        self.resultsdir = 'results'

        # training csv filename
        if traincsvfile is not None and not os.path.isfile(traincsvfile):
            raise RuntimeError('{} does not exist'.format(traincsvfile))
        self.traincsvfile = traincsvfile

        # training and test data variables
        self.x_train = None
        self.x_test  = None
        self.y_train = None
        self.y_test  = None

        # training parameters
        self.n_trees = 20
        self.max_feat = 'log2'

        # trained model filename
        if modelfile is not None and not os.path.isfile(modelfile):
            raise RuntimeError('{} does not exist'.format(modelfile))
        elif modelfile is None and self.traincsvfile is not None:
            self.modelfile = 'model_{}_{}.pkl'.format(self.n_trees, self.max_feat)
        else:
            self.modelfile = modelfile

        # store loaded model
        self.model = None

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
        df = (pd.read_csv(self.traincsvfile, header=None, sep=',')).values  # generate pd dataframe
        x = df.T[0:-1].T.astype(str)
        y = df.T[-1].astype(str)
        self.x_train, self.x_test, \
            self.y_train, self.y_test = train_test_split(x, y, test_size=testsize, random_state=seed)

    def trainrf(self):
        labels = np.unique(self.y_train)  # now it's the unique values in y array from text file
        print(f'Training data includes {labels.size} classes.')
        print(f'X matrix is sized: {self.x_train.shape}')  # shape of x data
        print(f'Y array is sized:  {self.y_train.shape}')  # shape of y data

        if '.' not in labels[0]:  # if labels are integers, check first value from y (come as string)
            rf = RandomForestClassifier(n_estimators=self.n_trees, max_features=self.max_feat, oob_score=True)
            self.y_train = self.y_train.astype(np.int)
        else:  # if labels are floats, use random forest regressor
            rf = RandomForestRegressor(n_estimators=self.n_trees, max_features=self.max_feat, oob_score=True)
            self.y_train = self.y_train.astype(np.float)

        print("Training model...")
        rf.fit(self.x_train, self.y_train)  # fit model to training data
        print('Score:', rf.oob_score_)

        try:  # export model to file
            joblib.dump(rf, self.modelfile)
        except Exception as e:
            print(f'ERROR: {e}')

    def loadrf(self):
        self.model = joblib.load(self.modelfile)  # loading the model in parallel
        device = 'cpu'  # set cpu as default device
        if torch.cuda.is_available():  # if cuda is available, assign model to GPU
            device = torch.device('cuda:0')  # assign device
            self.model = convert(self.model, 'pytorch')  # convert model to tensors for GPU
            self.model.to(device)  # assign model to GPU
        print(f'Loaded model {self.modelfile} into {device}.')

    def predictrf(self, rastfile='Vietnam.tif', ws=[5120, 5120]):
        # open rasters and get both data and coordinates
        rast_shape = self.data[0, :, :].shape  # getting the shape of the wider scene
        print ("rasy shape ", rast_shape)
        wsx, wsy = ws[0], ws[1]  # chunking and doing in memory sliding window predictions

        # if the window size is bigger than the image, ignore and predict full image
        if wsx > rast_shape[0]:
            wsx = rast_shape[0]
        if wsy > rast_shape[1]:
            wsy = rast_shape[1]

        self.prediction = np.zeros(rast_shape)  # crop out the window for prediction
        print(f'Window Size: {wsx} x {wsy}. Final prediction initial shape: {self.prediction.shape}')

        for sx in tqdm(range(0, rast_shape[0], wsx)):  # iterate over x-axis
            for sy in range(0, rast_shape[1], wsy):  # iterate over y-axis
                x0, x1, y0, y1 = sx, sx + wsx, sy, sy + wsy  # assign window indices
                if x1 > rast_shape[0]:  # if selected x-indices exceeds boundary
                    x1 = rast_shape[0]  # assign boundary to x-window
                if y1 > rast_shape[1]:  # if selected y-indices exceeds boundary
                    y1 = rast_shape[1]  # assign boundary to y-window

                window = self.data[:, x0:x1, y0:y1]  # get window
                window = window.stack(z=('y', 'x'))  # stack y and x axis
                window = window.transpose("z", "band").values  # reshape xarray, return numpy arr
                self.prediction[x0:x1, y0:y1] = (self.model.predict(window)).reshape((x1 - x0, y1 - y0))

        # save raster
        self.prediction = self.prediction.astype(np.int16)  # change type of prediction to int16
