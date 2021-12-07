import os
import glob
import logging
from tqdm import tqdm

import joblib
import numpy as np
import pandas as pd
import xarray as xr

from terragpu.engine import array_module, df_module

xp = array_module()
xf = df_module()

#try:
#    # import cupy as cp
#    # import cudf as cf
#    # from cuml.ensemble import RandomForestClassifier as RFC
#    # from cuml.ensemble import RandomForestRegressor as RFR
#    # from cuml.model_selection import train_test_split
#    #cp.random.seed(seed=None)
#    HAS_GPU = True

#except ImportError:
#    # from sklearn.ensemble import RandomForestClassifier as RFC
#    # from sklearn.ensemble import RandomForestRegressor as RFR
##    # from sklearn.model_selection import train_test_split
#    # from sklearn.metrics import accuracy_score
#    HAS_GPU = False

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"


# -------------------------------------------------------------------------------
# class RF
# This class performs training and classification of satellite imagery using a
# Random Forest classifier.
# -------------------------------------------------------------------------------
class RF(object):

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(
            self, train_csv: str = None, dataset_metadata: dict = None,
            model_metadata: str = None, model_filename: str = None,
            output_dir: str = 'results', predict_dir: str = None):

        # Working directory to store result artifacts
        self.output_dir = output_dir
        self.predict_dir = predict_dir
        self.model_filename = model_filename

        # Dataframe object to store dataset creation
        self.dataset_metadata = dataset_metadata
        self.model_metadata = model_metadata
        self.data_df = None

        # Training data CSV
        self.train_csv = train_csv

        # training parameters
        #self.n_trees = n_trees
        #self.max_features = max_features

        # training and test data variables, initialize them as empty
        #self.x_train = None
        #self.x_test = None
        #self.y_train = None
        #self.y_test = None

        # TODO: this needs some help ---- trained model filename
        #if train_csv is None and model_filename is not None \
        #        and not os.path.isfile(model_filename):
        #    raise RuntimeError(f'{model_filename} does not exist')
        #elif model_filename is None and train_csv is not None:
        #    self.modelfile = f'model_{n_trees}_{max_features}.pkl'
        #else:  # if a model name is given
        #    self.model_filename = model_filename

        # store loaded model - load model if filename exists
        #self.model = None
        #self.model_nfeat = None

        # store prediction if required
        #self.prediction = None

    # ---------------------------------------------------------------------------
    # external methods
    # ---------------------------------------------------------------------------
    def preprocess(self):
        """
        Preprocess Step - generate dataset
        """
        logging.info('Starting preprocess pipeline step...')

        # -----------------------------------------------------------------------
        # Validate dataset existance
        # -----------------------------------------------------------------------
        images_list = sorted(glob.glob(self.dataset_metadata.images_dir))
        labels_list = sorted(glob.glob(self.dataset_metadata.labels_dir))
        assert len(images_list) > 0 and len(images_list) == len(labels_list), \
            'No train images found or mismatch between train and label files'

        # Create dataframe and list to store points
        list_points = []
        df_points = xf.DataFrame(
            columns=self.dataset_metadata.bands + ['CLASS'])

        # -----------------------------------------------------------------------
        # Iteration over each file and process
        # -----------------------------------------------------------------------
        for image, label in tqdm(zip(images_list, labels_list)):

            # Open imagery, default to GPU if available
            filename = image.split('/')[-1]
            image = xp.moveaxis(
                xp.asarray(xr.open_rasterio(image).values), 0, -1)
            label = xp.asarray(xr.open_rasterio(label).values)

            # Labels are expected to have 2D and start from 0
            label = xp.squeeze(label) if len(label.shape) != 2 else label
            label = label - 1 if xp.min(label) == 1 else label

            print(image[0, 0, :], label[0, 0])

            tdims = image.shape

            # Reshape into tabular format and convert to dataframe
            image = image.reshape(
                (tdims[0] * tdims[1], len(self.dataset_metadata.bands)))
            image = xf.DataFrame(image, columns=self.dataset_metadata.bands)

            label = xf.DataFrame(
                label.reshape((tdims[0] * tdims[1],)), columns=['CLASS'])

            # Move to dataframe format for preprocessing
            data_df = xf.concat([image, label], axis=1)
            print(data_df)


            # Iterate over the classes
            # for c in range(self.dataset_metadata.max_classes):

        """
            for c in range(max_classes):

                indices = 0
                selected_points = 0
                num_points = dataset_metadata[filename][str(c)]
                print(f'Class {str(c)} points to extract {str(num_points)}')

                x_indices, y_indices = np.where(label == c)  # we extract all class c points from the imagery
                x_indices, y_indices = shuffle(x_indices, y_indices)  # we make sure values are fully shuffled
                print(f"Class {c}:", x_indices.shape, y_indices.shape)

                if x_indices.shape[0] != 0:
                    try:
                        while selected_points < num_points:

                            sv, lv = image[:, x_indices[indices], y_indices[indices]], \
                                int(label[x_indices[indices], y_indices[indices]])

                            if sv[0] != nodata_val:

                                list_points.append(
                                    pd.DataFrame(
                                        [np.append(sv, [lv])],
                                        columns=list(df_points.columns))
                                )                    
                                selected_points += 1
                            else:
                                print("YES")
                            indices += 1
                    except IndexError:
                        pass

                print(selected_points)

        df_points = pd.concat(list_points)
        df_points.to_csv(output_filename, index=False)
        """
        return

    def train(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def visualize(self):
        raise NotImplementedError

    def cross_visualize(self):
        raise NotImplementedError

    # ---------------------------------------------------------------------------
    # internal methods
    # ---------------------------------------------------------------------------
    def _split_data(self, testsize=0.30, seed=21):
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
