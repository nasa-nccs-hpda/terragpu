import os
from glob import glob
import logging
from tqdm import tqdm
from pathlib import Path

import dask
import joblib
import numpy as np
import pandas as pd
import xarray as xr

from sklearn.utils import shuffle
from terragpu import io
from terragpu.engine import array_module, df_module
from sklearn.metrics import accuracy_score

xp = array_module()
xf = df_module()

if xp.__name__ == 'cupy':
    from cuml.model_selection import train_test_split
    from cuml.ensemble import RandomForestClassifier
else:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

# -------------------------------------------------------------------------------
# class RF
# This class performs training and classification of satellite imagery using a
# Random Forest classifier.
# - TODO: labels as a shapefile (as points in the imagery)
# - TODO: future release, convert raster to dataframe, filter and extract points
# -------------------------------------------------------------------------------
class RF(object):

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(
            self,
            data_dir: str = None,
            labels_dir: str = None,
            inference_dir: str = None,
            csv_filename: str = 'training.csv',
            n_classes: int = None,
            n_points: int = 10000,
            train_size: float = 0.70,
            n_estimators: int = 20,
            max_features: str = 'log2',
            model_filename: str = 'output.pkl',
            output_dir: str = 'output',
            inference_output_dir: str = 'output',
            window_size: list = [5120, 5120],
            output_nodata: int = -17,
            bands: list = ['CB', 'B', 'G', 'Y', 'R', 'RE', 'NIR1', 'NIR2']
        ):

        # Set data directory values
        self.data_dir = data_dir
        self.labels_dir = labels_dir
        self.inference_dir = inference_dir
        self.csv_filename = csv_filename

        # Output filenames
        self.csv_filename = csv_filename
        self.output_dir = output_dir
        self.inference_output_dir = inference_output_dir
        self.model_filename = os.path.join(
            self.output_dir, model_filename)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.inference_output_dir, exist_ok=True)

        # Set some metadata
        self.bands = bands
        self.n_classes = n_classes
        self.n_points = n_points
        self.train_size = train_size
        
        # Set model hyperparameters
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.window_size = window_size
        self.output_nodata = output_nodata

        assert self.n_classes is not None, 'n_classes must be set to an integer.'

    def preprocess(self):
        """
        Preprocessing pipeline step.
        """
        logging.info("Preprocessing stage, generating CSV file from points dataset.")

        assert self.data_dir is not None and self.labels_dir is not None, \
            'data_dir and labels_dir cannot be None for preprocessing.'

        # Identify filenames from directory
        self.data_filenames = sorted(glob(self.data_dir))
        self.label_filenames = sorted(glob(self.labels_dir))
        
        assert len(self.data_filenames) == len(self.label_filenames) and \
            self.data_filenames, 'Number of data and label files do not match.'

        # Create dataframe and list to store points
        list_points = []
        df_points = xf.DataFrame(columns=self.bands + ['CLASS'])

        # Attempt to specify number of points per class per image
        total_points = int(
            self.n_points / len(self.data_filenames) / self.n_classes)

        # Iterate over data and label filenames
        for image,label in zip(self.data_filenames, self.label_filenames):

            # Open imagery
            filename = image.split('/')[-1]
            image = io.imread(image)
            label = io.imread(label)
            nodata_val = image._FillValue

            image = image.data.compute()
            label = label.data.compute()

            assert image.shape[0] == len(self.bands), \
                f"Image bands: {image.shape[0]} do not match bands: {self.bands}"

            # Some preprocessing
            label = xp.squeeze(label) if len(label.shape) != 2 else label
            
            # Iterate over the imagery and select the points
            for class_id in range(self.n_classes):

                coords = 0
                selected_points = 0
                logging.info(f'Class {class_id} points to extract {total_points}')                

                # We extract all class c points from the imagery
                x_coords, y_coords = xp.where(label == class_id)

                # We make sure values are fully shuffled
                x_coords, y_coords = shuffle(x_coords, y_coords)
                logging.info(f"Class {class_id}:", x_coords.shape, y_coords.shape)

                # Only run if class is present in the imagery
                if x_coords.shape[0] == 0:
                    logging.info(f'Class {class_id} not present in the {filename}, skipping.')
                    continue

                # After confirming the class is present in the imagery
                try:
                    
                    # Counter for the total number of points to extract
                    while selected_points < total_points:

                        # Stored data and label value
                        sv, lv = image[:, x_coords[coords], y_coords[coords]], \
                            int(label[x_coords[coords], y_coords[coords]])

                        # If no data is present, ignore
                        if sv[0] != nodata_val:
                            
                            # Append to the dataframe
                            list_points.append(
                                xf.DataFrame(
                                    [xp.append(sv, [lv])],
                                    columns=list(df_points.columns))
                            )

                            # Increase selected points counter          
                            selected_points += 1
                        
                        # Increase index value for the total iterations
                        coords += 1
                except IndexError:
                    logging.info("Iterated over all points, could not find the {total_points} points")
                    pass                

        # Concatenate all points and output training CSV
        df_points = xf.concat(list_points)
        df_points.to_csv(self.csv_filename, index=False)
        return

    def train(self):
        """
        Training pipeline step.
        """
        # Validate the existance of a training file and no nodata values
        assert os.path.exists(self.csv_filename), f'{self.csv_filename} not found.'
        data_df = xf.read_csv(self.csv_filename, sep=',')
        assert not data_df.isnull().values.any(), f'Na found: {self.csv_filename}'

        logging.info(f'Loading dataset from: {self.csv_filename}')

        # Shuffle and split the dataset
        data_df = data_df.sample(frac=1).reset_index(drop=True)

        # dask_cudf does not support iloc operations, the objects gett converted to plain cudf
        x = data_df.iloc[:, :-1].astype(xp.float32)
        y = data_df.iloc[:, -1].astype(xp.int32)

        # Split training and test datasets
        if xp.__name__ == 'cupy':
            self.x_train, self.x_test, self.y_train, self.y_test = \
                train_test_split(x, y, train_size=self.train_size)
        else:
            self.x_train, self.x_test, self.y_train,self.y_test = \
                train_test_split(x, y, test_size=1.0 - self.train_size)

        del data_df, x, y
        logging.info(f'x_train: {self.x_train.shape[0]} elements')
        logging.info(f'x_test:  {self.x_test.shape[0]} elements')
        logging.info(f'y_train: {self.y_train.shape[0]} elements')
        logging.info(f'y_test:  {self.y_test.shape[0]} elements')

        # Perform training step
        rf_model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_features=self.max_features
        )
        rf_model.fit(self.x_train, self.y_train)

        # Save model output
        if xp.__name__ == 'cupy':
            score = accuracy_score(
                self.y_test.to_array(),
                rf_model.predict(self.x_test).to_array())
        else:
            score = accuracy_score(
                self.y_test, rf_model.predict(self.x_test))
        logging.info(f'Training accuracy: {score}')

        # Export model to filename
        try:
            joblib.dump(rf_model, self.model_filename)
            logging.info(f'Model has been saved as {self.model_filename}')
        except Exception as e:
            logging.error(f'ERROR: {e}')
        
        return

    def predict(self):
        """
        Prediction pipeline step.
        """
        assert self.inference_dir is not None, \
            'inference_dir cannot be None for preprocessing.'

        # Load pickled model into memory
        self.get_model()

        # Identify filenames from directory
        self.inference_filenames = sorted(glob(self.inference_dir))
        
        assert len(self.inference_filenames) > 0, \
            'Number of data and label files do not match.'

        for predict_filename in self.inference_filenames:
 
            # Read image into array
            image = io.imread(predict_filename)

            # Perform prediction
            prediction = self._predict_sliding_window(image)

            # Save image to disk
            output_filename = os.path.join(
                self.inference_output_dir,
                f'{Path(predict_filename).stem}.pred.tif'
            )

            # Save to disk
            io.imsave(prediction, output_filename)
        return


    def _predict_sliding_window(self, image):
        """
        Underlying prediction step for _predict_sliding_window parallelization.
        """
        # open rasters and get both data and coordinates
        temp_slicing = image[0, :, :]
        rast_shape = temp_slicing.shape  # shape of the wider scene
        
        # in memory sliding window predictions
        wsy, wsx = self.window_size[0], self.window_size[1]

        # if the window size is bigger than the image, predict full image
        if wsy > rast_shape[0]:
            wsy = rast_shape[0]
        if wsx > rast_shape[1]:
            wsx = rast_shape[1]

        prediction = xr.DataArray(
            xp.zeros(rast_shape),
            name='prediction',
            coords=temp_slicing.coords,
            dims=temp_slicing.dims,
            attrs=temp_slicing.attrs
        )
        prediction.attrs['_FillValue'] = self.output_nodata
        logging.info(f'wsize: {wsx}x{wsy}. Prediction shape: {prediction.shape}')

        for sy in tqdm(range(0, rast_shape[0], wsy)):  # iterate over x-axis
            for sx in range(0, rast_shape[1], wsx):  # iterate over y-axis
                y0, y1, x0, x1 = sy, sy + wsy, sx, sx + wsx  # assign window
                if y1 > rast_shape[0]:  # if selected x exceeds boundary
                    y1 = rast_shape[0]  # assign boundary to y-window
                if x1 > rast_shape[1]:  # if selected x exceeds boundary
                    x1 = rast_shape[1]  # assign boundary to y-window

                window = image[:, y0:y1, x0:x1]  # get window
                window = window.stack(z=('y', 'x'))  # stack y and x axis
                window = window.transpose("z", "band").data.compute()

                 # perform sliding window prediction
                prediction[y0:y1, x0:x1] = \
                    self.model.predict(window).reshape((y1 - y0, x1 - x0))
                
                window = window[:, 1].reshape((y1 - y0, x1 - x0))
                prediction.data[y0:y1, x0:x1][
                    window == temp_slicing.rio.nodata] = self.output_nodata
        
        # return raster
        return prediction.astype(xp.int8)

    def all(self):
        """
        Execute all pipeline steps.
        """
        self.preprocess()  # preprocessing
        self.train()  # training
        self.predict()  # predict
        return
        
    def get_model(self):
        assert os.path.exists(self.model_filename), f'{self.model_filename} not found.'
        self.model = joblib.load(self.model_filename)  # loading pkl in parallel
        logging.info(f'Loaded model {self.model_filename}.')
        return


if __name__ == '__main__':

    random_forest = RF(
        data_dir = '/adapt/nobackup/projects/ilab/projects/Senegal/LCLUC_Senegal_Cloud/example_data/images/*.tif',
        labels_dir = '/adapt/nobackup/projects/ilab/projects/Senegal/LCLUC_Senegal_Cloud/example_data/labels/*.tif',
        #inference_dir = '/adapt/nobackup/projects/ilab/projects/Senegal/LCLUC_Senegal_Cloud/example_data/images/*.tif',
        inference_dir='/att/nobackup/mwooten3/Senegal_LCLUC/VHR/priority-tiles/Konrad-tiles/M1BS/WV03_20200214_M1BS_104001005741C600-toa.tif',
        n_points = 1000,
        n_classes = 2
    )
    random_forest.all()
