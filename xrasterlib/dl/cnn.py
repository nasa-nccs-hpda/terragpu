import gc
from datetime import datetime
import logging
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split  # train/test data split
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from tensorflow.keras.callbacks import ReduceLROnPlateau

try:
    import cupy as cp
    cp.random.seed(seed=None)
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from xrasterlib.raster import Raster
from xrasterlib.utils import read_config

from xrasterlib.dl.loss import tanimoto_dual_loss
from xrasterlib.dl.loss import dice_coef
from xrasterlib.dl.loss import dice_coef_bin
from xrasterlib.dl.loss import focal_loss_cat, focal_loss_bin
from xrasterlib.dl.loss import ce_dl_bin, jaccard_distance_loss

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Development"

# -------------------------------------------------------------------------------
# class CNN
# This class performs training and classification of satellite imagery using a
# Convolutional Neural Networks.
# -------------------------------------------------------------------------------


class CNN(Raster):

    # ---------------------------------------------------------------------------
    # __init__
    # ---------------------------------------------------------------------------
    def __init__(self, config=None):
        super().__init__()

        # assert of configuration file was not given
        assert config is not None, "Config file must be provided."
        self.cfg = read_config(config)

        self.cfg_data = self.cfg['Data']
        self.cfg_prep = self.cfg['Preprocess']
        self.cfg_train = self.cfg['Train']
        self.cfg_pred = self.cfg['Predict']

        # training and test data variables, initialize them as empty
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        # Train hyperparameters
        self.bsize = int(self.cfg_train['bsize'])
        self.epochs = int(self.cfg_train['epochs'])
        self.startepoch = int(self.cfg_train['startepoch'])

        # Train parameters
        self.modelsave = self.cfg_train['modelsave']
        self.savedir = self.cfg_train['savedir']
        self.get_loss()
        self.get_metrics()
        self.get_callbacks()

        # store prediction if required
        self.prediction = None

    # ---------------------------------------------------------------------------
    # methods
    # ---------------------------------------------------------------------------

    # ---------------------------------------------------------------------------
    # input
    # ---------------------------------------------------------------------------
    def nploader(self, fname):
        return np.load(fname, allow_pickle=True)

    def get_loss(self):
        """
        Get loss function.
        """
        losses = {
            'categorical_crossentropy': 'categorical_crossentropy',
            'ce_dl_bin': ce_dl_bin,
            'dice': dice_coef,
            'dice_bin': dice_coef_bin,
            'focal': focal_loss_cat(alpha=.25, gamma=2),
            'focal_bin': focal_loss_bin(alpha=.25, gamma=2),
            'jaccard': jaccard_distance_loss,
            'tanimoto': tanimoto_dual_loss
        }
        self.loss = losses.get(self.cfg_train['loss'], "Invalid loss")

    def get_metrics(self):
        """
        Get training metrics.
        """
        metrics = {
            'accuracy': 'accuracy',
            'dice': dice_coef,
            'dice_bin': dice_coef_bin,
        }
        self.metrics = list(
            map(metrics.get, self.cfg_train['metrics'].split(' '))
        )

    def get_callbacks(self):
        """
        Get training callbacks.
        """
        callbacks = {
            'TensorBoard': TensorBoard(
                log_dir=self.savedir, write_graph=True,
                histogram_freq=int(self.cfg_train['histfreq'])
            ),
            'ModelCheckpoint': ModelCheckpoint(
                self.modelsave[:-3]+'_{epoch:02d}-{val_loss:.2f}.h5',
                verbose=2, save_best_only=eval(self.cfg_train['savebestonly']),
                period=int(self.cfg_train['period_checkpoint'])
            ),
            'EarlyStopping': EarlyStopping(
                patience=int(self.cfg_train['patience_earlystop']),
                monitor=self.cfg_train['monitor_earlystop']
            ),
            'CSVLogger': CSVLogger(
                self.modelsave[:-3] +
                '_'+datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv',
                append=True, separator=';'
            ),
            'ReduceLROnPlateau': ReduceLROnPlateau(
                monitor=self.cfg_train['monitor_earlystop'],
                factor=float(self.cfg_train['factor_plateu']),
                patience=int(self.cfg_train['patience_plateu']),
                min_lr=float(self.cfg_train['min_lr_plateu'])
            ),
            'GCCollect': GC_Callback()
        }
        self.callbacks = list(
            map(callbacks.get, self.cfg_train['callbacks'].split(' '))
        )


class GC_Callback(tf.keras.callbacks.Callback):
    """
    Avoid TF garbage collection and free GPU memory during training.
    """
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

# -------------------------------------------------------------------------------
# class CNN Unit Tests
# -------------------------------------------------------------------------------


if __name__ == "__main__":

    # Running Unit Tests
    logging.basicConfig(level=logging.INFO)

    # Local variables
    filename = '../../examples/cnn/cnn.conf'
    cnn_obj = CNN(config=filename)

    unit_tests = [1, 2, 3, 4]

    # 1. Create raster object
    if 1 in unit_tests:
        logging.info(f"Unit Test #1: {cnn_obj.loss} {cnn_obj.metrics}")
        logging.info(f"Unit Test #1: {cnn_obj.callbacks}")

    #    raster = Raster(filename, bands)
    #    assert raster.data.shape[0] == 8, "Number of bands should be 8."
    #    logging.info(f"Unit Test #1: {raster.data} {raster.bands}")
