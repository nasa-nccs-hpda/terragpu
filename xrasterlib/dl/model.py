import logging
import gc
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.callbacks import TensorBoard, CSVLogger
from tensorflow.keras.callbacks import ReduceLROnPlateau

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"

# ---------------------------------------------------------------------------
# module model
#
# Functions for the enhancement and construction of NN workflows.
# Includes utilities to build models and to process data.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Module Methods
# ---------------------------------------------------------------------------


# ------------------------------ Custom Callbacks -------------------------- #

class GC_Callback(tf.keras.callbacks.Callback):
    """
    Avoid TF garbage collection and free GPU memory during training.
    """
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


def get_callbacks(model_name='model_for_rsensing.h5', logs_dir='logs',
                  callbacks=['TensorBoard', 'ModelCheckpoint'], verb=2,
                  hist_freq=5, bestonly=True, patience_earlystop=20,
                  patience_plateu=5, monitor='val_loss', factor_plateu=0.2,
                  min_lr_plateu=0.0, period_checkpoint=5
                  ):
    """
    Generate tensorflow callbacks.
    :param model_name: string to store model name
    :param logs_dir: directory to save model output
    :param callbacks: list of strings with callbacks to include
    :param verb: model verbosity
    :param hist_freq: frequency of histograms
    :param bestonly: save best-only models
    :param patience_earlystop: epochs to wait for model to stop
    :param patience_plateu: validation value to wait for to stop model
    :param monitor: what metric to monitor for stopping model
    :param factor_plateu: float value to wait on plateu
    :param min_lr_plateu: minimum value to wait on plateu
    :param period_checkpoint: epochs to wait to save model and stats
    return list of initialized callbacks.
    """
    callbacks_gen = []  # list to store callbacks for TensorFlow

    if 'TensorBoard' in callbacks:
        callbacks_gen.append(
            TensorBoard(
                log_dir=logs_dir,
                write_graph=True,
                histogram_freq=hist_freq
                )
            )
    if 'ModelCheckpoint' in callbacks:
        callbacks_gen.append(
            ModelCheckpoint(
                model_name[:-3]+'_{epoch:02d}-{val_loss:.2f}.h5',
                verbose=verb,
                save_best_only=eval(bestonly),
                period=period_checkpoint
                )
            )
    if 'EarlyStopping' in callbacks:
        callbacks_gen.append(
            EarlyStopping(
                patience=patience_earlystop,
                monitor=monitor
                )
            )
    if 'CSVLogger' in callbacks:
        callbacks_gen.append(
            CSVLogger(
                model_name[:-3] +
                '_'+datetime.now().strftime("%Y%m%d-%H%M%S")+'.csv',
                append=True,
                separator=';'
                )
            )
    if 'ReduceLROnPlateau' in callbacks:
        callbacks_gen.append(
            ReduceLROnPlateau(
                monitor=monitor,
                factor=factor_plateu,
                patience=patience_plateu,
                min_lr=min_lr_plateu
                )
            )
    if 'GCCollect' in callbacks:
        callbacks_gen.append(GC_Callback())
    return callbacks_gen


# -------------------------------------------------------------------------------
# module model Unit Tests
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Add unit tests here:
    callbacks = get_callbacks()  # testing callbacks
