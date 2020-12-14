import os
import sys
import logging
import numpy as np
import argparse
from datetime import datetime

SEED = 42
np.random.seed(SEED)

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"

# ---------------------------------------------------------------------------
# module utils
#
# General functions for the enhancement and construction of NN workflows.
# Includes utilities to build models and to process data.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Module Methods
# ---------------------------------------------------------------------------


# --------------------------- Data Retrieval Options ----------------------- #

def arg_parser_prepdata():
    """ Legacy Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--x-data', nargs='*', dest='x',
                        help='filenames to load x data from',
                        default=['Data.tif'], type=str)
    parser.add_argument('-y', '--y-data', nargs='*', dest='y',
                        help='filename to load y data from',
                        default=['Train.tif'], type=str)
    parser.add_argument('-np', '--n-patches', action='store', dest='npatch',
                        help='number of patches to crop image',
                        default='16000', type=int)
    parser.add_argument('-ts', '--tile-size', action='store', dest='tsize',
                        help='tile size for each segment',
                        default='256', type=int)
    parser.add_argument('-nc', '--n-classes', action='store', dest='nclass',
                        help='number of classes present',
                        default='19', type=int)
    parser.add_argument('-c', '--n-channels', action='store', dest='nchannels',
                        help='number of input channels to use',
                        default='6', type=int)
    parser.add_argument('-m', '--method', action='store', dest='method',
                        default='rand', type=str,
                        help='data prep method to use',
                        choices=['rand', 'aug', 'cond', 'augcond', 'cloud'])
    parser.add_argument('-ov', '--overlap', action='store', dest='overlap',
                        help='fraction of overlap for prediction',
                        default='50', type=int)
    parser.add_argument('-sdir', '--save-dir', action='store', dest='savedir',
                        help='directory to save prepared data', default='')
    return parser.parse_args()


def arg_parser_train():
    """ Legacy Parse command line arguments """
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument('-x', '--x-data', nargs='*', dest='x',
                        help='filenames to load x data from',
                        default=['Data.tif'], type=str)
    parser.add_argument('-y', '--y-data', nargs='*', dest='y',
                        help='filename to load y data from',
                        default=['Train.tif'], type=str)
    parser.add_argument('-ts', '--tile-size', action='store', dest='tsize',
                        help='tile size for each segment',
                        default='256', type=int)
    parser.add_argument('-nc', '--n-classes', action='store', dest='nclass',
                        help='number of classes present',
                        default='19', type=int)
    parser.add_argument('-c', '--n-channels', action='store', dest='nchannels',
                        help='number of final channels to use',
                        default='6', type=int)
    parser.add_argument('-ic', '--init-channels', action='store',
                        dest='init_nchannels', default='6', type=int,
                        help='number of initial channels to use')
    # Preprocessing arguments
    parser.add_argument('-nf', '--norm-factor', action='store',
                        dest='norm_factor', default='255.0', type=float,
                        help='normalization factor to use')
    parser.add_argument('-std', '--stand-method', action='store',
                        dest='stand_method', type=str,
                        help='standardization method', default='local',
                        choices=['local', 'global', 'none'])
    parser.add_argument('-stdst', '--stand-strategy', action='store',
                        dest='stand_strategy', type=str,
                        help='standard. strategy', default='per-batch',
                        choices=['per-batch', 'per-image', 'none'])
    parser.add_argument('-aug', '--on-the-fly-aug', action='store',
                        dest='on_the_fly_aug', type=bool,
                        help='wether or not to perform on the fly aug',
                        default='False', choices=[True, False])
    # Model arguments
    parser.add_argument('-mf', '--model-file', action='store',
                        dest='model_file', type=str,
                        help='filename to save model in', default='model.h5')
    parser.add_argument('-v', '--validation-split', action='store',
                        dest='val_split', default=.20, type=float,
                        help='percent of data to use for validation')
    parser.add_argument('-se', '--start-epoch', action='store',
                        dest='start_epoch',  default=0, type=int,
                        help='epoch to start training the model')
    parser.add_argument('-b', '--batch-size', action='store', dest='bsize',
                        help='number of samples per batch',
                        default=4096, type=int)
    parser.add_argument('-e', '--epochs', action='store', dest='epochs',
                        help='number of epochs', default=100, type=int)
    parser.add_argument('-net', '--network', action='store', dest='network',
                        help='set network architecture', default='unet',
                        choices=['unet', 'deepunet', 'resunet', 'tiramisu',
                                 'segnet', 'deeplabv3+', 'unetcrf', 'pspnet50',
                                 'pspnet', 'resunet-a', 'resnet',
                                 'fcnresnet50', 'fcnresnet101'
                                 ])
    parser.add_argument('-o', '--optimizer', action='store', dest='optimizer',
                        help='set an optimizer', default='Adadelta',
                        choices=['SGD', 'RMS', 'Adam',
                                 'Adam_Decay', 'Adadelta'
                                 ])
    parser.add_argument('-m', '--maps', nargs='*', dest='nmaps',
                        default=[64, 128, 256, 512, 1024],
                        help='set number of maps in layers', type=int)
    parser.add_argument('-r', '--learning-rate', action='store',
                        dest='learn_rate', default=.001, type=float,
                        help='set a learning rate')
    parser.add_argument('-g', '--gradient', action='store', dest='gradient',
                        help='rho gradient hyperparameter',
                        default='0.95', type=float)
    parser.add_argument('-mom', '--momentum', action='store', dest='momentum',
                        help='momentum hyperparameter',
                        default='0.90', type=float)
    parser.add_argument('-l', '--loss', action='store', dest='loss',
                        help='set a loss function',
                        choices=['mse', 'categorical_crossentropy', 'dice',
                                 'dice_bin', 'binary_crossentropy', 'focal',
                                 'focal_bin', 'tanimoto', 'ce_dl_bin',
                                 'jaccard'
                                 ],
                        default='categorical_crossentropy', type=str)
    parser.add_argument('-met', '--metrics', nargs='*', dest='metrics',
                        default=['accuracy'],
                        help='metrics to show during model training', type=str)
    parser.add_argument('-ld', '--log-dir', action='store', dest='log_dir',
                        type=str, default='model',
                        help='directory to save model logs')
    # Callbacks arguments
    parser.add_argument('-sbo', '--save-bestonly', action='store',
                        dest='save_bestonly', type=str,
                        help='save best only model', default="False")
    parser.add_argument('-cb', '--callbacks', nargs='*', dest='callbacks',
                        default=['TensorBoard'], type=str,
                        help='callbacks to add to the model',
                        choices=['TensorBoard', 'ModelCheckpoint',
                                 'EarlyStopping', 'CSVLogger',
                                 'ReduceLROnPlateau', 'ConfusionMatrix',
                                 'VisPredictions', 'GCCollect'
                                 ])
    parser.add_argument('-pes', '--patience-earlystop', action='store',
                        dest='patience_earlystop', default='20', type=int,
                        help='patience for early stop callback')
    parser.add_argument('-ppu', '--patience-plateu', action='store',
                        dest='patience_plateu', default='10', type=int,
                        help='patience for plateu callback')
    parser.add_argument('-mes', '--monitor-earlystop', action='store',
                        dest='monitor_earlystop', default='val_loss',
                        help='monitor early stop callback',
                        choices=['val_loss', 'accuracy',
                                 'val_accuracy', 'loss'
                                 ],
                        type=str)
    parser.add_argument('-fppu', '--factor-plateu', action='store',
                        dest='factor_plateu', default='0.2', type=float,
                        help='decrease factor plateu callback')
    parser.add_argument('-lppu', '--minlr-plateu', action='store',
                        dest='min_lr_plateu', default='0.2', type=float,
                        help='minimum learning rate plateu callback')
    parser.add_argument('-pcp', '--period-checkpoint', action='store',
                        dest='period_checkpoint', default='5', type=int,
                        help='period for saving model with checkpoint cb')
    # System resources
    parser.add_argument('-gpudev', '--gpu-device', action='store',
                        dest='gpu_devs', default="0", type=str,
                        help='gpu devices to use')
    # Predict
    parser.add_argument('-ov', '--overlap', action='store', dest='overlap',
                        help='fraction of overlap for prediction',
                        default='0.25', type=float)
    parser.add_argument('-mi', '--model-info', action='store',
                        dest='model_info',
                        help='filename with model information',
                        nargs='?', const='')
    parser.add_argument('-sf', '--stand-file', action='store',
                        dest='stand_file',
                        help='filename with saved norms',
                        default='data_norm.csv')
    parser.add_argument('-svd', '--save-data', action='store', dest='s_data',
                        help='filename with saved prob array',
                        default='result.npy')
    parser.add_argument('-svi', '--save-img', action='store', dest='s_img',
                        help='filename with saved tif', default='result.tif')
    return parser.parse_args()


# ------------------------- Image Processing Functions ---------------------- #

def getOccurrences(labels=[], fname='occurences.csv', nclasses=7):
    """
    Return pixel occurences per class.
    :param labels: numpy array with labels in int format
    :param fname: filename to save output
    :param nclasses: number of classes to look for
    :return: CSV file with class and occurrence per class
    """
    f = open(fname, "w+")
    f.write("class,occurence\n")
    for classes in range(nclasses):
        occ = np.count_nonzero(labels == classes)
        f.write(f'{classes},{occ}\n')
    f.close()


def compute_imf_weights(ground_truth, n_classes=None,
                        ignored_classes=[]
                        ) -> np.array:
    """
    Compute inverse median frequency weights for class balancing.

    For each class i, it computes its frequency f_i, i.e the ratio between
    the number of pixels from class i and the total number of pixels.
    Then, it computes the median m of all frequencies. For each class the
    associated weight is m/f_i.

    :param ground_truth: the annotations array
    :param nclasses: number of classes (defaults to max(ground_truth))
    :param ignored_classes: id of classes to ignore (optional)
    :return: numpy array with the IMF coefficients
    """
    n_classes = np.max(ground_truth) if n_classes is None else n_classes
    weights = np.zeros(n_classes)
    frequencies = np.zeros(n_classes)

    for c in range(0, n_classes):
        if c in ignored_classes:
            continue
        frequencies[c] = np.count_nonzero(ground_truth == c)

    # Normalize the pixel counts to obtain frequencies
    frequencies /= np.sum(frequencies)
    # Obtain the median on non-zero frequencies
    idx = np.nonzero(frequencies)
    median = np.median(frequencies[idx])
    weights[idx] = median / frequencies[idx]
    weights[frequencies == 0] = 0.
    return weights


# ------------------------- Output/logging Functions ---------------------- #

def create_logfile(description='output', logdir='results'):
    """
    Create logfile instance to log all output to file.
    :param args: argparser object
    :param logdir: log directory to store log file
    :return: logfile instance, stdour and stderr being logged to file
    """
    logfile = os.path.join(logdir, '{}_log_{}.log'.format(
        datetime.now().strftime("%Y%m%d-%H%M%S"), description)
    )
    logging.warning(f'See {logfile} for output')
    so = se = open(logfile, 'w')  # open our log file
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')  # stdout buffering
    os.dup2(so.fileno(), sys.stdout.fileno())  # redirect to the log file
    os.dup2(se.fileno(), sys.stderr.fileno())
    return logfile


# -------------------------------------------------------------------------------
# module preprocessing Unit Tests
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    # Add unit tests here when required. No unit tests required at this time.

