import logging
import random
import numpy as np
import pandas as pd
import rasterio as rio
import tensorflow as tf
from PIL import Image
import itertools
import tifffile as tiff
import io

import matplotlib.pyplot as plt
import matplotlib.colors as pltc
import matplotlib.patches as mpatches

import seaborn as sns
from webcolors import name_to_rgb
from deeprsensing.metrics import iou_val, acc_val, prec_val, recall_val

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"

# ---------------------------------------------------------------------------
# module visualize
#
# Visualization functions for NN workflows.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Module Methods
# ---------------------------------------------------------------------------


# ----------------------- Color Method Functions  -------------------------- #

def gen_cmap(nclasses=19, random_colors=True):
    """
    Args:
        nclasses: integer with number of classes
        random:   boolean value, if True, colors are generated randomly
    Returns:
        colormap object with nclasses randomly generated colors
    """
    cc = []
    if random_colors:
        all_colors = [k for k, v in pltc.cnames.items()]
        for color in random.sample(range(len(all_colors)), nclasses):
            cc.append(all_colors[color])
        return pltc.ListedColormap(cc)
    else:
        cc = ['gray', 'forestgreen', 'fuchsia', 'lemonchiffon', 'indigo',
              'lightcyan', 'mediumturquoise', 'navy', 'orange', 'pink',
              'rebeccapurple', 'sandybrown', 'silver', 'slategray',
              'springgreen', 'steelblue', 'tomato', 'violet', 'yellow'
              ]
        return pltc.ListedColormap(cc[:nclasses])


# ----------------------- Segmentation classes  -------------------------- #

def get_RIT18_classes():
    classes = {
            'Other Class/Image Border':      'black',
            'Road Markings':                 'yellow',
            'Tree':                          'darkgreen',
            'Building':                      'gray',
            'Vehicle (Car, Truck, or Bus)':  'pink',
            'Person':                        'tomato',
            'Lifeguard Chair':               'firebrick',
            'Picnic Table':                  'brown',
            'Black Wood Panel':              'burlywood',
            'White Wood Panel':              'white',
            'Orange Landing Pad':            'orange',
            'Water Buoy':                    'powderblue',
            'Rocks':                         'lightslategrey',
            'Other Vegetation':              'lightgreen',
            'Grass':                         'darkorange',
            'Sand':                          'khaki',
            'Water (Lake)':                  'darkblue',
            'Water (Pond)':                  'dodgerblue',
            'Asphalt (Parking Lot/Walkway)': 'fuchsia'
    }
    return classes


def get_Vietnam_classes(cloud=False):
    classes = {
        'tree':        'green',
        'water':       'darkblue',
        'build':       'red',
        'shadow':      'purple',
        'other':       'linen',
        'small tree':  'lightgreen'
    }
    if cloud:
        classes['cloud'] = 'black'
    return classes


# ---------------------- Benchmarks Visualizations  ------------------------- #

def gen_barplot(outimg='benchmark.png'):

    # Data taken from benchmark
    labels = ['1GPU CNN', '1GPU RNN', '4GPU CNN']
    keras = [149, 271, 207]
    pytorch = [104, 199, 138]
    tensorflow = [133, 188, 0]

    x = np.arange(len(labels))  # the label locations
    width = 0.30  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - (width/3), keras, width, label='Keras')
    rects2 = ax.bar(x + (width/3) * 2, pytorch, width, label='PyTorch')
    rects3 = ax.bar(x + (width/3) * 5, tensorflow, width, label='TensorFlow')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Wall Time (s)')
    ax.set_xlabel('Neural Network Architecture per Framework')
    ax.set_title('Training Wall time per Framework')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.text(0.03, 0.95,
            'CNN: (VGG-style) on CIFAR-10\nRNN: IMDB – Sentiment Analysis',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes
            )
    ax.legend()

    def autolabel(rects):
        """
        Attach a text label above each bar in *rects*, displaying its height.
        """
        for rect in rects:
            if rect.get_height() != 0:
                height = rect.get_height()
                ax.annotate('{}'.format(height),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom'
                            )
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    fig.tight_layout()
    plt.savefig(outimg)


# ----------------------- Segmentation Visualizations  ---------------------- #

def confusion_matrix(y_true=[], y_pred=[], nclasses=7, norm=True):
    """
    Args:
        y_true:   2D numpy array with ground truth
        y_pred:   2D numpy array with predictions (already processed)
        nclasses: number of classes
    Returns:
        numpy array with confusion matrix
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    con_mat = tf.math.confusion_matrix(
        labels=y_true, predictions=y_pred, num_classes=nclasses
         ).numpy()
    if norm:
        con_mat = np.around(
            con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis],
            decimals=2
            )
    where_are_NaNs = np.isnan(con_mat)
    con_mat[where_are_NaNs] = 0
    return pd.DataFrame(
        con_mat, index=range(nclasses), columns=range(nclasses)
        )


def validation_graph(y_true=[], y_pred=[], classes='None', show=False,
                     outimg='validation.png', title='Model', con_mat_df=[]):
    """

    Args:
        y_true:
        y_pred:
        classes:
        show:
        outimg:
        title:
        con_mat_df:

    Returns:

    """
    fig, axes = plt.subplots(1, 3, figsize=(28, 6))

    # Add figures
    img_true = axes[0].imshow(
        y_true, aspect='auto', cmap=pltc.ListedColormap(classes.values())
        )
    img_pred = axes[1].imshow(
        y_pred, aspect='auto', cmap=pltc.ListedColormap(classes.values())
        )
    img_conf = sns.heatmap(
        con_mat_df, annot=True, cmap=plt.cm.Blues, ax=axes[2]
        )
    logging.info(f'Visualization objects {img_pred} {img_conf}')

    axes[0].set_title('Ground Truth',     fontweight='bold')
    axes[1].set_title('Prediction',       fontweight='bold')
    axes[2].set_title('Confusion Matrix', fontweight='bold')

    for i in range(2):
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    # Setup colorbar and colorbar ticks
    fig.tight_layout()  # remove empty space between subplots
    cbar = fig.colorbar(
        img_true, ax=[axes[0], axes[1]], pad=0.01, location='right'
        )  # works on the right side of image
    cbar.ax.invert_yaxis()
    cbar.set_ticks([0.5, 1.3, 2.2, 3.0, 3.9, 4.7, 5.6])
    cbar.ax.set_yticklabels(classes.keys())

    # Add arrows to the image
    axes[2].annotate('', xy=(0.002, 0.6),
                     xytext=(-1.5, 0.6), va='center', multialignment='right',
                     arrowprops={'arrowstyle': '-|>', 'lw': 7, 'ec': 'g'})
    axes[2].annotate('', xy=(0.002, 1.5),
                     xytext=(-1.5, 1.5), va='center', multialignment='right',
                     arrowprops={
                         'arrowstyle': '-|>', 'lw': 7, 'ec': 'darkblue'
                         }
                     )
    axes[2].annotate('', xy=(0.002, 2.6),
                     xytext=(-1.5, 2.6), va='center', multialignment='right',
                     arrowprops={'arrowstyle': '-|>', 'lw': 7, 'ec': 'red'})
    axes[2].annotate('', xy=(0.002, 3.5),
                     xytext=(-1.2, 3.5), va='center', multialignment='right',
                     arrowprops={'arrowstyle': '-|>', 'lw': 7, 'ec': 'purple'})
    axes[2].annotate('', xy=(0.002, 4.6),
                     xytext=(-1.5, 4.6), va='center', multialignment='right',
                     arrowprops={'arrowstyle': '-|>', 'lw': 7, 'ec': 'linen'})
    axes[2].annotate('', xy=(0.002, 5.5),
                     xytext=(-1.1, 5.5), va='center', multialignment='right',
                     arrowprops={
                         'arrowstyle': '-|>', 'lw': 7, 'ec': 'lightgreen'
                         }
                     )
    axes[2].annotate('', xy=(0.002, 6.6),
                     xytext=(-1.5, 6.55), va='center', multialignment='right',
                     arrowprops={'arrowstyle': '-|>', 'lw': 7, 'ec': 'black'})

    # Add title on lower level and ticks to heatmap
    iou_score = round(iou_val(y_true, y_pred) * 100, 2)
    acc_score = round(acc_val(y_true.flatten(), y_pred.flatten()) * 100, 2)
    plt.text(-18, 7.3,
             f'Model: {title}, Pixel Acc: {acc_score}%, IoU: {iou_score}%',
             fontweight='bold', fontsize=10
             )
    plt.minorticks_off()
    plt.xticks(np.arange(len(classes.values()))+0.5, classes.keys())
    plt.yticks([])

    fig1 = plt.gcf()
    if show:
        plt.show()
    plt.draw()
    fig1.savefig(outimg)


def validation_graph_HD(y_true=[], y_pred=[], classes='None', show=False,
                        outimg='validation.png', title='Model', con_mat_df=[],
                        model_info='model.csv'):
    """
    Args:
        y_true:
        y_pred:
        classes:
        show:
        outimg:
        title:
        con_mat_df:
        model_info:
    Returns:
    """

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    dataframe = pd.read_csv(model_info, sep=';')

    # Add figures
    img_true = axes[0, 0].imshow(
        npy_to_png(seg=y_true, classes=classes,
                   outimg=outimg[:-4]+'_truth.png', ret=True
                   ), aspect='auto'
    )
    img_pred = axes[0, 1].imshow(
        npy_to_png(seg=y_pred, classes=classes,
                   outimg=outimg[:-4]+'_pred.png', ret=True,
                   aspect='auto'
                   )
    )
    mod_conf = sns.heatmap(
        con_mat_df, annot=True, cmap=plt.cm.Blues, ax=axes[1, 0],
        cbar_kws={"shrink": .80}
    )
    mod_info = dataframe.plot(
        x='epoch', y=['accuracy', 'loss', 'val_accuracy', 'val_loss'],
        style='.-', markevery=5, ax=axes[1, 1]
    )
    logging.info(f'{img_true},{img_pred},{mod_conf},{mod_info}')

    # Configure figure ticks
    for i in range(2):
        axes[0, i].set_xticks([])
        axes[0, i].set_yticks([])
    plt.minorticks_off()
    axes[1, 0].set_xticks(np.arange(len(classes.values())) + 0.5, minor=False)
    axes[1, 0].set_yticks(np.arange(len(classes.values())) + 0.5, minor=False)
    axes[1, 0].set_xticklabels(classes.keys(), rotation='horizontal')
    axes[1, 0].set_yticklabels(classes.keys(), rotation='horizontal')

    # Adding legend
    labLegend = []
    for lab in classes:
        labLegend.append(mpatches.Patch(color=classes[lab], label=lab))
    axes[0, 0].legend(
        handles=labLegend, loc='lower center', bbox_to_anchor=(1.15, -0.09),
        fancybox=True, ncol=7
    )

    # Add title on lower level and ticks to heatmap
    iou_score = round(iou_val(y_true, y_pred) * 100, 2)
    acc_score = round(acc_val(y_true.flatten(), y_pred.flatten()) * 100, 2)
    prec_score, prec_score_class = prec_val(y_true.flatten(), y_pred.flatten())
    rec_score, rec_score_class = recall_val(y_true.flatten(), y_pred.flatten())
    logging.info("Precission and Recall score: ", prec_score, rec_score)

    axes[0, 0].set_title('Ground Truth',      fontweight='bold')
    axes[0, 1].set_title('Prediction',        fontweight='bold')
    axes[1, 0].set_title('Confusion Matrix',  fontweight='bold')
    axes[1, 1].set_title('Model Performance', fontweight='bold')
    plt.suptitle(f'Pixel Acc: {acc_score}%, IoU: {iou_score}%')

    # Saving and plotting figure
    fig1 = plt.gcf()
    if show:
        plt.show()
    plt.draw()
    fig1.savefig(outimg)


# ------------------------ Tensorboard Visualizations  ---------------------- #

def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call
    """
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def image_grid(val_true, pred_images, class_cmap):
    """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(15, 15))
    for i in range(1, 64, 2):  # change to 64 later
        # Start next subplot.
        plt.subplot(8, 8, i, title="Truth")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(val_true[i-1], cmap=class_cmap)  # plt.cm.binary)
        # Start next subplot.
        plt.subplot(8, 8, i + 1, title="Prediction")
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(pred_images[i-1], cmap=class_cmap)  # plt.cm.binary)
    return figure


def plot_confusion_matrix(cm, class_names=['a', 'b', 'c']):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names: list with classes for confusion matrix
    Return: confusion matrix figure.
    """
    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plotFilters(conv_filter):
    fig, axes = plt.subplots(1, 3, figsize=(5, 5))
    axes = axes.flatten()
    for img, ax in zip(conv_filter, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# ------------------------ Saving Visualizations  ---------------------------#


def npy_to_png(seg='segments.npy', classes='None', outimg='img.png', ret=True):
    """
    Args:
        seg: numpy array with anotated predictions
        classes: dict object with class:color for each class
        outimg: string with filename for png image
    Returns:
        saves png image
    """
    c = list(classes.values())  # list of colors from classes
    im = Image.new('RGB', seg.shape[::-1])
    for i in range(seg.shape[0]):
        for j in range(seg.shape[1]):
            im.putpixel((j, i), name_to_rgb(c[int(seg[i, j])]))
    im.save(outimg)
    if ret:
        return im


def npy_to_tif(raster_f='image.tif', segments='segment.npy',
               outtif='segment.tif', ndval=-9999
               ):
    """
    Args:
        raster_f:
        segments:
        outtif:
    Returns:
    """
    # get geospatial profile, will apply for output file
    with rio.open(raster_f) as src:
        meta = src.profile
        nodatavals = src.read_masks(1).astype('int16')
    print(meta)

    # load numpy array if file is given
    if type(segments) == str:
        segments = np.load(segments)
    segments = segments.astype('int16')
    print(segments.dtype)  # check datatype

    nodatavals[nodatavals == 0] = ndval
    segments[nodatavals == ndval] = nodatavals[nodatavals == ndval]

    out_meta = meta  # modify profile based on numpy array
    out_meta['count'] = 1  # output is single band
    out_meta['dtype'] = 'int16'  # data type is float64

    # write to a raster
    with rio.open(outtif, 'w', **out_meta) as dst:
        dst.write(segments, 1)


# -------------------------------------------------------------------------------
# module model Unit Tests
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # number of unit tests to run
    unit_tests = [1, 2, 3]

    # Test #1: return vietnam colors
    if 1 in unit_tests:
        cmap_colors = gen_cmap(nclasses=19, random_colors=True)
        logging.info(cmap_colors)

    # Test #2: return vietnam classes
    if 2 in unit_tests:
        classes = get_Vietnam_classes()
        logging.info(classes)

    # Test #3: generate confusion matrix
    if 3 in unit_tests:
        y_true = np.random.randint(7, size=(64, 64))
        y_pred = np.random.randint(7, size=(64, 64))
        con_mat_df = confusion_matrix(y_true, y_pred, nclasses=7)
        logging.info(con_mat_df)

    # Test #4: generate validation graph
    if 4 in unit_tests:
        image = tiff.imread("Keelin_10_20190228_isotraining_1_1.tif")
        image[image < 0] = 6
        image[image > 6] = 6
        logging.info(np.unique(image))
        con_mat_df = confusion_matrix(image, image, nclasses=7)
        logging.info(con_mat_df)
        validation_graph(
            y_true=image, y_pred=image, classes=classes, show=True,
            outimg='validation.png', con_mat_df=con_mat_df,
            title='unet_vietnam_binary_Adadelta_48000_64_0.0001_64_35-0.25'
            )
