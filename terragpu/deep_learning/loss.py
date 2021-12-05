import logging
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"

# ---------------------------------------------------------------------------
# module metrics
#
# General functions to compute custom losses.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Module Methods
# ---------------------------------------------------------------------------


# ------------------------------ Loss Functions -------------------------- #

def dice_coef_bin_loss(y_true, y_pred):
    return 1.0 - dice_coef_bin(y_true, y_pred)


def dice_coef_bin(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / \
        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / \
        (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred, numLabels=6):
    dice = 0
    for index in range(numLabels):
        dice -= dice_coef(y_true[:, :, :, index],
                          y_pred[:, :, :, index]
                          )
    return dice


def generalized_dice(y_true, y_pred, exp):
    GD = []
    print(K.shape(y_pred))
    smooth = 1.0
    for i in range(y_pred.shape[2]):
        y_true_per_label = K.cast(
            K.not_equal(y_true[:, :, i], 0), K.floatx()
            )  # Change to binary
        y_pred_per_label = y_pred[:, :, i]
        weight = K.pow(1/K.sum(y_true_per_label, axis=1), exp)
        intersection = K.sum(y_true_per_label * y_pred_per_label, axis=1)
        union = K.sum(y_true_per_label + y_pred_per_label, axis=1)
        GD.append(weight * (2. * intersection + smooth) / (union + smooth))
    GD_out = K.stack(GD)
    return GD_out


def exp_dice_loss(exp=1.0):
    """
    :param exp: exponent. 1.0 for no exponential effect, i.e. log Dice.
    """
    def inner(y_true, y_pred):
        """
        Computes the average exponential log Dice coefficients
        :param y_true: one-hot tensor * by label weights, (bsize, n_pix, n_lab)
        :param y_pred: softmax probabilities, same shape as y_true
        :return: average exponential log Dice coefficient.
        """
        dice = dice_coef(y_true, y_pred)
        dice = K.clip(dice, K.epsilon(), 1 - K.epsilon())
        dice = K.pow(-K.log(dice), exp)
        if K.ndim(dice) == 2:
            dice = K.mean(dice, axis=-1)
        return dice
    return inner


def ce_dl_bin(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return 1 - (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f))
    return K.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)


def jaccard_distance_loss(y_true, y_pred, numLabels=6, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets.
    This has been shifted so it converges on 0 and is smoothed to
    avoid exploding or disapearing gradient.
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    Modified by jordancaraballo to support one-hot tensors.
    """
    jacloss = 0
    for index in range(numLabels):
        y_true_f = K.flatten(y_true[:, :, :, index])
        y_pred_f = K.flatten(y_pred[:, :, :, index])
        intersection = K.sum(K.abs(y_true_f * y_pred_f))
        sum_ = K.sum(K.abs(y_true_f) + K.abs(y_pred_f))
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        jacloss += (1 - jac) * smooth
    return jacloss


def tanimoto_loss(label, pred):
    """
    Softmax version of tanimoto loss.
                           Nc    Np
                           ∑  wj ∑ pij * lij
                          J=1   i=1
    T(pij,lij) = -------------------------------------
                   Nc   Np
                   ∑ wj ∑ ((pij)^2 + (lij)^2 - pij * lij)
                  J=1  i=1
        where:
          Nc = n classes; Np = n pixels i; wj = weights per class J
          pij = probability of pixel for class J
          lij = label of pixel for class J
    wj can be calculated straight from the last layer or using
        wj = (Vj)^-2, where Vj is the total sum of true positives per class
    """
    square = tf.square(pred)
    sum_square = tf.reduce_sum(square, axis=-1)
    product = tf.multiply(pred, label)
    sum_product = tf.reduce_sum(product, axis=-1)
    denominator = tf.subtract(tf.add(sum_square, 1), sum_product)
    loss = tf.divide(sum_product, denominator)
    loss = tf.reduce_mean(loss)
    return 1.0-loss


def tanimoto_dual_loss(label, pred):
    """
    Dual Tanimoto loss
        ~
        T(pij,lij) = ( Tanimoto(pij,lij) + Tanimoto(1-pij, 1-lij) ) / 2.0
    """
    loss1 = tanimoto_loss(pred, label)
    pred = tf.subtract(1.0, pred)
    label = tf.subtract(1.0, label)
    loss2 = tanimoto_loss(pred, label)
    loss = (loss1+loss2)/2.0
    return loss


def focal_loss_bin(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
        FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if
            the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
        model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)],
        metrics=["accuracy"], optimizer=adam)
    """
    def focal_loss_bin_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
        return - K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_bin_fixed


def focal_loss_cat(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
             m
        FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
            c=1
        where m = number of classes, c = class and o = observation
    Parameters:
        alpha -- the same as weighing factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
        model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)],
        metrics=["accuracy"], optimizer=adam)
    """
    def focal_loss_cat_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

        # Compute mean loss in mini_batch
        return K.mean(loss, axis=1)

    return focal_loss_cat_fixed


def tversky_negative(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / \
        (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return K.pow((1 - tv), gamma)


def tversky(y_true, y_pred, alpha=0.6, beta=0.4):
    """
    Function to calculate the Tversky loss for imbalanced data
    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param alpha: weight of false positives
    :param beta: weight of false negatives
    :param weight_map:
    :return: the loss
    """
    y_t = y_true[..., 0]
    y_t = y_t[..., np.newaxis]
    # weights
    y_weights = y_true[..., 1]
    y_weights = y_weights[..., np.newaxis]

    ones = 1
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_t
    g1 = ones - y_t

    tp = tf.reduce_sum(y_weights * p0 * g0)
    fp = alpha * tf.reduce_sum(y_weights * p0 * g1)
    fn = beta * tf.reduce_sum(y_weights * p1 * g0)

    EPSILON = 0.00001
    numerator = tp
    denominator = tp + fp + fn + EPSILON
    score = numerator / denominator
    return 1.0 - tf.reduce_mean(score)


def true_positives(y_true, y_pred):
    """compute true positive"""
    y_t = y_true[..., 0]
    y_t = y_t[..., np.newaxis]
    return K.round(y_t * y_pred)


def false_positives(y_true, y_pred):
    """compute false positive"""
    y_t = y_true[..., 0]
    y_t = y_t[..., np.newaxis]
    return K.round((1 - y_t) * y_pred)


def true_negatives(y_true, y_pred):
    """compute true negative"""
    y_t = y_true[..., 0]
    y_t = y_t[..., np.newaxis]
    return K.round((1 - y_t) * (1 - y_pred))


def false_negatives(y_true, y_pred):
    """compute false negative"""
    y_t = y_true[..., 0]
    y_t = y_t[..., np.newaxis]
    return K.round((y_t) * (1 - y_pred))


def sensitivity(y_true, y_pred):
    """compute sensitivity (recall)"""
    y_t = y_true[..., 0]
    y_t = y_t[..., np.newaxis]
    tp = true_positives(y_t, y_pred)
    fn = false_negatives(y_t, y_pred)
    return K.sum(tp) / (K.sum(tp) + K.sum(fn))


def specificity(y_true, y_pred):
    """compute specificity (precision)"""
    y_t = y_true[..., 0]
    y_t = y_t[..., np.newaxis]
    tn = true_negatives(y_t, y_pred)
    fp = false_positives(y_t, y_pred)
    return K.sum(tn) / (K.sum(tn) + K.sum(fp))


# -------------------------------------------------------------------------------
# module model Unit Tests
# -------------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Add unit tests here

