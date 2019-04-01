import keras.backend as K
import tensorflow as tf
import keras
import numpy as np

def f1_metrics(y_true, y_pred):
    """
    f1 Metrics:
    Source: https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    """
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_loss(y_true, y_pred):
    """
    Macro f1 loss:
    Source: https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    """
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)



def dice_loss(y_true, y_pred, smooth=1.):

    """ Loss function base on dice coefficient.
    source:     https://analysiscenter.github.io/radio/_modules/radio/models/keras/losses.html

    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    smooth : float
        small real value used for avoiding division by zero error.

    Returns
    -------
    keras tensor
        tensor containing dice loss.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    answer = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return -answer

def dice_metrics(y_true,y_pred):
    return - dice_loss(y_true,y_pred)

def tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, smooth=1e-10):
    """ Tversky loss function.
    source:     https://analysiscenter.github.io/radio/_modules/radio/models/keras/losses.html

    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    alpha : float
        real value, weight of '0' class.
    beta : float
        real value, weight of '1' class.
    smooth : float
        small real value used for avoiding division by zero error.

    Returns
    -------
    keras tensor
        tensor containing tversky loss.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
    answer = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
    return -answer



def jaccard_coef_logloss(y_true, y_pred, smooth=1e-10):
    """ Loss function based on jaccard coefficient.
    source:     https://analysiscenter.github.io/radio/_modules/radio/models/keras/losses.html

    Parameters
    ----------
    y_true : keras tensor
        tensor containing target mask.
    y_pred : keras tensor
        tensor containing predicted mask.
    smooth : float
        small real value used for avoiding division by zero error.

    Returns
    -------
    keras tensor
        tensor containing negative logarithm of jaccard coefficient.
    """
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    truepos = K.sum(y_true * y_pred)
    falsepos = K.sum(y_pred) - truepos
    falseneg = K.sum(y_true) - truepos
    jaccard = (truepos + smooth) / (smooth + truepos + falseneg + falsepos)
    return -K.log(jaccard + smooth)

def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth



def IoU_metrics(y_true, y_pred):
    intersection = y_true * y_pred
    notTrue = 1 - y_true
    union = y_true + (notTrue * y_pred)
    return K.sum(intersection)/K.sum(union)

def IoU_loss(y_true, y_pred):
    return -IoU_metrics(y_true,y_pred)


# KL divergeance + reconstruction loss
# def vae_loss(z_mean,z_log_var):
#     print("ici")
#
#     def keras_vae_loss(y_true, y_pred):
#         xent_loss = keras.losses.mse(y_true, y_pred)
#         kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#         return  xent_loss+kl_loss
#
#     return keras_vae_loss

def reconstruction_loss(y_true,y_pred):
    return keras.losses.mse(K.flatten(y_true), K.flatten(y_pred))

def vae_loss(z_mean,z_log_var,input_shape):

    def keras_vae_loss(y_true, y_pred):
        # xent_loss = reconstruction_loss(y_true,y_pred) * 128 * 128
        # kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        rec_loss = reconstruction_loss(y_true,y_pred) * np.prod(input_shape)
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss,axis=-1)
        kl_loss *= -0.5
        return K.mean(rec_loss + kl_loss)
        # return  xent_loss+kl_loss

    return keras_vae_loss


# PSNR (Signal Noise Ratio) loss
def psnr_loss(y_true,y_pred):
  return -10.0 * K.log(1.0 / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0)


# wasserstein_loss
def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)
