"""Grad-CAM: which pixels pushed the model toward its prediction."""
import keras
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf
from keras import layers

from catdog.models import BACKBONE_NAME


def _last_conv(model):
    convs = [l for l in model.layers if isinstance(l, layers.Conv2D)]
    return convs[-1]


def gradcam(model, image):
    """Return (heatmap in [0, 1] at the image's size, P(Dog)) for one HxWx3 image in [0, 255].

    Works for both the custom CNN (conv layers in the top-level graph) and the transfer model
    (conv layers inside the nested backbone). The forward pass is replayed layer by layer so the
    tape can see the last conv feature map.
    """
    x = tf.convert_to_tensor(image[None], tf.float32)
    names = {l.name for l in model.layers}
    target = None if BACKBONE_NAME in names else _last_conv(model)

    with tf.GradientTape() as tape:
        tape.watch(x)  # record every op, even through frozen (non-trainable) layers
        h, feats = x, None
        for layer in model.layers:
            if isinstance(layer, layers.InputLayer):
                continue
            if layer.name == BACKBONE_NAME:
                # EfficientNet's last conv block ends in "top_activation"; use the feature map after it.
                split = keras.Model(layer.inputs, [layer.get_layer("top_activation").output, layer.output])
                feats, h = split(h, training=False)
            else:
                h = layer(h, training=False)
                if layer is target:
                    feats = h
        p_dog = h[0, 0]
        score = tf.where(p_dog >= 0.5, p_dog, 1.0 - p_dog)  # explain the predicted class

    grads = tape.gradient(score, feats)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(feats[0] * weights, axis=-1))
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    cam = tf.image.resize(cam[..., None], image.shape[:2])[..., 0]
    return cam.numpy(), float(p_dog)


def overlay(image, heatmap, alpha=0.4):
    """Blend a jet-colored heatmap onto an image in [0, 255]; returns uint8."""
    colored = cm.jet(heatmap)[..., :3] * 255
    return np.clip((1 - alpha) * image + alpha * colored, 0, 255).astype(np.uint8)
