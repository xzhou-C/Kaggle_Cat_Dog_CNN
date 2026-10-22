"""Model definitions. Every model takes raw RGB pixels in [0, 255] and outputs P(Dog)."""
import keras
from keras import layers

BACKBONE_NAME = "backbone"


def augmentation(seed=None):
    # Active only when training=True; an identity at evaluation and inference time.
    return keras.Sequential([
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(0.05, fill_mode="reflect", seed=seed),
        layers.RandomZoom(0.2, 0.2, fill_mode="reflect", seed=seed),
        layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=seed),
        layers.RandomContrast(0.1, seed=seed),
        layers.RandomBrightness(0.2, value_range=(0, 255), seed=seed),
    ], name="augmentation")


def custom_cnn(image_size=(160, 160), n_channel=32, dropout=0.2, l2=1e-4, head="flatten", convs_per_block=1):
    """The from-scratch CNN from the original notebook (iteration 4): 5 conv blocks + a classifier head.

    head="flatten" is the original Flatten -> Dense(512) head. head="gap" replaces it with
    GlobalAveragePooling2D, which drops about 6.5M parameters and is not tied to object position.
    convs_per_block=2 stacks two 3x3 convs before each pooling layer (a 5x5 receptive field).
    """
    reg = keras.regularizers.L2(l2)
    inputs = keras.Input(shape=image_size + (3,))
    x = augmentation()(inputs)
    x = layers.Rescaling(1.0 / 255)(x)
    for i in range(5):
        for _ in range(convs_per_block):
            x = layers.Conv2D(n_channel * 2**i, 3, padding="same", kernel_initializer="he_normal",
                              kernel_regularizer=reg)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(dropout)(x)
    if head == "gap":
        x = layers.GlobalAveragePooling2D()(x)
    else:
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.L2(5e-3))(x)
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="custom_cnn")


def efficientnet(image_size=(160, 160), weights="imagenet", dropout=0.2):
    """EfficientNetB0 pretrained on ImageNet with a new binary head.

    Keras' EfficientNet does its own input scaling, so it takes [0, 255] pixels directly.
    """
    backbone = keras.applications.EfficientNetB0(
        include_top=False, weights=weights, input_shape=image_size + (3,), pooling="avg")
    backbone.name = BACKBONE_NAME
    backbone.trainable = False
    inputs = keras.Input(shape=image_size + (3,))
    x = augmentation()(inputs)
    x = backbone(x, training=False)  # keep BatchNorm in inference mode, also when fine-tuning
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="efficientnet_b0")


def unfreeze_top(model, n_layers=40):
    """Unfreeze the last n_layers of the backbone for fine-tuning, leaving BatchNorm frozen."""
    backbone = model.get_layer(BACKBONE_NAME)
    backbone.trainable = True
    for layer in backbone.layers[:-n_layers]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


MODELS = {"cnn": custom_cnn, "effnet": efficientnet}
