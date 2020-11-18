import tensorflow as tf


def vgg_st(
    num_classes: int = 100,
    activation: str = "elu",
    regularization_factor: float = 50e-4,
    bias_regularization_factor: float = None,
    batch_normalization: bool = False,
) -> None:
    l2_reg = tf.keras.regularizers.l2(regularization_factor)
    if bias_regularization_factor is None:
        l2_bias_reg = None
    else:
        l2_bias_reg = tf.keras.regularizers.l2(bias_regularization_factor)

    height, width, channel = 32, 32, 3
    input_shape = (height, width, channel)

    input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=l2_reg,
        bias_regularizer=l2_bias_reg,
    )(input)
    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation=activation)(x)
    x = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=l2_reg,
        bias_regularizer=l2_bias_reg,
    )(x)
    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation=activation)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(rate=0.25)(x)

    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=l2_reg,
        bias_regularizer=l2_bias_reg,
    )(x)
    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation=activation)(x)
    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=l2_reg,
        bias_regularizer=l2_bias_reg,
    )(x)
    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation=activation)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(rate=0.25)(x)

    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=l2_reg,
        bias_regularizer=l2_bias_reg,
    )(x)
    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation=activation)(x)
    x = tf.keras.layers.Conv2D(
        filters=128,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=l2_reg,
        bias_regularizer=l2_bias_reg,
    )(x)
    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation=activation)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(rate=0.25)(x)

    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=l2_reg,
        bias_regularizer=l2_bias_reg,
    )(x)
    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation=activation)(x)
    x = tf.keras.layers.Conv2D(
        filters=256,
        kernel_size=(3, 3),
        padding="same",
        kernel_regularizer=l2_reg,
        bias_regularizer=l2_bias_reg,
    )(x)
    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation=activation)(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(rate=0.25)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(
        units=512, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg
    )(x)
    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation=activation)(x)

    x = tf.keras.layers.Dropout(rate=0.5)(x)
    x = tf.keras.layers.Dense(
        units=num_classes, kernel_regularizer=l2_reg, bias_regularizer=l2_bias_reg
    )(x)
    if batch_normalization:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation="softmax")(x)

    model = tf.keras.Model(inputs=[input], outputs=[x])

    return model
