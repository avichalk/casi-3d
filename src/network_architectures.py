from tensorflow.keras.layers import Activation, Add, BatchNormalization, Concatenate, \
    Conv2D, GaussianNoise, Input, MaxPool2D, UpSampling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from network_components import dilated_block, dilated_residual_block, res_bottlneck,res_block

def restrict_net_residual_block(
        activation='relu',
        depth=2,
        filters=16,
        input_dims=(100, 100, 1),
        final_activation='selu',
        loss=None,
        noise_std=0.1,
        merge=Add(),
        optimizer=SGD(momentum=0.9),
        output_channels=1,
        
):
    """A U-Net without skip connections.

    Args:
        activation:
        depth:
        filters:
        final_activation:
        input_dims:
        loss:
        noise_std:
        optimizer:
        output_channels:

    Returns:
        (keras.models.Model) A compiled and ready-to-use Restrict-Net.
    """

    if loss is None:
        loss = 'mse'

    inputs = Input(shape=input_dims)

    pred = GaussianNoise(stddev=noise_std)(inputs)

    # Restriction
    for _ in range(depth):
        pred = res_bottlneck(
            pred,
            filters=filters,
            activation=activation,
            project=True,
            merge=merge)
        
        pred = res_bottlneck(
            pred,
            filters=filters,
            activation=activation,
            project=True,
            merge=merge)

        pred = MaxPool2D(padding='same')(pred)
        filters *= 2

    pred = BatchNormalization()(pred)
    pred = Activation(activation)(pred)
    pred = Conv2D(filters, (3, 3), padding='same')(pred)

    pred = Flatten()(pred)
    ###### pred = Dense(1, input_shape=(2,), activation="sigmoid")(pred)   ### I do not think the input shape is 2, i.e., only adopt two numbers? 
   
    ###### modified by DX
    pred = Dense(256, activation=activation)(pred)
    pred = Dense(1, activation=final_activation)(pred)
    ######
    
    model = Model(inputs=inputs, outputs=pred)
    model.compile(optimizer=optimizer, loss=loss)
    return model

def restrict_net(
        activation='selu',
        depth=2,
        filters=16,
        final_activation='selu',
        input_dims=(None, None, None, 1),
        loss=None,
        noise_std=0.1,
        optimizer=SGD(momentum=0.9),
        output_channels=1,
        
):
    """A U-Net without skip connections.

    Args:
        activation:
        depth:
        filters:
        final_activation:
        input_dims:
        loss:
        noise_std:
        optimizer:
        output_channels:

    Returns:
        (keras.models.Model) A compiled and ready-to-use Restrict-Net.
    """
    if loss is None:
        loss = 'mse'

    inputs = Input(shape=input_dims)
    pred = GaussianNoise(stddev=noise_std)(inputs)

    # Restriction
    for _ in range(depth):
        pred = BatchNormalization()(pred)
        pred = Activation(activation)(pred)
        pred = Conv2D(filters, (3, 3,3), padding='same')(pred)

        pred = BatchNormalization()(pred)
        pred = Activation(activation)(pred)
        pred = Conv2D(filters, (3,3, 3), padding='same')(pred)

        pred = MaxPool2D()(pred)
        filters *= 2

    pred = BatchNormalization()(pred)
    pred = Activation(activation)(pred)
    pred = Conv2D(filters, (3, 3,3), padding='same')(pred)

    # Reconstitution
    for _ in range(depth):
        pred = UpSampling2D()(pred)
        filters //= 2
        pred = Conv2D(filters, (3, 3), padding='same')(pred)

        pred = BatchNormalization()(pred)
        pred = Activation(activation)(pred)
        pred = Conv2D(filters, (3, 3), padding='same')(pred)

        pred = BatchNormalization()(pred)
        pred = Activation(activation)(pred)
        pred = Conv2D(filters, (3, 3), padding='same')(pred)

    pred = Conv2D(output_channels, (1, 1))(pred)
    pred = BatchNormalization()(pred)
    pred = Activation(final_activation)(pred)

    model = Model(inputs=inputs, outputs=pred)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def u_net(
        activation='selu',
        depth=4,
        filters=16,
        final_activation='selu',
        input_dims=(None, None, None),
        loss=None,
        noise_std=0.1,
        optimizer=SGD(momentum=0.9),
        output_channels=1,
):
    """A Keras implementation of a U-Net.
     See https://arxiv.org/pdf/1505.04597.pdf for details.

    Deviations:
        - Uses a BN-activation-conv structure rather than conv-activation
        - Uses padded convolutions to simplify dimension arithmetic
        - Does not use reflection expanded inputs
        - Cropping is not used on the skip connections
        - Uses 3x3 up-conv, rather than 2x2

    Args:
        activation:
        depth:
        filters:
        final_activation:
        input_dims:
        loss:
        noise_std:
        optimizer:
        output_channels:

    Returns:
        (keras.models.Model) A compiled and ready-to-use U-Net.
    """
    if loss is None:
        loss = 'mse'

    inputs = Input(shape=input_dims)
    pred = GaussianNoise(stddev=noise_std)(inputs)

    # Restriction
    crosses = []
    for _ in range(depth):
        pred = BatchNormalization()(pred)
        pred = Activation(activation)(pred)
        pred = Conv2D(filters, (3, 3), padding='same')(pred)

        pred = BatchNormalization()(pred)
        pred = Activation(activation)(pred)
        pred = Conv2D(filters, (3, 3), padding='same')(pred)

        crosses.append(pred)

        pred = MaxPool2D()(pred)
        filters *= 2

    pred = BatchNormalization()(pred)
    pred = Activation(activation)(pred)
    pred = Conv2D(filters, (3, 3), padding='same')(pred)

    # Reconstitution
    for cross in crosses[::-1]:
        pred = UpSampling2D()(pred)
        filters //= 2
        pred = Conv2D(filters, (3, 3), padding='same')(pred)

        pred = Concatenate()([pred, cross])
        pred = BatchNormalization()(pred)
        pred = Activation(activation)(pred)
        pred = Conv2D(filters, (3, 3), padding='same')(pred)

        pred = BatchNormalization()(pred)
        pred = Activation(activation)(pred)
        pred = Conv2D(filters, (3, 3), padding='same')(pred)

    pred = Conv2D(output_channels, (1, 1))(pred)
    pred = BatchNormalization()(pred)
    pred = Activation(final_activation)(pred)

    model = Model(inputs=inputs, outputs=pred)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def residual_u_net(
        activation='selu',
        depth=3,
        filters=16,
        final_activation='selu',
        input_dims=(None, None, None, 1),
        loss=None,
        merge=Add(),
        noise_std=0.1,
        optimizer=SGD(momentum=0.9),
        output_channels=1,
):
    """A U-Net with residual blocks at each level.

    Args:
        activation:
        depth:
        filters:
        final_activation:
        input_dims:
        loss:
        merge:
        noise_std:
        optimizer:
        output_channels:

    Returns:
        (keras.models.Model) A compiled and ready-to-use Residual U-Net.
    """
    if loss is None:
        loss = 'mse'

    inputs = Input(shape=input_dims)
    pred = GaussianNoise(stddev=noise_std)(inputs)
    pred = res_block(
            pred,
            filter_shape=(7, 7, 7),
            filters=filters,
            activation=activation,
            project=True,
            merge=merge)

    # Restriction
    crosses = []
    for _ in range(depth):
        pred = res_bottlneck(
            pred,
            filters=filters,
            activation=activation,
            project=True,
            merge=merge)
        
        pred = res_bottlneck(
            pred,
            filters=filters,
            activation=activation,
            project=True,
            merge=merge)

        crosses.append(pred)

        pred = MaxPool2D()(pred)
        filters *= 2

    pred = res_bottlneck(
        pred,
        filters=filters,
        activation=activation,
        project=True,
        merge=merge)

    # Reconstitution
    for cross in crosses[::-1]:
        pred = UpSampling2D()(pred)
        filters //= 2
        pred = Conv2D(filters, (3, 3, 3), padding='same')(pred)

        pred = Concatenate()([pred, cross])
        pred = res_bottlneck(
            pred,
            filters=filters,
            activation=activation,
            project=True,
            merge=merge)
#        pred = res_bottlneck(
#            pred,
#            filters=filters,
#            activation=activation,
#            project=True,
#            merge=merge)

    pred = Conv2D(output_channels, (1, 1, 1))(pred)
    pred = BatchNormalization()(pred)
    pred = Activation(final_activation)(pred)

    model = Model(inputs=inputs, outputs=pred)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def dilated_net(
        activation='selu',
        depth=3,
        filters=32,
        final_activation='sigmoid',
        input_dims=(None, None, None,1),
        loss=None,
        merge=Concatenate(),
        noise_std=0.1,
        optimizer=SGD(momentum=0.9),
        output_channels=1,
):
    """A neural network primarily composed of dilated convolutions.
    Uses exponentially dilated convolutions to operate on multi-scale features.
    No up-sampling or down-sampling is used, since sequential dilated convolutions
    have extremely large effective receptive fields.

    Args:
        activation:
        depth:
        filters:
        final_activation:
        input_dims:
        loss:
        merge:
        noise_std:
        optimizer:
        output_channels:

    Returns:
        (keras.models.Model) A compiled and ready-to-use Residual-U-Net.
    """
    if loss is None:
        loss = 'mse'

    inputs = Input(shape=input_dims)
    pred = GaussianNoise(stddev=noise_std)(inputs)

    for _ in range(depth):
        pred = dilated_block(pred,
                             filters=filters,
                             activation=activation,
                             merge=merge)

    pred = BatchNormalization()(pred)
    pred = Activation(activation)(pred)
    pred = Conv2D(filters, (3, 3, 3), padding='same')(pred)

    pred = Conv2D(output_channels, (1, 1, 1))(pred)
    pred = BatchNormalization()(pred)
    pred = Activation(final_activation)(pred)

    model = Model(inputs=inputs, outputs=pred)
    model.compile(optimizer=optimizer, loss=loss)
    return model


def dilated_res_net(
        activation='selu',
        depth=2,
        filters=32,
        final_activation='sigmoid',
        input_dims=(None, None, None, 1),
        loss=None,
        merge=Add(),
        noise_std=0.1,
        optimizer=SGD(momentum=0.9),
        output_channels=1,
):
    """A neural network primarily composed of dilated convolutions.
    Uses exponentially dilated convolutions to operate on multi-scale features.
    No up-sampling or down-sampling is used, since sequential dilated convolutions
    have extremely large effective receptive fields.

    Args:
        activation:
        depth:
        filters:
        final_activation:
        input_dims:
        loss:
        merge:
        noise_std:
        optimizer:
        output_channels:

    Returns:
        (keras.models.Model) A compiled and ready-to-use Residual-U-Net.
    """
    if loss is None:
        loss = 'mse'

    inputs = Input(shape=input_dims)
    pred = GaussianNoise(stddev=noise_std)(inputs)

    pred = Conv2D(filters, (1, 1, 1))(pred)
    for _ in range(depth):
        pred = dilated_residual_block(
            pred,
            filters=filters,
            activation=activation,
            merge=merge)

    pred = BatchNormalization()(pred)
    pred = Activation(activation)(pred)
    pred = Conv2D(filters, (3, 3, 3), padding='same')(pred)

    pred = Conv2D(output_channels, (1, 1, 1))(pred)
    pred = BatchNormalization()(pred)
    pred = Activation(final_activation)(pred)

    model = Model(inputs=inputs, outputs=pred)
    model.compile(optimizer=optimizer, loss=loss)
    return model
