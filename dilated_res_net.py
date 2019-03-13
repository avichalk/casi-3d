"""

"""

from pathlib import Path
import argparse
import logging
import sys

import numpy as np
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.losses import mse
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

import network_architectures as arch
from network_components import cyclic_lr_schedule, CyclicLRScheduler, soft_iou_loss
from preprocessing import load_fits, density_preprocessing, normalize, pad_data, prediction_to_fits
from visualizations import triplet_comparison_figure


def main(
        model_factory=arch.residual_u_net,
        mode='classify_co',
        epochs=200,
        cycles=5,
        lr0=0.2,
        batch_size=8,
        filters=16,
        noise_std=0.,
        rot_range=0,
        w_shift_range=0.,
        h_shift_range=0.,
        channel_shift_range=0.,
        h_flip=False,
        v_flip=False,
        zoom_range=0.,
        verbose=False,):
    Path(f'../models/{model_factory.__name__}/figs').mkdir(parents=True, exist_ok=True)
    mode = mode.strip().lower()

    logging.basicConfig(filename=f'../models/{model_factory.__name__}/{mode}.log',
                        level=logging.DEBUG)
    if verbose:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    if mode == 'classify_co':
        files = ['../data/CO_integrate_binarized/co_all_rot.fits',
                 '../data/CO_integrate_binarized/co_all_rot_tracer.fits']
        x, y = load_fits(files)
        x -= np.mean(x)
        x /= np.std(x)
        x = np.expand_dims(pad_data(x), axis=-1)
        y = np.expand_dims(pad_data(y), axis=-1)
        activation = 'selu'
        final_activation = 'sigmoid'
        loss = soft_iou_loss
    elif mode == 'regress_co':
        files = ['../data/CO_integrate/co_all_rot.fits',
                 '../data/CO_integrate/co_all_rot_tracer.fits']
        x, y = load_fits(files)

        log_op = lambda x: np.log10(1. + x - np.min(x))
        x = np.expand_dims(pad_data(normalize(log_op(x))), axis=-1)
        y = np.expand_dims(pad_data(normalize(log_op(y))), axis=-1)
        activation = 'selu'
        final_activation = 'selu'
        loss = mse
    elif mode == 'regress_density':
        files = ['../data/Density/wind_2cr_flatrho_2315_256.fits',
                 '../data/Density/wind_2cr_flattracer_2315_256.fits']
        x, y = density_preprocessing(*load_fits(files))
        activation = 'selu'
        final_activation = 'selu'
        loss = mse
    else:
        raise ValueError(f"Invalid mode: provided {mode}, expected 'classify_co',"
                         " 'regress_co', or 'regress_density'.")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25)
    batches_per_epoch = int(np.ceil(len(x_train) / batch_size))

    model = model_factory(
        activation=activation,
        filters=filters,
        final_activation=final_activation,
        input_dims=x.shape[1:],
        loss=loss,
        noise_std=noise_std,
    )

    model.summary(print_fn=logging.info)

    train_gen = ImageDataGenerator(
        rotation_range=rot_range,
        width_shift_range=w_shift_range,
        height_shift_range=h_shift_range,
        channel_shift_range=channel_shift_range,
        horizontal_flip=h_flip,
        vertical_flip=v_flip,
        zoom_range=zoom_range,
        fill_mode='constant',
        cval=0,
    )
    test_gen = ImageDataGenerator(fill_mode='constant', cval=0)
    checkpoint = ModelCheckpoint(
        filepath=f'../models/{model_factory.__name__}/{mode}_model.h5',
        save_best_only=True)
    logger = CSVLogger(f'../models/{model_factory.__name__}/{mode}_errors.csv')
    lr_func = cyclic_lr_schedule(
        lr0=lr0,
        total_steps=epochs * batches_per_epoch,
        cycles=cycles)
    lr_schduler = CyclicLRScheduler(schedule=lr_func)

    model.fit_generator(
        train_gen.flow(x_train, y_train, batch_size=batch_size),
        steps_per_epoch=batches_per_epoch,
        epochs=epochs,
        validation_data=test_gen.flow(x_val, y_val, batch_size=batch_size),
        callbacks=[lr_schduler, checkpoint, logger],
        verbose=verbose,
    )

    model = load_model(
        f'../models/{model_factory.__name__}/{mode}_model.h5',
        custom_objects={'soft_iou_loss': soft_iou_loss})

    pred = model.predict(x_test, verbose=verbose)
    inds = np.random.choice(np.arange(len(x_test)), size=5, replace=False)
    inds.sort()
    triplet_comparison_figure(
        x=x_test,
        y_true=y_test,
        y_pred=pred,
        inds=inds,
        output_path=f'../models/{model_factory.__name__}/figs/{mode}',
        # norm='log',
    )

    full_pred = model.predict_generator(
        test_gen.flow(x, batch_size=batch_size, shuffle=False),
        verbose=verbose,
    )
    prediction_to_fits(
        full_pred,
        ref_files=files,
        outpath=f'../models/{model_factory.__name__}/{mode}.fits',
    )

    error = model.evaluate(x_test, y_test)
    logging.info(f'Valid/Test Samples: {len(x_test)}\nTest Error: {error}\n\n')
    return error


def arg_parser():
    parser = argparse.ArgumentParser(description='Train an ANN to identify bubbles in astronomical images.')
    parser.add_argument('mode', choices=['classify_co', 'regress_co', 'regress_density'])

    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('-c', '--cycles', default=5, type=int)
    parser.add_argument('-e', '--epochs', default=5, type=int)
    parser.add_argument('-f', '--filters', default=16, type=int)
    parser.add_argument('-l', '--lr0', default=0.2, type=float)
    parser.add_argument('-m', '--model_factory', default='residual_u_net',
                        choices=['restrict_net', 'u_net', 'residual_u_net', 'dilated_net', 'dilated_residual_net'],
                        type=lambda x: getattr(arch, x))
    parser.add_argument('-n', '--noise_std', default=0., type=float)
    parser.add_argument('-v', '--verbose', action='count')
    return parser


if __name__ == '__main__':
    main(**vars(arg_parser().parse_args()))
