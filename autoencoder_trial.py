import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from astropy.io import fits
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Input, Dropout, Conv2D, UpSampling2D, MaxPooling2D, Concatenate
from keras.models import Model, model_from_json
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.python.client import device_lib

from network_components import res_bottlneck
from visualizations import training_data_video


def main():
    name = 'res_unet_auto'

    with open('hypers.json', 'r') as f:
        params = json.load(f)

    model_hypers = params['model_hypers']
    train_hypers = params['train_hypers']
    hypers = {**model_hypers, **train_hypers}

    with fits.open('../data/Observational/perseus_b5_12co_nnorm.fits') as fits_data:
        x = fits_data[0].data
    x = np.where(np.isnan(x), np.nanmean(x), x)
    x = np.pad(x,
               ((0, 0), (76, 0), (76, 0)),
               'constant',
               constant_values=0.)
    x = np.expand_dims(x, axis=-1)
    x_train, x_test, _, _ = train_test_split(x, x, test_size=0.1)

    model = ShellIdentifier(name, model_hypers=model_hypers)
    model.fit(x_train, x_train, **train_hypers)

    test_error = model.evaluate(x_test,
                                x_test,
                                batch_size=hypers['batch_size'])
    error = model.evaluate(x, x)
    print(f'Test error of trained model: {test_error}\n\n')
    print(f'Total error of final model: {error}\n\n')

    rot_x = np.rot90(x, axes=(1, 2))
    rot_x_test = np.rot90(x_test, axes=(1, 2))
    rot_test_error = model.evaluate(rot_x_test,
                                    rot_x_test,
                                    batch_size=hypers['batch_size'])
    rot_error = model.evaluate(rot_x, rot_x)
    print(f'Test error on rotated data: {rot_test_error}\n\n')
    print(f'Total error on rotated data: {rot_error}\n\n')

    training_data_video(x,
                        model.predict(x),
                        output_name='encoder_total',
                        labels=['Data', 'Predicted'])
    training_data_video(rot_x,
                        model.predict(rot_x),
                        output_name='rot_encoder_total',
                        labels=['Data', 'Predicted'])
    training_data_video(x_test,
                        model.predict(x_test),
                        output_name='encoder_test',
                        labels=['Data', 'Predicted'])
    training_data_video(rot_x_test,
                        model.predict(rot_x_test),
                        output_name='rot_encoder_test',
                        labels=['Data', 'Predicted'])


class ShellIdentifier:
    def __init__(self,
                 name,
                 model_hypers=None,
                 load=False):
        self.name = name
        self.gpu_count = get_gpu_count()

        if load and not model_hypers:
            self.load_init(name)
        else:
            self.new_init(model_hypers)

    def new_init(self, model_hypers):
        if self.gpu_count > 1:
            with tf.device('/cpu:0'):
                self.model = self.build_model(**model_hypers)
            self.multi_gpu_model = multi_gpu_model(self.model,
                                                   gpus=self.gpu_count)
            self.multi_gpu_model.compile(optimizer=SGD(lr=0.005, momentum=0.9), loss='mse')
        else:
            self.model = self.build_model(**model_hypers)
            self.model.compile(optimizer=SGD(lr=0.005, momentum=0.9), loss='mse')

    def load_init(self, name):
        if self.gpu_count > 1:
            with tf.device('/cpu:0'):
                self.model = load_model(name)
            self.multi_gpu_model = multi_gpu_model(self.model,
                                                   gpus=self.gpu_count)
            self.multi_gpu_model.compile(optimizer=SGD(lr=0.005, momentum=0.9), loss='mse')
        else:
            self.model = load_model(name)
            self.model.compile(optimizer=SGD(lr=0.005, momentum=0.9), loss='mse')

    def fit(self, x, y, epochs=1, batch_size=64, verbose=1):
        if self.gpu_count > 1:
            model = self.multi_gpu_model
            batch_size = batch_size * self.gpu_count
        else:
            model = self.model

        x_train, x_val, y_train, y_val = train_test_split(x,
                                                          y,
                                                          test_size=0.11)
        gen = ImageDataGenerator(rotation_range=90,
                                 # width_shift_range=0.1,
                                 # height_shift_range=0.1,
                                 # horizontal_flip=True,
                                 # vertical_flip=True,
                                 fill_mode='constant',
                                 cval=0.)

        csv_logger = CSVLogger(f'../logs/{self.name}_training.csv',
                               append=True)
        checkpoint = ModelCheckpoint(f'../models/{self.name}.h5',
                                     save_weights_only=True,
                                     save_best_only=True)

        model.fit_generator(gen.flow(x_train, y_train, batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] / batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            callbacks=[csv_logger, checkpoint],
                            validation_data=(x_val, y_val))

        model.load_weights(f'../models/{self.name}.h5')

        self.save()

        return self

    def predict(self, x, batch_size=64):
        if self.gpu_count > 1:
            model = self.multi_gpu_model
            batch_size = batch_size * self.gpu_count
        else:
            model = self.model

        preds = []
        for chunk in pred_generator(x):
            preds.append(model.predict(chunk, batch_size=batch_size))

        return np.concatenate(preds)

    def evaluate(self, x, y, batch_size=64):
        if self.gpu_count > 1:
            model = self.multi_gpu_model
            batch_size = batch_size * self.gpu_count
        else:
            model = self.model

        return model.evaluate(x, y, batch_size=batch_size)

    def build_model(self,
                    filters=16,
                    drop_rate=0.1,
                    activation='selu',
                    last_activation='selu'):
        inputs = Input(shape=(256, 256, 1))
        pred = Dropout(rate=drop_rate)(inputs)

        pred = res_bottlneck(pred,
                             filters=filters,
                             activation=activation,
                             shortcut_conv=True)
        cross1 = pred

        pred = MaxPooling2D()(pred)
        filters *= 2
        pred = Dropout(rate=drop_rate)(pred)
        pred = res_bottlneck(pred,
                             filters=filters,
                             activation=activation,
                             shortcut_conv=True)
        cross2 = pred

        pred = MaxPooling2D()(pred)
        filters *= 2
        pred = Dropout(rate=drop_rate)(pred)
        pred = res_bottlneck(pred,
                             filters=filters,
                             activation=activation,
                             shortcut_conv=True)
        cross3 = pred

        pred = MaxPooling2D()(pred)
        filters *= 2
        pred = Dropout(rate=drop_rate)(pred)
        pred = res_bottlneck(pred,
                             filters=filters,
                             activation=activation,
                             shortcut_conv=True)
        cross4 = pred

        pred = MaxPooling2D()(pred)
        filters *= 2
        pred = Dropout(rate=drop_rate)(pred)
        pred = res_bottlneck(pred,
                             filters=filters,
                             activation=activation,
                             shortcut_conv=True)
        pred = UpSampling2D()(pred)
        filters //= 2
        pred = res_bottlneck(pred,
                             filters=filters,
                             activation=activation,
                             shortcut_conv=True)

        pred = Concatenate()([pred, cross4])
        pred = res_bottlneck(pred,
                             filters=filters,
                             activation=activation,
                             shortcut_conv=True)
        pred = UpSampling2D()(pred)
        filters //= 2
        pred = res_bottlneck(pred,
                             filters=filters,
                             activation=activation,
                             shortcut_conv=True)

        pred = Concatenate()([pred, cross3])
        pred = res_bottlneck(pred,
                             filters=filters,
                             activation=activation,
                             shortcut_conv=True)
        pred = UpSampling2D()(pred)
        filters //= 2
        pred = res_bottlneck(pred,
                             filters=filters,
                             activation=activation,
                             shortcut_conv=True)

        pred = Concatenate()([pred, cross2])
        pred = res_bottlneck(pred,
                             filters=filters,
                             activation=activation,
                             shortcut_conv=True)
        pred = UpSampling2D()(pred)
        filters //= 2
        pred = res_bottlneck(pred,
                             filters=filters,
                             activation=activation,
                             shortcut_conv=True)

        pred = Concatenate()([pred, cross1])
        pred = res_bottlneck(pred,
                             filters=filters,
                             activation=activation,
                             shortcut_conv=True)

        pred = Conv2D(1, (1, 1), activation=last_activation)(pred)
        return Model(inputs=inputs, outputs=pred)

    def save(self):
        with open(f'../models/{self.name}.json', 'w') as f:
            f.write(self.model.to_json())

        self.model.save_weights(f'../models/{self.name}.h5')


def log_hypers(log_path, hypers):
    if Path(log_path).is_file():
        hyper_log = pd.read_csv(log_path)
    else:
        hyper_log = pd.DataFrame()

    hypers["timestamp"] = datetime.datetime.now()

    hyper_log = pd.concat([hyper_log, pd.DataFrame(hypers, index=[0])])
    hyper_log = hyper_log[hyper_log['epochs'] >= 10]

    hyper_log.to_csv(log_path, index=False)


def chunk_generator(x, y, chunk_size=256):
    x, y = shuffle(x, y)

    for i in range(0, x.shape[0], chunk_size):
        yield x[i:i + chunk_size], y[i:i + chunk_size]


def load_model(name, optimizer='nadam', loss='mse'):
    with open(f'../models/{name}.json', 'r') as f:
        model = model_from_json(f.read())

    model.compile(optimizer=optimizer,
                  loss=loss)

    model.load_weights(f'../models/{name}.h5')

    return model


def pred_generator(x, chunk_size=256):
    for i in range(0, x.shape[0], chunk_size):
        yield x[i:i + chunk_size]


def get_gpu_count():
    devices = device_lib.list_local_devices()
    return len([x.name for x in devices if x.device_type == 'GPU'])


if __name__ == '__main__':
    main()
