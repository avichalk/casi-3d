import datetime
import json
import sys
from pathlib import Path

import keras.backend as K
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import LearningRateScheduler,CSVLogger, ModelCheckpoint,EarlyStopping
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from vailtools import losses


import network_architectures as arch
#from preprocessing_log_binary2 import co_preprocessing, density_preprocessing
from preprocessing_log_binary import co_preprocessing, density_preprocessing
#from preprocessing_linear_binary import co_preprocessing, density_preprocessing


def main():

    if len(sys.argv) == 2:
        name = sys.argv[1]
    else:
        name = 'test'

    with open('hypers_3.json', 'r') as f:
        params = json.load(f)

    model_hypers = params['model_hypers']
    train_hypers = params['train_hypers']
    data_hypers = params['data_hypers']
    hypers = {**model_hypers, **train_hypers, **data_hypers}

    if 'co' in data_hypers['data_path'].lower():
        preprocessing = co_preprocessing
    elif 'density' in data_hypers['data_path'].lower():
        preprocessing = density_preprocessing
    else:
        raise ValueError(
            f"Incorrect data path, given {data_hypers['data_path']}"
        )

    x, y = preprocessing(data_path=data_hypers['data_path'])

    model = ShellIdentifier(name, model_hypers=model_hypers)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.205)

    model.fit(x_train, y_train, **train_hypers)

    error = model.evaluate(x_test,
                           y_test,
                           batch_size=hypers['batch_size'])

    hypers["error"] = error
    log_hypers('../models/hypers_3.csv', hypers)

    print(f'Test error of trained model: {error}\n\n')

    pred = model.predict(x)

    error = model.evaluate(y, pred)

    print(f'Total error of final model: {error}\n\n')

    np.savez_compressed(f'../data/ModelOutputs/{name}_outputs',
                        X=x,
                        Y=y,
                        P=pred)

    
class ShellIdentifier:
    def __init__(self,
                 name,
                 model_hypers=None,
                 load=False):
        self.name = name
        self.gpu_count = get_gpu_count()
        self.model = None
        self.multi_gpu_model = None

        if load and (model_hypers is None):
            self.load_init(name)
        else:
            self.new_init(model_hypers)

    def new_init(self, model_hypers):
        if self.gpu_count > 1:
            with tf.device('/cpu:0'):
                self.model = self.build_model(**model_hypers)

            self.multi_gpu_model = multi_gpu_model(self.model,
                                                   gpus=self.gpu_count)
            self.multi_gpu_model.compile(optimizer=SGD(lr=0.02, momentum=0.9), loss=losses.iou_loss)
        else:
            self.model = self.build_model(**model_hypers)
            self.model.compile(optimizer=SGD(lr=0.02, momentum=0.9), loss=losses.iou_loss)

    def load_init(self, name):
        if self.gpu_count > 1:
            with tf.device('/cpu:0'):
                self.model = load_model(name)

            self.multi_gpu_model = multi_gpu_model(self.model,
                                                   gpus=self.gpu_count)
            self.multi_gpu_model.compile(optimizer=SGD(lr=0.02, momentum=0.9), loss=losses.iou_loss)
        else:
            self.model = load_model(name)
            self.model.compile(optimizer=SGD(lr=0.02, momentum=0.9), loss=losses.iou_loss)

    def fit(self, x, y, epochs=1, batch_size=64, verbose=1):
        if self.gpu_count > 1:
            model = self.multi_gpu_model
            batch_size = batch_size * self.gpu_count
        else:
            model = self.model

        x_train, x_val, y_train, y_val = train_test_split(x,
                                                          y,
                                                          test_size=0.255)
#        gen = ImageDataGenerator(rotation_range=5,
#                                 width_shift_range=0.1,
#                                 height_shift_range=0.1,
#                                 horizontal_flip=True,
#                                 vertical_flip=True,
#                                 fill_mode='constant',
#                                 cval=0.)

        csv_logger = CSVLogger(f'../logs/{self.name}_training.csv',
                               append=True)
        checkpoint = ModelCheckpoint(f'../models/{self.name}.h5',
                                     save_weights_only=True,
                                     save_best_only=True)

#        model.fit_generator(gen.flow(x_train, y_train, batch_size=batch_size),
#                            steps_per_epoch=x_train.shape[0] / batch_size,
#                            epochs=epochs,
#                            verbose=verbose,
#                            callbacks=[csv_logger, checkpoint],
#                            validation_data=(x_val, y_val))

        def scheduler(epoch):
            lrate=K.get_value(model.optimizer.lr)
            if epoch % 19==0:
                K.set_value(model.optimizer.lr, lrate/2.0)
            if epoch % 31==0:
                K.set_value(model.optimizer.lr, lrate*1.5)
            return K.get_value(model.optimizer.lr) 
       
        change_lr = LearningRateScheduler(scheduler)
        early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001,patience=100, mode='auto') 
        
        model.fit(x=x_train,y=y_train,
                  batch_size=batch_size,          
                            epochs=epochs,
                            verbose=verbose,
                            callbacks=[csv_logger, checkpoint,change_lr,early_stop],
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
                    noise_std=0.1,
                    activation='selu',
                    last_activation='selu'):
        return arch.residual_u_net(filters=filters,
                                   noise_std=noise_std,
                                   activation=activation,
                                   final_activation=last_activation,
                                   )
#        return arch.dilated_res_net(filters=filters,
#                                   noise_std=noise_std,
#                                   activation=activation,
#                                   final_activation=last_activation
#                                   )
            

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


def load_model(name, optimizer='nadam', loss=losses.iou_loss):
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
