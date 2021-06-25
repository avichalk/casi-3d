"""
Contains methods which prepare data so that it may be fed into predictive
models, as well as multiple helper functions which handle loading the data and
other similar tasks.
"""


import re
from pathlib import Path
import random

import numpy as np
from astropy.io import fits
import os


def density_preprocessing(x, y):
    x = normalize(x)
    y = normalize(y)

    x = np.sign(x) * np.log10(1 + np.abs(x))
    x /= np.std(x)
    y = np.sign(y) * np.log10(1 + np.abs(y))
    y /= np.std(y)

    x = slice_density(x)
    y = slice_density(y)

    x = np.expand_dims(x, axis=-1)
    y = np.expand_dims(y, axis=-1)
    return x, y

def co_preprocessing(data_path=''):

    tracer_files = get_co_tracer_files(data_path)
    co_files = get_co_files(data_path)

    co = load_fits(co_files)
    tracer = load_fits(tracer_files)
    dataset = co+tracer
    labels = [1]*len(co)+[0]*len(tracer)

    for i in range(len(dataset)): # randomizes position
        index1 = random.randint(0, len(dataset)-1)
        index2 = random.randint(0, len(dataset)-1)
        temp_dat = dataset[index1]
        dataset[index1] = dataset[index2]
        dataset[index2] = temp_dat
        temp_lab = labels[index1]
        labels[index1] = labels[index2]
        labels[index2] = temp_lab

    x = np.asarray(dataset)
    y = np.abs(np.asarray(labels)) # true or false for filament

    min_data = np.min(x)
    x = np.log(x - min_data + 1.)

    mean_data = np.mean(x)
    std_data = np.std(x)
    if std_data == 0:
        std_data = 1

    x = x - mean_data
    x = x / std_data
    
    x = np.expand_dims(x, axis=-1)

    return x, y

def get_co_files(data_path):
    return [str(x) for x in Path(f'{data_path}/yes').glob('*.fits')]

def get_co_tracer_files(data_path):
    return [str(x) for x in Path(f'{data_path}/no').glob('*.fits')]


def load_fits(files):
    output_data = []

    for file in files:
        with fits.open(file) as fits_data:
            output_data.append(np.reshape(fits_data[0].data, (100, 100))) 

    return output_data


def prediction_to_fits(pred, ref_files=None, outpath='../models/prediction.fits'):
    with fits.open(ref_files[0]) as fits_ref:
        ref_shape = fits_ref[0].data.shape
        fits_ref[0].data[:] = np.squeeze(pred)[:ref_shape[0], :ref_shape[1], :ref_shape[2]]
        fits_ref.writeto(outpath, overwrite=True)


#def pad_data(data, value=0.):
#    return np.pad(data,
#                  ((0, 0), (0, 1), (0, 1)),
#                  'constant',
#                  constant_values=value)


def slice_density(data):
    slices = np.empty((np.sum(data.shape), data.shape[1], data.shape[2]))
    slice_count = 0

    for i in range(data.shape[0]):
        slices[slice_count, :] = data[i, :, :]
        slice_count += 1

    for j in range(data.shape[1]):
        slices[slice_count, :] = data[:, j, :]
        slice_count += 1

    for k in range(data.shape[2]):
        slices[slice_count, :] = data[:, :, k]
        slice_count += 1

    return slices


def normalize(data):
    data -= np.mean(data)
    if np.std(data) ==0:
        data=data
    else:
        data /= np.std(data)

    return data
