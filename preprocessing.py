"""
Contains methods which prepare data so that it may be fed into predictive
models, as well as multiple helper functions which handle loading the data and
other similar tasks.
"""


import re
from pathlib import Path

import numpy as np
from astropy.io import fits


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


def co_preprocessing(data_path='../data/CleanCO'):
    tracer_files = get_co_tracer_files(data_path)
    co_files = get_co_files(tracer_files)

    co = load_fits(co_files)
    tracer = load_fits(tracer_files)

    x = np.asarray(co)
    y = np.asarray(tracer)
    
#    min_xy=np.min([np.min(x),np.min(y)])
    
    x = np.log(x - np.min(x) + 1.)
    y = np.log(y - np.min(y) + 1.)

    x = normalize(x)
    y = normalize(y)

#    x = pad_data(x)
#    y = pad_data(y)

    x = np.expand_dims(x, axis=-1)
    y = np.expand_dims(y, axis=-1)

    return x, y


def get_co_files(tracer_files):
    return [x.replace('_tracer', '') for x in tracer_files]



def get_co_tracer_files(data_path):
    return [str(x) for x in Path(data_path).glob('*.fits')
            if re.match(r'.*tracer.*', str(x))]


def load_fits(files):
    output_data = []

    for file in files:
        with fits.open(file) as fits_data:
            output_data.append(fits_data[0].data)

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
