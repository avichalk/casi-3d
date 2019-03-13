"""
Implements methods which search the hyper-parameter space of models used in the
NSB project in order to find more optimal model configurations.
"""


import traceback
from datetime import datetime
from pathlib import Path
from random import choice
from shutil import copyfile

import numpy as np
import pandas as pd
from keras import backend as K

import dilated_res_net


def main():
    hyper_search(iters=50)


def hyper_search(iters,
                 log_path='../models/dilated_res_net/search_params.csv'):
    """
    Randomly explores the hyper-parameter space, saves the best model, and
    outputs runtime statistics.
    Logs configurations and their performance to a csv file.

    Args:
        iters (int) Number of parameter sets to test.

        log_path (str) Path to the search history log.
    """
    with open('../models/dilated_res_net/search.log', 'w') as f:
        f.write('')

    if Path(log_path).is_file():
        hyper_log = pd.read_csv(log_path)
        best_error = np.min(hyper_log.error)
        best_params = hyper_log.iloc[np.argmin(hyper_log.error)]
    else:
        hyper_log = pd.DataFrame()
        best_error = np.inf
        best_params = {}

    for i in range(iters):
        print(f'Search progress: {i+1}/{iters}\n\tBest Error={best_error}')
        hypers = {
            'epochs': 100,
            'batch_size': 16,
            'filters': 8,
            'noise_std': np.random.exponential(scale=0.1),
            'rot_range': np.random.randint(low=361),
            'w_shift_range': np.random.exponential(scale=0.1),
            'h_shift_range': np.random.exponential(scale=0.1),
            'channel_shift_range': np.random.exponential(scale=0.01),
            'h_flip': choice([True, False]),
            'v_flip': choice([True, False]),
            'zoom_range': np.random.exponential(scale=0.1)}

        try:
            error = dilated_res_net.main(**hypers, display=False)
        except KeyboardInterrupt:
            break
        except:
            with open('../models/dilated_res_net/search.log', 'a') as f:
                f.write(f'{datetime.now()}\n{traceback.format_exc()}\n\n')
            error = np.inf

        hypers['timestamp'] = datetime.now()
        hypers['error'] = error

        if error < best_error:
            print(f'Minimum error updated: {error}\n')
            best_error = error
            best_params = hypers
            copyfile('../models/dilated_res_net/model.h5',
                     '../models/dilated_res_net/search_model.h5')
            copyfile('../models/dilated_res_net/training_log.csv',
                     '../models/dilated_res_net/search_errors.csv')

        hyper_log = pd.concat([hyper_log, pd.DataFrame([hypers])],
                              ignore_index=True)
        hyper_log.to_csv(log_path, index=False)

        # Frees resources for the next iteration
        K.clear_session()
    print(f'Hyper-parameter sweep complete. Best parameters\n{best_params}')


if __name__ == '__main__':
    main()
