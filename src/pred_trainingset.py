#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 21:41:50 2019

@author: xuduo
"""


import sys
import numpy as np
from shell_identifier_3_adaptive_lr import ShellIdentifier

def main():

    if len(sys.argv) == 2:
        model_name = sys.argv[1]
    else:
        model_name = 'test'

    model = ShellIdentifier(model_name, load=True)

    x, y = x,y=np.load('../data/temp_co/12co_log_noise_1102.npy')
    
    pred = model.predict(x,batch_size=8)

    error = model.evaluate(y, pred)

    print(f'Total error of final model: {error}\n\n')

    np.savez_compressed(f'../data/ModelOutputs/{model_name}_outputs',
                        X=x,
                        Y=y,
                        P=pred)



if __name__ == '__main__':
    main()









