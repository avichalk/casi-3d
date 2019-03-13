#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 13:59:43 2018

@author: xuduo
"""

import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from output_analysis import binarize
from preprocessing import get_co_files, load_fits
from shell_identifier_1 import ShellIdentifier


model = ShellIdentifier('unet_13co_test_3d_res3_ep60_log_noise_20180825', load=True)
x_all=np.load('/Users/xuduo/Desktop/project-CNN/Taurus/resample/maxpool5/all_image_crop.npy')
y_pred_all=x_all*0.0
x_all = np.where(np.isnan(x_all), np.ones(x_all.shape) * np.nanmean(x_all), x_all)
x_all = np.expand_dims(x_all, axis=-1)


x_all -=  np.min(x_all)
x_all = np.log(x_all + 1)
x_all -= np.mean(x_all)
x_all /= np.std(x_all)


#    for ctt in range(x_all.shape[0]):
#    for ctt in range(2):
for ctt in range(np.int_(x_all.shape[0]/10)):
    x=x_all[ctt*10:ctt*10+10,:,:,:]
    
    y_pred = model.predict(x)
    
    x=np.squeeze(x)
    y_pred=np.squeeze(y_pred)
    y_pred_all[ctt*10:ctt*10+10]=y_pred
   
ctt=35     
x=x_all[ctt*10:,:,:,:] 
if x.ndim ==3:
    x_all = np.expand_dims(x_all, axis=0)
y_pred = model.predict(x)


x=np.squeeze(x)
y_pred=np.squeeze(y_pred)
y_pred_all[ctt*10:,:,:,:]=y_pred
        
np.save('y_pred_all_maxpool5.npy',y_pred_all)
    
    
    
    
    
    
    
    