# -*- coding: utf-8 -*-

import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from output_analysis import binarize
from preprocessing import get_co_files, load_fits
from shell_identifier_3_adaptive_lr import ShellIdentifier
import time

def main():
    if len(sys.argv) == 2:
        model_name = sys.argv[1]
    else:
        model_name = 'test'

#    analyze_ngcc1333(model_name)
#    analyze_perseus_b5('unet_13co_test_3d_res3_ep60_log_0913_newset')
#    analyze_taurus('unet_13co_test_3d_res3_ep150_log_0913')
#    analyze_taurus('unet_13co_test_3d_res3_ep60_log_noise_20180825')
#    analyze_taurus('unet_13co_test_3d_res3_ep60_log_0913_newset')
#    analyze_taurus('unet_newset_20181107_ep120_adaptive_lr')
#    analyze_taurus('unet_newset_20181107_ep250')
#    analyze_taurus('unet_newset_20190108_ep277_log_adaptive_lr')
    
    
    analyze_taurus_all(model_name)
 
#    analyze_taurus_all('unet_fraction_20190320_ep277_log_IoU')
    
   # make_fits_bigmap_prediction()
        
def analyze_taurus_all(model_name):
    #file_path='../../../../../groups/yshirley/cnntrain/no'

    model = ShellIdentifier(model_name, load=True)
    file_path=''
#    model = ShellIdentifier('unet_13co_test_3d_res2_20180723', load=True)
        
    
#    for bubble_num in range(37):
#    file_name=''
    
#    with fits.open(''+file_name+'.fits') as f:
#        x = f[0].data
#    x=fits.open('/Users/xuduo/Desktop/project-CNN/radmc/co13_3d_all_20180806/co13_0980_y27_99_conv1_rot0.fits')[0].data 
#    bubble_13co_tracer=fits.open('/Users/xuduo/Desktop/project-CNN/radmc/co13_file_conv/co13_0980_y21_99_tracer_conv4.fits')[0].data 
#    x=np.load(file_path+'all_image_crop_step10.npy')


#    for ctt_1 in range(
    x = 0
    with fits.open(f'test_fit_file.fits') as fits_data:
        x = np.reshape(fits_data[0].data, (100, 100))


    for ctt_1 in range(1):
#    for ctt_1 in range(1):
        #x=np.load(file_path+'all_image_rescale/all_image_crop_step5_'+str(ctt_1)+'.npy')
#        x=np.load(file_path+'all_image_crop_step_16_renorm.npy')
        x = np.asarray(x)
        x = np.where(np.isnan(x), np.ones(x.shape) * np.nanmean(x), x)
        x = np.expand_dims(x, axis=-1)
        x = np.expand_dims(x, axis=0)
    
#        x -=  np.min(x)
#        x = np.log(x + 1)
#        x -= np.mean(x)
#        x /= np.std(x)
#        batch_size=x.shape[0]
#        y_pred=x*0.0+0.0
        y_pred = model.predict(x,batch_size=8)
        
#        n_time=np.int(batch_size/7)
#        residual_time=np.int(batch_size-n_time*7)
#        for ctt in range(n_time):
#            y_pred[ctt*7:(ctt+1)*7] = model.predict(x[ctt*7:(ctt+1)*7])
#        if residual_time >0:
#            y_pred[n_time*7:] = model.predict(x[n_time*7:])
        
        
#        x=np.squeeze(x)
        y_pred=np.squeeze(y_pred)
        #np.save(file_path+'pred_frac_20190317/pred_all_image_crop_step5_'+str(ctt_1)+'.npy',y_pred)
        print(y_pred)
#        np.save(file_path+'pred_all_image_crop_step_16.npy',y_pred)
#    fits.writeto('../output_fits/pred_{}.fits'.format('co13_0980_y27_99_conv1_rot0'),y_pred,overwrite=True)
        


def make_fits_bigmap_prediction():
    
    file_path='/work/05184/xuduo117/project_CNN/NeuralSpaceBubbles/data_taurus'
    
    step_size=5
    
    image_pooling=fits.open(file_path+'/co13_resample_avgpool_3_with_hdr.fits')[0].data
    
    x_shape=image_pooling.shape[2]
    y_shape=image_pooling.shape[1]
    
    x_num=np.int_(x_shape/step_size)
    y_num=np.int_(y_shape/step_size)
    
    pred_all_bigmap=image_pooling[:,:,:]*0.0-100
    
    for ctt_x in range(x_num+1):
        pred_all=np.load(file_path+'/pred_frac_20190317/pred_all_image_crop_step5_'+str(ctt_x)+'.npy')
        for ctt_y in range(y_num+1):
            y_s,x_s=get_boundary_inverse(image_pooling,ctt_x,ctt_y,step_size,x_shape,y_shape)
            old_crop=pred_all_bigmap[:,y_s:y_s+64,x_s:x_s+64]
            new_crop=pred_all[ctt_y,:,:,:]
            combine_crop=np.maximum(old_crop,new_crop)
    #        combine_crop[indx_overlap]=combine_crop[indx_overlap]/2
            pred_all_bigmap[:,y_s:y_s+64,x_s:x_s+64]=combine_crop
        

    fits.writeto(file_path+'/pred_big_fits/pred_taurus_12co_subgrad_ME1_1103.fits',pred_all_bigmap,overwrite='True')


def get_boundary_inverse(image,x_s,y_s,step_size,x_shape,y_shape):
    x_s=x_s*step_size
    y_s=y_s*step_size
    if x_s+64>x_shape:
        x_s=x_shape-64
    if y_s+64>y_shape:
        y_s=y_shape-64
        
    return y_s,x_s


if __name__ == '__main__':
    main()
