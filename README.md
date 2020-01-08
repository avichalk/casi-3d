# Convolutional Approach to Shell Identification - 3D (`CASI-3D`)


## Description
CASI-3D (Convolutional Approach to Shell Identification - 3D) is a deep learning method to identify signatures of stellar feedback in molecular line spectra, such as 13CO. CASI-3D is developed from [CASI-2D](https://casi-project.gitlab.io/casi-2d) ([Van Oort+2019](https://iopscience.iop.org/article/10.3847/1538-4357/ab275e/meta)) in order to exploit the full 3D spectral information. 

## Contents
 * data: .fits files containing 
 * models: Pre-trained models.
 * src: Source code for building, training, and evaluating shell identifiers.
    * network_architectures.py, network_components.py: Convolutional Neural Networks (CNNs) architecture.
    * preprocessing.py, preprocessing_log_binary2.py: Preprocessing data cubes. 
    * shell_identifier_3_adaptive_lr.py, shell_identifier_3_adaptive_lr_IoU.py: Main scripts.
    * output_analysis.py, pred_trainingset.py, visualizations.py, real_bubbles_test.py: Evaluate training result.











