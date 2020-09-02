# Convolutional Approach to Structure Identification - 3D (`CASI-3D`)


## Description
[CASI-3D](https://arxiv.org/abs/2001.04506) (Convolutional Approach to Structure Identification - 3D) is a deep learning method to identify signatures of stellar feedback in molecular line spectra, such as 12CO and 13CO. CASI-3D is developed from [CASI-2D](https://casi-project.gitlab.io/casi-2d) ([Van Oort+2019](https://iopscience.iop.org/article/10.3847/1538-4357/ab275e/meta)) in order to exploit the full 3D spectral information. 

## Contents
 * data: .fits files containing several training samples. The entire training set will be shared upon request.
 * models: Pre-trained models. ME1 is trained to predict only the position of feedback. MF is trained to predict the fraction of the mass coming from feedback in each voxel.
 * src: Source code for building, training, and evaluating shell identifiers.
    * network_architectures.py, network_components.py: Convolutional Neural Networks (CNNs) architecture.
    * preprocessing.py, preprocessing_log_binary2.py: Preprocessing data cubes. 
    * shell_identifier_3_adaptive_lr.py, shell_identifier_3_adaptive_lr_IoU.py: Main scripts.
    * output_analysis.py, pred_trainingset.py, visualizations.py, real_bubbles_test.py: Evaluate training result.

## Getting Started
### Training CASI-3D
```bash
python shell_identifier_3_adaptive_lr.py model_name
```
The hyper-parameters are set in hypers_3.json. The user can modify the CNNs architecture in network_architectures.py. 

### Applying CASI-3D to new data
```bash
python real_bubbles_test.py model_name
```
The CASI-3D models (model ME1 and model MF) require the new 13CO data with a shape of 64 * 64 * 32 (position-position-velocity). The region should have an area close to 2.7 pc * 2.7 pc. The velocity resolution should be close to 0.25 km/s. The user can change the path of the new data in real_bubbles_test.py. Both CASI-3D models are trained on bubbles driven by low- or intermediate- mass young stars. Both models have been successfully tested on the Taurus molecular cloud ([Xu+ 2020](https://arxiv.org/abs/2001.04506)) and the Perseus molecular cloud (Xu+ in prep). 


## Requirements
python 3.6

astropy 3.2.1

keras 2.2.4

numpy 1.16.4

pandas 0.24.2

scikit-learn 0.21.2

scipy 1.3.0

tensorflow 1.14.0











