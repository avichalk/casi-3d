"""
Generates several useful visualizations for the NSB project.
"""

import subprocess
import sys


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
from keras.models import model_from_json
from keras.utils import plot_model


def main():
    if len(sys.argv) == 2:
        name = sys.argv[1]
    else:
        name = 'test'

#    convergence_visualization(name)
#    nsf_proposal_figure_5(name)
#    spatial_error_distribution(name)
    data_path = f'../data/ModelOutputs/{name}_outputs.npz'

    with np.load(data_path) as data:
        x, y_true, y_pred = data['X'], data['Y'], data['P']
        
    comparison_video(x, y_true, y_pred, '3d_'+name)


def spatial_error_distribution(model_name):
    data_path = f'../data/ModelOutputs/{model_name}_outputs.npz'

    with np.load(data_path) as data:
        y_true, y_pred = data['Y'], data['P']

    spatial_mae = np.mean(np.abs(y_true - y_pred), axis=0)

    fig, ax = plt.subplots()

    ax.set_title('Spatial MAE')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    cax = ax.imshow(np.squeeze(spatial_mae), cmap='Greys')
    fig.colorbar(cax)

    fig.savefig(f'../visualizations/{model_name}_mae_dist.png')

    spatial_mse = np.mean(np.power(y_true - y_pred, 2), axis=0)

    fig, ax = plt.subplots()

    ax.set_title('Spatial MSE')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    cax = ax.imshow(np.squeeze(spatial_mse), cmap='Greys')
    fig.colorbar(cax)

    fig.savefig(f'../visualizations/{model_name}_mse_dist.png')


def nsf_proposal_figure_5(model_name):
    data_path = f'../data/ModelOutputs/{model_name}_outputs.npz'

    with np.load(data_path) as data:
        x, y_true, y_pred = data['X'], data['Y'], data['P']

    # co_indicies
    # inds = [667, 108, 883]

    # density indicies
    inds = [200, 400, 600]
    
    x = x[inds]
    y_pred = y_pred[inds]
    y_true = y_true[inds]

    f, axes = plt.subplots(len(inds), 3, figsize=(7, 7))

    arrays = [np.squeeze(arr) for arr in (x, y_true, y_pred)]

    vmin = np.min(np.c_[x, y_true, y_pred])
    vmax = np.max(np.c_[x, y_true, y_pred]) / 4

    for i in range(len(inds)):
        for j in range(3):
            axes[i, j].imshow(arrays[j][i], vmin=vmin, vmax=vmax, cmap='Greys')
            axes[i, j].get_xaxis().set_ticks([])
            axes[i, j].get_yaxis().set_ticks([])

    axes[0, 0].set_title('$^{12}$CO Data')
    axes[0, 0].title.set_fontsize(20)

    axes[0, 1].set_title('Tracer')
    axes[0, 1].title.set_fontsize(20)

    axes[0, 2].set_title('Prediction')
    axes[0, 2].title.set_fontsize(20)

    axes[0, 0].set_ylabel('1', rotation=0, labelpad=20)
    axes[0, 0].yaxis.label.set_fontsize(20)

    axes[1, 0].set_ylabel('2', rotation=0, labelpad=20)
    axes[1, 0].yaxis.label.set_fontsize(20)

    axes[2, 0].set_ylabel('3', rotation=0, labelpad=20)
    axes[2, 0].yaxis.label.set_fontsize(20)

    plt.savefig(f"../visualizations/nsf_prop_fig_5_{model_name}.png")
    plt.close()


def training_data_video(x, y, output_name, labels=None):
    if labels is None:
        labels = ['Data', 'Target']
    x, y = [np.squeeze(arr) for arr in (x, y)]

    vmin = min(x.min(), y.min())
    vmax = max(x.max(), y.max())

    for i in range(x.shape[0]):
        f, axes = plt.subplots(1, 2)
        axes[0].imshow(x[i], vmin=vmin, vmax=vmax)
        axes[0].set_title(f'{labels[0]} {i}')

        axes[1].imshow(y[i], vmin=vmin, vmax=vmax)
        axes[1].set_title(f'{labels[1]} {i}')
        
        for ax in axes:
            ax.axis('off')

        plt.savefig(f'slice_{i:04d}.png')
        plt.close()

    subprocess.run(f'ffmpeg -framerate 30 -y -i slice_%04d.png ../visualizations/Videos/{output_name}.mp4', shell=True)
    subprocess.run('rm slice*', shell=True)


def observational_data_video(x, output_name):
    x = np.squeeze(x)

    vmin = x.min()
    vmax = x.max()

    for i in range(x.shape[0]):
        f, ax = plt.subplots(1, 1)
        ax.imshow(x[i], vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(f'Velocity Slice {i}')
        ax.axis('off')

        plt.savefig(f'slice_{i:04d}.png')
        plt.close()

    subprocess.run(f'ffmpeg -framerate 30 -i -y slice_%04d.png ../visualizations/Videos/{output_name}.mp4', shell=True)
    subprocess.run('rm slice*', shell=True)


def convergence_visualization(model_name):
    df = pd.read_csv(f'../logs/{model_name}_training.csv')
    df.columns = ['Epoch', 'Training', 'Validation']
    df.set_index('Epoch').plot(alpha=0.8)

    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.title('Model Convergence')

    plt.ylim(-0.1, 1.)

    plt.savefig(f'../visualizations/{model_name}_convergence.png')
    plt.close()


def comparison_video(x, y_true, y_pred, output_name):
    x, y_true, y_pred = [np.squeeze(arr) for arr in (x, y_true, y_pred)]

    vmin = min(x.min(), y_true.min(), y_pred.min())
    vmax = max(x.max(), y_true.max(), y_pred.max())

    # Try shifting the color map to improve contrast
#    vmax -= vmin
#    vmin *= 2.

    for i in range(int(x.shape[0]/50)):
#    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            f, axes = plt.subplots(1, 3)
#            axes[0].imshow(x[i][j], vmin=vmin, vmax=vmax)
            axes[0].imshow(x[i][j])
            axes[0].set_title(f'Data {i}')
    
#            axes[1].imshow(y_true[i][j], vmin=vmin, vmax=vmax)
            axes[1].imshow(y_true[i][j],vmin=0, vmax=1)
            axes[1].set_title(f'Target {i}')
    
#            axes[2].imshow(y_pred[i][j], vmin=vmin, vmax=vmax)
            im3=axes[2].imshow(y_pred[i][j],vmin=0, vmax=1)
            axes[2].set_title(f'Prediction {i}')
            f.colorbar(im3, ax=axes.ravel().tolist(), shrink=0.33)

            for ax in axes:
                ax.axis('off')

            plt.savefig(f'slice_{i*x.shape[1]+j:05d}.png')
            plt.close()

    subprocess.run(f'ffmpeg -framerate 30 -y -i slice_%05d.png ../visualizations/Videos/{output_name}.mp4', shell=True)
    subprocess.run('rm slice*', shell=True)


def visualize_model(name, out_path='../visualizations'):
    with open(f'../ModelConfigs/{name}.json', 'r') as f:
        model = model_from_json(f.read())

    plot_model(model,
               to_file=f'{out_path}/{name}_TB.png',
               show_shapes=True,
               rankdir='TB')
    plot_model(model,
               to_file=f'{out_path}/{name}_LR.png',
               show_shapes=True,
               rankdir='LR')


def triplet_comparison_figure(x, y_true, y_pred, inds, output_path, norm=None):
    x = np.squeeze(x)
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    vmin = np.amin(np.c_[y_true[inds], y_pred[inds]])
    vmax = np.amax(np.c_[y_true[inds], y_pred[inds]])

    if norm == 'log':
        x_norm = colors.SymLogNorm(vmin=x.min(), vmax=x.max(), linthresh=0.03)
        y_norm = colors.SymLogNorm(vmin=vmin, vmax=vmax, linthresh=0.03)
    else:
        x_norm, y_norm = None, None

    for ind in inds:
        f, axes = plt.subplots(1, 3)
        axes[0].imshow(x[ind], cmap='Greys', norm=x_norm)
        axes[0].set_title('Input')

        axes[1].imshow(y_true[ind], vmin=vmin, vmax=vmax, cmap='Greys', norm=y_norm)
        axes[1].set_title('Target')

        axes[2].imshow(y_pred[ind], vmin=vmin, vmax=vmax, cmap='Greys', norm=y_norm)
        axes[2].set_title('Prediction')

        for ax in axes:
            ax.axis('off')

        plt.savefig(f'{output_path}_{ind}.png')
        plt.close(f)


def psnr(y_true, y_pred):
    return 10 * np.log10(np.amax(np.c_[y_true, y_pred]) ** 2 / mse(y_true, y_pred))


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


if __name__ == '__main__':
    main()
