#! /usr/bin/python3
import os
import matplotlib.pyplot as plt


def plot_energy_history(date, n_site, n_boson, t, U, V=None, dense_hidden_units=None, n_step, energy_history):
    #(デバッグ中)
    if not V:
        if dense_hidden_units:
            save_path = savefile_path_BH_Dense(date, n_site, n_boson, t, U, dense_hidden_units)

    file_path = save_path + '.png'

    fig_overall = plt.figure()
    ax = fig_overall.add_subplot()
    ax.set_title('overall'), ax.set_xlabel('number of training'), ax.set_ylabel('eloc_mean / J')
    ax.plot(
        np.arange(n_step),
        energy_history,
    )

    if not os.path.exists():

    fig_overall.savefig(file_path)

def savefile_path_BH_Dense(date, n_site, n_boson, t, U, hidden_units):
    current_path = os.getcwd()
    savedir_path = current_path + '/tmp_data'
    file_name = 'BH'\
        + '_{0:%Y}{0:%m}{0:%d}_{0:%H}{0:%M}{0:%S}'.format(date) \
        + '_site{:>03d}_bosons{:>03d}_t{}_U{}'.format(n_site, n_boson, t, U) \
        + '_dense{}'.format(hidden_units)
    save_path = savedir_path + '/' + file_name
    return save_path

