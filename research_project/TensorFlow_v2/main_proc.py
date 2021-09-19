#! /usr/bin/python3
import datetime
import numpy as np

from network import Network
from state import SampledState
from energy import local_energy
# import utils

BHM = 0
eBHM = 1

n_site = 8
n_boson = 8
dense_hidden_units = [40, 40]
sample = 1000
t = 1.0
U = 1.0
n_step = 2000
BH_type = BHM

def main():
    date = datetime.datetime.now()
    print(
        '=====> START\n'
        f'date: {date}\n' \
        f'bosons: {n_boson}, sites: {n_site}\n' \
        f'hopping interaction: {t}, on-site interaction: {U}' \
        #f'nearest neighbor interaction: {V}' \
        '\n' \
        )

    # initialize
    net = Network(n_site, n_boson, dense_hidden_units)
    print(net.summary()) # for debug
    state = SampledState(sample, n_site, n_boson, net)
    state.init_thermalize(net)
    print('=====> Initialize Completed')
    print(f'Initialize state: \n {state.num}') # for debug

    # training
    energy_history = np.zeros(n_step)
    for step in range(n_step):
        for i in range(32):
            state.try_flip(net)
        eloc = local_energy(sample, n_site, t, U, net, state)
        net.fit(state.num, eloc)

        eloc_mean = eloc.mean()
        energy_history[step] = eloc_mean

    # save files
    # (デバッグ中)utils.csv_write_energy_history(date, n_boson, n_site, t, U, hidden_units, )
    # (デバッグ中)utils.plot_energy_history(date, n_site, n_boson, t, U, dense_hidden_units, n_step, energy_history)

    import matplotlib.pyplot as plt
    fig2 = plt.figure()
    ax2 = fig2.add_subplot()
    ax2.set_title('overall'), ax2.set_xlabel('number of updates'), ax2.set_ylabel('eloc_mean /J')
    ax2.plot(
        np.arange(n_step),
        energy_history,
    )
    fig2.savefig("tmp_data/" \
        + "BHM" \
        + "_site{:>03d}_bosons{:>03d}_t{}_U{}".format(n_site, n_boson, t, U) \
        + '_dense{}'.format(dense_hidden_units) \
        + "_{0:%Y}{0:%m}{0:%d}_{0:%H}{0:%M}{0:%S}".format(date) \
        + '.png'
    )
    print('=====> END\n')

if __name__ == '__main__':
    main()