#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import datetime
import matplotlib.pyplot as plt
# from bosehubbard_exact_diagonalize import Hamiltonian_Exact_Diagonalize
import csv
from tensorflow.python.layers.core import dense
from tensorflow.python.layers.convolutional import conv1d

# for debug
# np.random.seed(0)
# tf.compat.v1.set_random_seed(0) # 現状tfのrandom_seedは使っていない

OPEN_BOUNDARY = 0
PERIODIC_BOUNDARY = 1

class Network:
    def __init__(self, _boundary_condition = OPEN_BOUNDARY):
        self.prepare_model(_boundary_condition)
        self.prepare_session()

    def prepare_model(self, _boundary_condition):
        """
        boundary_condition: 0 (open boundary), 1 (periodic boundary)
        """
        x = tf.compat.v1.placeholder(tf.float32, [None, n_site])
        if _boundary_condition == OPEN_BOUNDARY:
            u = tf.reshape(x, [-1, n_site, 1])
            for i in range(cnn_layer):
                u = conv1d(
                    inputs=u,
                    filters=n_channel[i],
                    kernel_size=filter_size[i],
                    padding="same",
                    activation=tf.nn.relu
                )
        elif _boundary_condition == PERIODIC_BOUNDARY:
            u = tf.reshape(x, [-1, n_site, 1])
            for i in range(cnn_layer):
                lower_pad = u[:, :(filter_size[i]-1), :]
                u_pbc = tf.concat([u, lower_pad], axis=1)
                u = conv1d(
                    inputs=u_pbc,
                    filters=n_channel[i],
                    kernel_size=filter_size[i],
                    padding="valid",
                    activation=tf.nn.relu
                )
        u_flat = tf.layers.flatten(u)
        output = dense(
            inputs=u_flat,
            units=1,
            use_bias=False,
            activation=None
        )

        eloc = tf.compat.v1.placeholder(tf.float32, [None, 1])
        ene = tf.reduce_mean(eloc)
        loss = tf.reduce_sum(output * (eloc - ene))
        train_step = tf.compat.v1.train.AdamOptimizer().minimize(loss)

        self.x, self.output = x, output
        self.eloc, self.ene, self.loss = eloc, ene, loss
        self.train_step = train_step

    def prepare_session(self):
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())
        self.sess = sess

    def forward(self, num):
        return self.sess.run(self.output, feed_dict={self.x:num}).ravel()

    def optimize(self, state, eloc):
        eloc = eloc.reshape(sample, 1)
        self.sess.run(self.train_step,
                      feed_dict={self.x:state.num, self.eloc:eloc})


class SampledState:
    num: None
    thermalization_n = 1024

    def __init__(self, net):
        self.num = np.zeros(sample * n_site)
        self.num = self.num.reshape(sample, n_site)
        for i in range(sample):
            for j in range(n_boson):
                self.num[i][j%n_site] += 1
        self.lnpsi = net.forward(self.num)

    def try_flip(self, net):
        num_tmp = np.copy(self.num)
        for i in range(sample):
            site_1 = np.random.randint(n_site)
            site_2 = np.random.randint(n_site)
            if num_tmp[i][site_1] > 0 and site_1 != site_2:
                num_tmp[i][site_1] -= 1
                num_tmp[i][site_2] += 1
        lnpsi_tmp = net.forward(num_tmp)
        r = np.random.rand(sample)
        isflip = r < np.exp(2 * (lnpsi_tmp - self.lnpsi))
        for i in range(sample):
            if isflip[i]:
                self.num[i] = num_tmp[i]
                self.lnpsi[i] = lnpsi_tmp[i]

    def thermalize(self, net):
        for i in range(SampledState.thermalization_n):
            self.try_flip(net)

#-----------------------------------

def LocalEnergy(net, state):
    st = np.zeros((sample, n_site, 2, n_site))
    st += state.num.reshape(sample, 1, 1, n_site)
    for i in range(sample):
        for j in range(n_site):
            if state.num[i][j] > 0:
                st[i][j][0][j] -= 1
                st[i][j][0][(j+1)%n_site] += 1
                st[i][j][1][j] -= 1
                st[i][j][1][(j-1+n_site)%n_site] += 1
    st = st.reshape(sample * n_site * 2, n_site)
    lnpsi2 = net.forward(st).reshape(sample, n_site, 2)
    eloc = np.zeros(sample)
    for i in range(sample):
        onsite = hopping = nearest_neighbor = 0
        for j in range(n_site):
            if state.num[i][j] > 0:
                hopping += -1.0 * t * np.sqrt(state.num[i][j]
                                       * (state.num[i][(j+1)%n_site] + 1)) \
                    * np.exp(lnpsi2[i][j][0] - state.lnpsi[i])
                hopping += -1.0 * t * np.sqrt(state.num[i][j]
                                       * (state.num[i][(j-1+n_site)%n_site] + 1)) \
                    * np.exp(lnpsi2[i][j][1] - state.lnpsi[i])
                onsite += 0.5 * U * state.num[i][j] * (state.num[i][j] - 1)
                nearest_neighbor += 0.5 * V * state.num[i][j] * state.num[i][(j+1)%n_site]
                nearest_neighbor += 0.5 * V * state.num[i][j] * state.num[i][(j-1+n_site)%n_site]
        eloc[i] = hopping + onsite + nearest_neighbor
    return eloc

def relativization(num, date):
    num_mean = np.zeros(n_site)
    for i in range(sample):
        for j in range(n_site):
            num_mean[j] += num[i][j]
    num_mean = num_mean / sample
    fig3 = plt.figure()
    ax3 = fig3.add_subplot()
    ax3.set_title("relativization")
    ax3.set_xlabel("sites")
    ax3.set_ylabel("number of bosons")
    ax3.plot(num_mean)
    print(f"num_mean: {num_mean}")
    fig3.savefig("tmp_data/"
        + "extended-BHM" \
        + "_{0:%Y}{0:%m}{0:%d}_{0:%H}{0:%M}{0:%S}".format(date) \
        + "_site{:>03d}_bosons{:>03d}_t{}_U{}_V{}".format(n_site, n_boson, t, U, V) \
        + "_convlayers{}_filter{}_channels{}_boundary{}".format(cnn_layer, filter_size, n_channel, boundary_condition) \
        + "_basis_mean.png"
    )

# -------------- main -----------------
def main():
    date = datetime.datetime.now()
    print(
        "{}\n"
        "bosons: {}, sites: {}\n"
        "hopping interaction: {}, on-site interaction: {}, nearest neighbor interaction: {}\n"
        # "exact_energy_ground_state ==> {}\n"
        "------> START\n"
        .format(date, n_boson, n_site, t, U, V,
        # energy_ground_state,
            )
        )

    net = Network(_boundary_condition=boundary_condition)
    state = SampledState(net)
    state.thermalize(net)
    datafile = "tmp_data/" \
        + "extended-BHM" \
        + "_{0:%Y}{0:%m}{0:%d}_{0:%H}{0:%M}{0:%S}".format(date) \
        + "_site{:>03d}_bosons{:>03d}_t{}_U{}_V{}".format(n_site, n_boson, t, U, V) \
        + "_convlayers{}_filter{}_channels{}_boundary{}.csv".format(cnn_layer, filter_size, n_channel, boundary_condition)
    fcsv_evlv = open(datafile, "w", newline="")
    csvwriter = csv.writer(fcsv_evlv, lineterminator="\n")
    csvwriter.writerow(["step", "energy"])

    energy_history = np.zeros(n_step)
    for step in range(n_step):
        for i in range(32):
            state.try_flip(net)
        eloc = LocalEnergy(net, state)
        net.optimize(state, eloc)

        energy_ave = eloc.mean()
        energy_history[step] = energy_ave
        # for debug
        print("step: {:>6d} Eave: {:7.5f}".format(step+1, energy_ave, flush=True))

        fcsv_evlv.flush()
        csvwriter.writerow([
            "{:>6d}".format(step+1),
            "{:>+10.7f}".format(energy_ave)
        ])

    fig2 = plt.figure(figsize = (6.4, 4.8)) # default_size(6.4, 4.8)
    ax2 = fig2.add_subplot()
    ax2.set_title("overall")
    ax2.set_xlabel("number of updates")
    ax2.set_ylabel("Eave /J")
    ax2.plot(
        np.arange(n_step),
        energy_history,
    )
    fig2.savefig("tmp_data/" \
        + "extended-BHM" \
        + "_{0:%Y}{0:%m}{0:%d}_{0:%H}{0:%M}{0:%S}".format(date) \
        + "_site{:>03d}_bosons{:>03d}_t{}_U{}_V{}".format(n_site, n_boson, t, U, V) \
        + "_convlayers{}_filter{}_channels{}_boundary{}.png".format(cnn_layer, filter_size, n_channel, boundary_condition)

    )
    relativization(state.num, date)
    print("-----> FINISH\n")

# ----- ここまでが繰り返し
n_site = 8
n_boson = 8
sample = 1000
t = 1.0
U = 1.0
V = 1.0
n_step = 5
# hidden_units = 10
# n_hidden_layer = 1
cnn_layer = 3
n_channel = [3,3,3]
filter_size = [5,3,3]
boundary_condition = OPEN_BOUNDARY
i = 0
while i < 10:
    main()
    i += 1