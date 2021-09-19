import numpy as np

def local_energy(sample, n_site, t, U, net, state):

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

    log_psi2 = net.output(st).reshape(sample, n_site, 2)

    eloc = np.zeros(sample)
    for i in range(sample):
        onsite = hopping = 0
        for j in range(n_site):
            if state.num[i][j] > 0:
                hopping += -1.0 * t * np.sqrt(state.num[i][j]
                                       * (state.num[i][(j+1)%n_site] + 1)) \
                    * np.exp(log_psi2[i][j][0] - state.log_psi[i])
                hopping += -1.0 * t * np.sqrt(state.num[i][j]
                                       * (state.num[i][(j-1+n_site)%n_site] + 1)) \
                    * np.exp(log_psi2[i][j][1] - state.log_psi[i])
                onsite += 0.5 * U * state.num[i][j] * (state.num[i][j] - 1)
        eloc[i] = hopping + onsite
    eloc = eloc.reshape([sample, 1])
    return eloc