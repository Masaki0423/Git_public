import numpy as np

class SampledState:
    num: None
    thermalization_n = 1000

    def __init__(self, sample, n_site, n_boson, net):
        self.sample = sample
        self.n_site = n_site
        self.n_boson = n_boson

        self.__init_basis()

        self.log_psi = net.output(self.num)

    def __init_basis(self):
        self.num = np.zeros(self.sample * self.n_site)
        self.num = self.num.reshape(self.sample, self.n_site)

        for i in range(self.sample):
            for j in range(self.n_boson):
                self.num[i][j%self.n_site] += 1

    def try_flip(self, net):
        num_tmp = np.copy(self.num)
        for i in range(self.sample):
            site_1 = np.random.randint(self.n_site)
            site_2 = np.random.randint(self.n_site)
            if num_tmp[i][site_1] > 0 and site_1 != site_2:
                num_tmp[i][site_1] -= 1
                num_tmp[i][site_2] += 1

        log_psi_tmp = net.output(num_tmp)

        r = np.random.rand(self.sample).reshape(self.sample, 1)

        isflip = r < np.exp(2 * (log_psi_tmp - self.log_psi))

        for i in range(self.sample):
            if isflip[i]:
                self.num[i] = num_tmp[i]
                self.log_psi[i] = log_psi_tmp[i]

    def init_thermalize(self, net):
        for i in range(SampledState.thermalization_n):
            self.try_flip(net)
