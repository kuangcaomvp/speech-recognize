#!/usr/bin/env python


import numpy as np

class CTCPrefixScore(object):
    """
    百度搜：
    ‘HYBRID CTC/ATTENTION ARCHITECTURE FOR END-TO-END SPEECH RECOGNITION’
    """
    def __init__(self, x, blank, eos):
        self.logzero = -10000000000.0
        self.blank = blank
        self.eos = eos
        self.input_length = len(x)
        self.x = x

    def initial_state(self):
        r = np.full((self.input_length, 2), self.logzero, dtype=np.float32)
        a = self.x[:, self.blank]
        r[:, 1] = np.cumsum(a, axis=0)
        return r


    def __call__(self, y, cs, r_prev):
        shape = y.shape
        batch = shape[0]
        output_length = shape[1] - 1  # ignore sos
        b_x = np.tile(self.x[np.newaxis, :, :], (batch, 1, 1))
        cs_e = np.expand_dims(cs, 1)
        xs = b_x[np.arange(0, cs.shape[0])[:, np.newaxis, np.newaxis], :, cs_e]
        xs = np.squeeze(xs, 1)
        xs = np.transpose(xs, (0, 2, 1))

        r = np.full((batch, self.input_length, 2, cs.shape[1]), self.logzero, dtype=np.float32)
        if output_length == 0:
            r[:, 0, 0] = xs[:, 0]

        r_sum = np.logaddexp(r_prev[:, :, 0], r_prev[:, :, 1])
        last = y[:, -1]
        log_phi = np.tile(r_sum[:, :, np.newaxis], (1, 1, cs.shape[1]))

        for i in range(batch):
            if output_length > 0 and last[i] in cs[i]:
                log_phi[i, :, list(cs[i]).index(last[i])] = r_prev[i, :, 1]

        start = max(output_length, 1)
        log_psi = r[:, start - 1, 0]
        for t in range(start, self.input_length):
            r[:, t, 0] = np.logaddexp(r[:, t - 1, 0], log_phi[:, t - 1]) + xs[:, t]
            r[:, t, 1] = np.logaddexp(r[:, t - 1, 0], r[:, t - 1, 1]) + b_x[:, t, self.blank][:, np.newaxis]
            log_psi = np.logaddexp(log_psi, log_phi[:, t - 1] + xs[:, t])

        eos_pos = np.where(cs == self.eos)
        if len(eos_pos[0]) > 0:
            log_psi[eos_pos[0], eos_pos[1]] = r_sum[eos_pos[0], -1]

        return log_psi, np.rollaxis(r, 3, 1)
