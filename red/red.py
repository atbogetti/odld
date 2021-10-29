#!/usr/bin/env python
from h5py import File as H5File
import numpy as np
from mcbs import mcbs_ci, sequence_macro_flux_to_rate

UNIT_TIME = 0.002
TAU = 0.002*1000
CONCENTRATION = 1


class DurationCorrector(object):
    @staticmethod
    def from_kinetics_file(directh5, istate, fstate, dtau, n_iters=None):
        iter_slice = slice(n_iters)

        if isinstance(directh5, H5File):
            dataset = directh5['durations'][iter_slice]
        else:
            with H5File(directh5, 'r') as directh5:
                dataset = directh5['durations'][iter_slice]

        torf = np.logical_and(dataset['istate'] == istate, dataset['fstate'] == fstate)
        torf = np.logical_and(torf, dataset['weight'] > 0)

        durations = dataset['duration']
        weights = dataset['weight']

        weights[~torf] = 0.  # mask off irrelevant flux

        return DurationCorrector(durations, weights, dtau)

    def __init__(self, durations, weights, dtau, maxduration=None):
        self.weights = np.array(weights)
        self.durations = np.array(durations)
        self.dtau = dtau
        self._f_tilde = None
        self._f_int1 = None

        if maxduration is None:
            self.maxduration = self.durations.shape[0]
        else:
            self.maxduration = maxduration

        if dtau is None:
            all_durations = []
            all_durations.extend(durations)
            all_durations.extend(np.arange(maxduration))
            uniq_durations = np.unique(all_durations)  # unique sorts automatically
            self.dtau = np.min(np.diff(uniq_durations))

        self._build_map()

    @property
    def event_duration_histogram(self):
        return self._f_tilde

    @property
    def cumulative_event_duration_histogram(self):
        return self._f_int1

    def _build_map(self):
        weights = self.weights
        durations = self.durations
        maxduration = self.maxduration
        dtau = self.dtau

        taugrid = np.arange(0, maxduration, dtau, dtype=float)
        f_map = np.zeros(weights.shape, dtype=int) - 1
        for i, tau in enumerate(taugrid):
            matches = np.logical_and(durations >= tau, durations < tau + dtau)
            f_map[matches] = i 

        self.taugrid = taugrid
        self.f_map = f_map

    def correction(self, iters, freqs=None):
        r"""
        Return the correction factor

        __                              __  -1
        |  t=theta  tau=t                |
        |    |\      |\                  |
        |    |       | ~                 |
        |    |       | f(tau) dtau dt    |      * maxduration
        |   \|      \|                   |
        |   t=0    tau=0                 |
        |_                              _|

        where
            ~`                        ^
            f(tau) is proportional to f(tau)/(theta-tau), and is normalized to
                            ^
        integrate to 1, and f(tau) is sum of the weights of walkers with
        duration time tau.

        ---------
        Arguments
        ---------
        maxduration: the maximum duration time that could have been observed in
            the simulation, which is usually equal to the length of the
            simulation. This should be in units of tau.
        """

        if iters is None:
            iters = np.arange(len(self.weights))

        if freqs is None:
            freqs = np.ones(len(iters), dtype=float)

        maxduration = np.max(iters) + 1

        f_map = self.f_map[iters]
        weights = self.weights[iters]
        taugrid = self.taugrid#[self.taugrid < maxduration]

        weights *= freqs[:, None]

        dtau = self.dtau

        f_tilde = np.zeros(len(taugrid), dtype=float)
        for i, tau in enumerate(taugrid):
            if tau < maxduration:
                f_tilde[i] = weights[f_map == i].sum() / (maxduration - tau + 1)

        if f_tilde.sum() != 0:
            f_tilde /= (f_tilde.sum()*dtau)

        self._f_tilde = f_tilde
        # now integrate f_tilde twice
        # integral1[t/dtau] gives the integral of f_tilde(tau) dtau from 0 to t
        self._f_int1 = integral1 = np.zeros(f_tilde.shape)

        for i, tau in enumerate(taugrid):
            if i > 0 and tau < maxduration:
                integral1[i] = np.trapz(f_tilde[:i+1], taugrid[:i+1])

        integral2 = np.trapz(integral1, taugrid)

        if integral2 == 0:
            return 0.
        return maxduration/integral2


def get_raw_rates(directh5, istate, fstate, n_iters=None):
    rate_evol = directh5['rate_evolution'][slice(n_iters), istate, fstate]
    avg = rate_evol['expected']
    ciub = rate_evol['ci_ubound']
    cilb = rate_evol['ci_lbound']

    return avg, cilb, ciub


def calc_avg_rate(directh5_path, istate, fstate, **kwargs):
    """
    Return the raw or RED-corrected rate constant with the confidence interval.

    ---------
    Arguments
    ---------
    dt: timestep (ps)
    nstiter: duration of each iteration (number of steps)
    ntpr: report inteval (number of steps)

    """

    n_iters = kwargs.pop("n_iters", None)
    # tau = kwargs.pop("tau", TAU)
    conc = kwargs.pop("conc", CONCENTRATION)

    dt = kwargs.pop("dt", UNIT_TIME)
    ntpr = kwargs.pop("report_interval", 20)
    nstiter = kwargs.pop("n_steps_iter", 1000)
    callback = kwargs.pop("callback", None)

    red = kwargs.pop("red", False)

    if len(kwargs) > 0:
        raise ValueError("unparsed kwargs")

    dtau = float(ntpr) / nstiter
    tau = dt * nstiter
    dc = None

    with H5File(directh5_path, 'r') as directh5:
        if n_iters is None:
            n_iters = directh5['rate_evolution'].shape[0]

        rate_evol = directh5['rate_evolution'][n_iters-1, istate, fstate]
        r = rate_evol['expected']

        if red:
            dc = DurationCorrector.from_kinetics_file(directh5, istate, fstate, dtau, n_iters)

    if callback is not None:
        kw = {"correction": dc}
        callback(**kw)

    iters = np.arange(n_iters)

    correction = dc.correction(iters) if dc else 1.

    rate = r / (tau * conc)
    rate *= correction

    return rate


def calc_rates(directh5_path, istate, fstate, **kwargs):
    """
    Return the raw and RED-corrected rate constants vs. iterations. 
    This code is faster than calling calc_rate() iteratively

    ---------
    Arguments
    ---------
    dt: timestep (ps)
    nstiter: duration of each iteration (number of steps)
    ntpr: report inteval (number of steps)

    """

    n_iters = kwargs.pop("n_iters", None)
    # tau = kwargs.pop("tau", TAU)
    conc = kwargs.pop("conc", CONCENTRATION)

    dt = kwargs.pop("dt", UNIT_TIME)
    ntpr = kwargs.pop("report_interval", 20)
    nstiter = kwargs.pop("n_steps_iter", 1000)
    callback = kwargs.pop("callback", None)

    red = kwargs.pop("red", False)

    if len(kwargs) > 0:
        raise ValueError("unparsed kwargs")

    dtau = float(ntpr) / nstiter
    tau = dt * nstiter
    dc = None

    with H5File(directh5_path, 'r') as directh5:
        rate_evol, cilb, ciub = get_raw_rates(directh5, istate, fstate, n_iters)
        if n_iters is None:
            n_iters = len(rate_evol)
        if red:
            dc = DurationCorrector.from_kinetics_file(directh5, istate, fstate, dtau, n_iters)

    if callback is not None:
        kw = {"correction": dc}
        callback(**kw)

    raw_rates = np.zeros(n_iters)
    raw_ci_lbounds = np.zeros(n_iters)
    raw_ci_ubounds = np.zeros(n_iters)

    rates = np.zeros(n_iters)
    ci_lbounds = np.zeros(n_iters)
    ci_ubounds = np.zeros(n_iters)

    for i in range(n_iters):
        i_iter = i + 1
        print("\riter %d/%d (%3.0f%%)"%(i_iter, n_iters, i_iter*100.0/n_iters), end="")

        r = rate_evol[i]
        l = cilb[i]
        u = ciub[i]

        iters = np.arange(i_iter)

        correction = dc.correction(iters) if dc else 1.

        raw_rates[i] = r / (tau * conc)
        rates[i] = raw_rates[i] * correction

        raw_ci_lbounds[i] = l / (tau * conc)
        raw_ci_ubounds[i] = u / (tau * conc)

        ci_lbounds[i] = raw_ci_lbounds[i] * correction
        ci_ubounds[i] = raw_ci_ubounds[i] * correction

    print("\n")

    return rates, ci_lbounds, ci_ubounds


def calc_rate_ci(directh5_path, assignh5_path, istate, fstate, **kwargs):
    """
    Return the raw or RED-corrected rate constant with the confidence interval.

    ---------
    Arguments
    ---------
    dt: timestep (ps)
    nstiter: duration of each iteration (number of steps)
    ntpr: report inteval (number of steps)

    """

    n_iters = kwargs.pop("n_iters", None)
    # tau = kwargs.pop("tau", TAU)
    conc = kwargs.pop("conc", CONCENTRATION)

    dt = kwargs.pop("dt", UNIT_TIME)
    ntpr = kwargs.pop("report_interval", 20)
    nstiter = kwargs.pop("n_steps_iter", 1000)
    callback = kwargs.pop("callback", None)
    alpha = kwargs.pop("alpha", 0.05)
    red = kwargs.pop("red", False)
    bayesian = kwargs.pop("bayesian", False)

    if len(kwargs) > 0:
        for k in kwargs:
            print(k)
        raise ValueError("unparsed kwargs")

    dtau = float(ntpr) / nstiter
    tau = dt * nstiter
    dc = None

    with H5File(directh5_path, 'r') as directh5:
        cond_flux = directh5['conditional_fluxes'][slice(n_iters), istate, fstate]

        if n_iters is None:
            n_iters = len(cond_flux)

        if red:
            dc = DurationCorrector.from_kinetics_file(directh5, istate, fstate, dtau, n_iters)

    with H5File(assignh5_path, 'r') as f:
        pops = f['labeled_populations'][:]
        pops = pops.sum(axis=2)

    data = {'dataset': cond_flux[:n_iters], 'pops': pops}
    kwargs = {'istate': istate, 'jstate': fstate, 'corrector': dc}
    r, lb, ub, std, data = mcbs_ci(data, sequence_macro_flux_to_rate, alpha, n_iters, kwargs=kwargs, return_data=True, bayesian=bayesian)

    if callback is not None:
        kw = {"correction": dc,
              "bsdata": data / (tau * conc)}
        callback(**kw)

    avg_rate = r / (tau * conc)
    ci_lbound = lb / (tau * conc)
    ci_ubound = ub / (tau * conc)
    sterr = std / (tau * conc)

    return avg_rate, ci_lbound, ci_ubound, sterr


class RateCalculator:
    def __init__(self, directh5, istate, fstate, assignh5=None, **kwargs):
        n_iters = kwargs.pop("n_iters", None)
        # tau = kwargs.pop("tau", TAU)
        conc = kwargs.pop("conc", CONCENTRATION)

        dt = kwargs.pop("dt", UNIT_TIME)
        ntpr = kwargs.pop("report_interval", 20)
        nstiter = kwargs.pop("n_steps_iter", 1000)

        if len(kwargs) > 0:
            for k in kwargs:
                print(k)
            raise ValueError("unparsed kwargs")

        dtau = float(ntpr) / nstiter
        tau = dt * nstiter

        with H5File(directh5, 'r') as f:
            state_labels = {}
            for i, raw_label in enumerate(f['state_labels']):
                label = raw_label.decode() if isinstance(raw_label, bytes) else raw_label
                state_labels[label] = i
            if istate not in state_labels:
                raise ValueError(f"istate not found: {istate}, available options are {list(state_labels.keys())}")
            if fstate not in state_labels:
                raise ValueError(f"istate not found: {fstate}, available options are {list(state_labels.keys())}")
            istate = state_labels[istate]
            fstate = state_labels[fstate]
            cond_fluxes = f['conditional_fluxes'][slice(n_iters), istate, fstate]

        if assignh5 is not None:
            with H5File(assignh5, 'r') as f:
                pops = f['labeled_populations'][slice(n_iters)]
                pops = pops.sum(axis=2)
        else:
            pops = None

        self._dc = None
        self._pops = pops
        self._cond_fluxes = cond_fluxes
        self._conc = conc
        self._tau = tau
        self._dtau = dtau
        self._directh5 = directh5
        self._assignh5 = assignh5
        self._istate = istate
        self._fstate = fstate

    @property
    def conditional_fluxes(self):
        return self._cond_fluxes

    @property
    def populations(self):
        return self._pops

    @property
    def tau(self):
        return self._tau

    @property
    def dtau(self):
        return self._dtau

    @property
    def concentration(self):
        return self._conc

    @property
    def istate(self):
        return self._istate

    @property
    def fstate(self):
        return self._fstate

    @property
    def n_iters(self):
        return len(self.conditional_fluxes)

    def _get_corrector(self):
        if self._dc is None:
            with H5File(self._directh5, 'r') as f:
                self._dc = DurationCorrector.from_kinetics_file(f, self.istate, self.fstate, self.dtau, self.n_iters)

        return self._dc

    def calc_rate(self, i_iter=None, red=False, **kwargs):
        reboot = kwargs.pop('rebootstrap', False)
        bayesian = kwargs.pop("bayesian", False)
        alpha = kwargs.pop('alpha', 0.05)
        callback = kwargs.pop('callback', None)

        if i_iter is None:
            i_iter = self.n_iters

        if bayesian:
            reboot = True

        dc = self._get_corrector() if red else None
        bsdata = None
        if reboot:
            if self._pops is None:
                raise ValueError('assign.h5 needs to be assigned in order to re-bootstrap')

            data = {'dataset': self._cond_fluxes[:i_iter], 'pops': self._pops[:i_iter]}
            ci_kwargs = {'istate': self.istate, 'jstate': self.fstate, 'corrector': dc}
            r, lb, ub, std, data = mcbs_ci(data, sequence_macro_flux_to_rate, alpha, i_iter, kwargs=ci_kwargs, return_data=True, bayesian=bayesian)
            bsdata = data / (self._tau * self._conc)
        else:
            found = False
            with H5File(self._directh5, 'r') as f:
                for i in range(f['rate_evolution'].shape[0]):
                    rate_evol = f['rate_evolution'][i, self.istate, self.fstate]
                    start = rate_evol['iter_start']
                    stop = rate_evol['iter_stop']

                    if i_iter >= start and i_iter < stop:
                        r = rate_evol['expected']
                        ub = rate_evol['ci_ubound']
                        lb = rate_evol['ci_lbound']
                        found = True
                        break

                if not found:
                    self.log.error("Can't find rate evolution data for iteration %d!" % i_iter)

            if dc:
                iters = np.arange(i_iter)
                correction = dc.correction(iters)
                r *= correction
                ub *= correction
                lb *= correction

        rate = r / (self._tau * self._conc)
        ci_ub = ub / (self._tau * self._conc)
        ci_lb = lb / (self._tau * self._conc)

        if callback is not None:
            kw = {'correction': dc,
                  'bsdata': bsdata}
            callback(**kw)

        return rate, ci_lb, ci_ub

    def calc_rates(self, n_iters=None, **kwargs):
        if n_iters is None:
            n_iters = self.n_iters

        rates = np.zeros(n_iters)
        ci_lbounds = np.zeros(n_iters)
        ci_ubounds = np.zeros(n_iters)

        for i in range(n_iters):
            i_iter = i + 1
            print("\riter %d/%d (%3.0f%%)"%(i_iter, n_iters, i_iter*100.0/n_iters), end="")

            r, l, u = self.calc_rate(i_iter, **kwargs)

            rates[i] = r
            ci_lbounds[i] = l
            ci_ubounds[i] = u

        print("\n")

        return rates, ci_lbounds, ci_ubounds


if __name__ == "__main__":
    from time import time

    DATA = bs_data = bbs_data = None
    def debug(**kwargs):
        global DATA
        DATA = kwargs.pop("bsdata")

        # np.savez("duration_dist.npz", taugrid=taugrid, f_raw=f_raw)

    start = time()
    # path to the w_direct output files
    name = 'ANALYSIS/DEFAULT'
    # directh5path = '%s/direct.h5'%name
    # assignh5path = '%s/assign.h5'%name
    directh5path = 'ANALYSIS/DEFAULT/direct.h5'
    assignh5path = 'ANALYSIS/DEFAULT/assign.h5'
    # number of timepoints per iteration; since the last timepoint of one 
    # iteration overlaps with the first timepoint of the next, we typically
    # have an odd number here (like 11 or 21 or 51).

    # B2
    # unit_time = 0.002
    # n_steps_per_iter = 5000
    # n_steps_per_report = 1000
    # istate = 0; fstate = 1

    # BUTANOL
    # unit_time = 0.002
    # n_steps_per_iter = 1000
    # n_steps_per_report = 20
    # istate = 1; fstate = 0

    # CALBINDIN
    unit_time = 1
    n_steps_per_iter = 1
    n_steps_per_report = 1
    istate = 'start'; fstate = 'end'

    # # WCA
    # unit_time = 0.004300187452843778
    # n_steps_per_iter = 250
    # n_steps_per_report = 250
    # istate = 0; fstate = 1

    red = True
    strr = 'red' if red else 'raw'

    # buffer to hold the corrected rate values; 
    print('estimating %s rates...'%strr)
    rater = RateCalculator(directh5path, istate, fstate, dt=unit_time,
                           n_steps_iter=n_steps_per_iter,
                           report_interval=n_steps_per_report,
                           assignh5=assignh5path)

    rates, lb, ub = rater.calc_rates(red=red, callback=None)

    # rates, lb, ub = calc_rates(directh5path, istate, fstate, dt=unit_time, 
    #                            n_steps_iter=n_steps_per_iter, 
    #                            report_interval=n_steps_per_report,
    #                            red=red, callback=None)

#    print('bootstrapping rates...')
#    # avg_rate, ci_lb, ci_ub, sterr = calc_rate_ci(directh5path, assignh5path, istate, fstate, 
#    #                                              dt=unit_time, 
#    #                                              n_steps_iter=n_steps_per_iter, 
#    #                                              report_interval=n_steps_per_report,
#    #                                              red=red,
#    #                                              bayesian=False,
#    #                                              callback=debug)
#    avg_rate, ci_lb, ci_ub = rater.calc_rate(red=red, bayesian=False, rebootstrap=True, callback=debug)
#    bs_data = DATA
#
#    print('bayesian bootstrapping rates...')
#    # bavg_rate, bci_lb, bci_ub, bsterr = calc_rate_ci(directh5path, assignh5path, istate, fstate, 
#    #                                                  dt=unit_time, 
#    #                                                  n_steps_iter=n_steps_per_iter, 
#    #                                                  report_interval=n_steps_per_report,
#    #                                                  red=red,
#    #                                                  bayesian=True,
#    #                                                  callback=debug)
#    bavg_rate, bci_lb, bci_ub = rater.calc_rate(red=red, bayesian=True, rebootstrap=True, callback=debug)
#    bbs_data = DATA
    end = time()
    # CALBINDIN Rates calculated in 110.85s
    print('Rates calculated in %.2fs'%(end-start))

    np.savez('%s_%s_rates.npz'%(name, strr), expected=rates, ci_lb=lb, ci_ub=ub)
#
#    if bs_data is not None:
#        np.savez('%s_%s_rate_ci.npz'%(name, strr), expected=avg_rate, ci_lb=ci_lb, ci_ub=ci_ub, bs_data=bs_data)
#
#    if bbs_data is not None:
#        np.savez('%s_%s_rate_ci_bb.npz'%(name, strr), expected=bavg_rate, ci_lb=bci_lb, ci_ub=bci_ub, bs_data=bbs_data)
